import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import mjrl.samplers.trajectory_sampler as trajectory_sampler
import mjrl.samplers.batch_sampler as batch_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve
from mjrl.utils.replay_buffer import PathBuffer

# Import Algs
from mjrl.algos.npg_cg import NPG
from mjrl.algos.behavior_cloning import BC

from tpi.core.config import cfg


class LFA(NPG):
    def __init__(self, env, policy, baseline, value,
                 demo_paths=None,
                 normalized_step_size=0.01,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=None,
                 save_logs=False,
                 kl_dist=None,
                 use_eval_adv=False,
                 lam_0=1.0,  # demo coef
                 lam_1=0.95, # decay coef
                 lam_2=0.99,
                 lam_3=0.01,
                 dapg_baseline=False,
                 num_pc=1000,
                 use_cuda=True,
                 bc_finetune=True,
                 bc_agent=None,
                 priority=False,
                 ft_batchsize=2048,
                 ft_lr=3e-4,
                 ft_interval=1,
                 pre_emb=None
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.value = value
        self.kl_dist = kl_dist if kl_dist is not None else 0.5*normalized_step_size
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.demo_paths = demo_paths
        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.lam_3 = lam_3
        self.num_pc = num_pc
        self.dapg_baseline = dapg_baseline
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.bc_finetune = bc_finetune
        self.bc_agent = bc_agent
        self.priority = priority
        self.ft_batchsize = ft_batchsize
        self.ft_lr = ft_lr
        self.ft_interval = ft_interval


        if pre_emb is not None:
            # set pre_emb to cuda
            for key in pre_emb:
                pre_emb[key] = pre_emb[key].cuda()
            self.policy.set_embedding(pre_emb)

        self.buffer = PathBuffer()

        self.iter_count = 0.0
        self.use_eval_adv = use_eval_adv
        if save_logs: self.logger = DataLog()

        for demo_path in demo_paths:
            demo_obs = demo_path['observations']
            obj_names = np.array([[self.policy.model.get_idx_from_emb(item[-self.num_pc*3:])] for item in demo_obs])
            demo_obs_with_key = np.concatenate((demo_obs[:, :-self.num_pc*3], obj_names), axis=1)
            demo_path['observations'] = demo_obs_with_key


    def compute_advantage(self, paths, demo_paths, gamma=0.995):
        for path in demo_paths:
            path['rewards'] = np.zeros(path['observations'].shape[0])
            path['baseline'] = self.baseline.predict(path)
            path['returns'] = self.value.predict(path)
            advantages = path['returns'] - path['baseline']
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            path['advantages'] = advantages + 1
        # fit value
        error_before, error_after = self.value.fit(paths, return_errors=True)



    def CPI_surrogate(self, observations, actions, advantages):
        adv_var = torch.from_numpy(advantages).float().to(self.device)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*adv_var)
        return surr

    def flat_vpg(self, observations, actions, advantages):
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).cpu().data.numpy() for g in vpg_grad])
        return vpg_grad


    def HVP(self, observations, actions, vector, regu_coef=None):
        regu_coef = self.FIM_invert_args['damping'] if regu_coef is None else regu_coef
        vec = torch.from_numpy(vector).float().to(self.device)
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            rand_idx = np.random.choice(num_samples, size=int(self.hvp_subsample*num_samples))
            obs = observations[rand_idx]
            act = actions[rand_idx]
        else:
            obs = observations
            act = actions
        old_dist_info = self.policy.old_dist_info(obs, act)
        new_dist_info = self.policy.new_dist_info(obs, act)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        grad_fo = torch.autograd.grad(mean_kl, self.policy.trainable_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
        h = torch.sum(flat_grad*vec)
        hvp = torch.autograd.grad(h, self.policy.trainable_params)
        hvp_flat = np.concatenate([g.contiguous().view(-1).cpu().data.numpy() for g in hvp])
        return hvp_flat + regu_coef*vector



    def train_from_paths(self, paths):
        self.buffer.add_data(paths)

        if not self.policy.has_frozen_pointnet:
            self.policy.freeze_pointnet()
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        if self.demo_paths is not None and self.lam_0 > 0.0:
            len_paths = len(self.demo_paths)
            sampled_demo_paths = [self.demo_paths[i] for i in np.random.choice(len_paths, 100)]
            demo_traj_len = [path['observations'].shape[0] for path in sampled_demo_paths]
            self.compute_advantage(paths, sampled_demo_paths)
            eval_adv = np.concatenate([path['advantages'] for path in sampled_demo_paths])
            test_scores = [self.policy.new_dist_info(path['observations'], path['actions'])[0].cpu().data.numpy().mean().ravel()[0] for path in sampled_demo_paths] 
            # normalize
            test_scores = [item - min(test_scores) for item in test_scores]
            test_scores = [1 - (item / max(test_scores)) for item in test_scores]
            weight = np.concatenate([np.ones(demo_traj_len[i])*test_scores[i] for i in range(len(demo_traj_len))])

            demo_obs = np.concatenate([path["observations"] for path in sampled_demo_paths])            
            demo_act = np.concatenate([path["actions"] for path in sampled_demo_paths])
            #demo_adv = self.lam_0 * (self.lam_1 ** self.iter_count) * np.ones(demo_obs.shape[0])
            discount_factor = (self.lam_1 ** self.iter_count)
            demo_adv = self.lam_0 * discount_factor * weight
            if self.use_eval_adv:
                demo_adv += self.lam_3 * (1 - (self.lam_2 ** self.iter_count)) * eval_adv
            if self.dapg_baseline:
                demo_adv = self.lam_0 * discount_factor * np.ones(demo_obs.shape[0])
            self.iter_count += 1
            # concatenate all
            all_obs = np.concatenate([observations, demo_obs])
            all_act = np.concatenate([actions, demo_act])
            print(f'adv: {np.mean(advantages)}, {np.std(advantages)}, {np.max(advantages)}, {np.min(advantages)}')
            print(f'demo adv: {np.mean(eval_adv)}, {np.std(eval_adv)}, {np.max(eval_adv)}, {np.min(eval_adv)}')
            all_adv = cfg.LFA_ADV_W * np.concatenate([advantages/(np.std(advantages) + 1e-8), demo_adv])
        else:
            all_obs = observations
            all_act = actions
            all_adv = advantages

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # finetune pointnet with collected samples
        if self.bc_finetune and int(self.iter_count) % self.ft_interval == 0:
            self.bc_agent.change_parameters(epochs=1, lr=self.ft_lr, batch_size=self.ft_batchsize)
            if self.priority:
                self.bc_agent.train(random_embedding=False, transform_from_idx=True, \
                                    expert_paths=self.buffer.get_data(), finetune=False)

            else:
                self.bc_agent.train(random_embedding=False, transform_from_idx=True, \
                                    expert_paths=paths, finetune=False)
        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).cpu().data.numpy().ravel()[0]
        print(f'surr before: {surr_before}')
        # LFA
        ts = timer.time()
        sample_coef = all_adv.shape[0]/advantages.shape[0]
        dapg_grad = sample_coef*self.flat_vpg(all_obs, all_act, all_adv)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, dapg_grad, x_0=dapg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        n_step_size = 2.0*self.kl_dist
        alpha = np.sqrt(np.abs(n_step_size / (np.dot(dapg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        shs = 0.5 * (npg_grad * hvp(npg_grad)).sum(0, keepdims=True)
        lm = np.sqrt(shs / 1e-2)
        full_step = npg_grad / lm[0]
        grads = torch.autograd.grad(self.CPI_surrogate(all_obs, all_act, all_adv), self.policy.trainable_params)
        loss_grad = torch.cat([grad.view(-1)for grad in grads]).cpu().detach().numpy()
        neggdotstepdir = (loss_grad * npg_grad).sum(0, keepdims=True)
        curr_params = self.policy.get_param_values()
        alpha = 1 # new implementation
        for k in range(10):
            new_params = curr_params + alpha * full_step
            self.policy.set_param_values(new_params, set_new=True, set_old=False)
            surr_after = self.CPI_surrogate(observations, actions, advantages).cpu().data.numpy().ravel()[0]
            kl_dist = self.kl_old_new(observations, actions).cpu().data.numpy().ravel()[0]
            
            actual_improve = (surr_after - surr_before)
            expected_improve = neggdotstepdir / lm[0] * alpha
            ratio = actual_improve / expected_improve
            print(f'ratio: {ratio}, lm: {lm}')
            
            if ratio.item() > .1 and actual_improve > 0:
                break
            else:
                alpha = 0.5 * alpha
                print('step size too high. backtracking. | kl = %f | suff diff = %f' % \
                        (kl_dist, surr_after-surr_before))
        
            if k == 9:
                alpha = 0

        new_params = curr_params + alpha * full_step
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).cpu().data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).cpu().data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass
        

        return base_stats
