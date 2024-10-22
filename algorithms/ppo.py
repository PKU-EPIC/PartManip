from utils import path2video
from utils.torch_jit_utils import *

from algorithms.algo_utils import RolloutStorage, ActorCritic, Normalization

import os
import time
from os.path import join as pjoin
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy 


class ppo:
    def __init__(self, vec_env, cfg, logger):
        # env infos
        self.vec_env = vec_env
        self.num_envs = cfg['num_envs']
        self.obs_mode = cfg['obs_mode']
        self.num_obs = vec_env.num_obs[self.obs_mode]
        self.num_actions = vec_env.num_actions
        self.max_episode_length = vec_env.max_episode_length
        self.default_succ_value = cfg['succ_value']

        # training
        self.model_cfg = cfg['model']
        self.max_iter = cfg['max_iterations']
        self.n_steps = cfg['n_steps']
        self.n_updates = cfg['n_updates']
        self.num_mini_batches = cfg['n_minibatches']
        self.device = cfg['device']

        # eval and save
        self.eval_round = cfg['eval_round']
        self.eval_freq = cfg['eval_frequence']
        self.save_freq = cfg['save_frequence']
        self.test_only = cfg['test_only']
        self.save_pose = cfg['save_pose']
        self.save_video = cfg['save_video']
        self.save_ckpt_dir = logger.save_ckpt_dir

        # learning rate
        self.lr_schedule =  cfg['lr_schedule']
        self.lr = cfg['lr']
        self.desired_kl = cfg['desired_kl']
        assert self.desired_kl > 0

        if self.lr_schedule != 'fixed' and self.lr_schedule != 'linear_decay' and self.lr_schedule != 'step_decay':
            raise NotImplementedError

        # parameters
        self.epsilon_clip =  cfg['epsilon_clip']
        self.gamma = cfg['gamma']
        self.lam = cfg['lam']
        
        # Tricks
        self.tricks = {}
        self.tricks_keys = ['mini_adv_norm', 'whole_adv_norm', 'use_state_norm', 'use_clipped_value_loss', 'use_grad_clip']
        for k in self.tricks_keys:
            self.tricks[k] =  cfg['tricks'][k]
        if self.tricks['use_grad_clip']:
            self.max_grad_norm = cfg['tricks']['max_grad_norm']
        if self.tricks['use_state_norm']:
            self.state_norm = Normalization(shape=self.num_obs, device=self.device)
            self.update_RMS = True

        # network
        self.actor_critic = ActorCritic(self.num_obs, self.num_actions, self.model_cfg).to(self.device) 
        self.storage = RolloutStorage(self.num_envs, self.n_steps, self.num_obs, self.num_actions, self.device, self.default_succ_value, self.tricks['whole_adv_norm'], cfg['sampler'])
        self.optimizer_actor = optim.Adam([{'params':self.actor_critic.actor.parameters()}, {'params':self.actor_critic.log_std}], lr=self.lr)
        self.optimizer_critic = optim.Adam(self.actor_critic.critic.parameters(), lr=self.lr)
        # Log
        self.logger = logger
        self.total_envsteps = 0
        self.total_time = 0
        self.curr_iter = 0
        # Resume from previous ckpt
        self.resume(cfg['resume'])

    def save(self, it):    
        os.makedirs(self.save_ckpt_dir, exist_ok=True)
        save_path = pjoin(self.save_ckpt_dir, f'model_{it}.pth')
        save_dict = {
            'iteration': it,
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'total_steps' : self.total_envsteps,
            'tricks': self.tricks,          
            'obs_mode': self.obs_mode,      # for dagger
            'model_cfg': self.model_cfg  # for dagger
        }
        if self.tricks['use_state_norm']:
            save_dict['state_running_ms'] = self.state_norm.running_ms.save()
        torch.save(save_dict, save_path)
        print(f'save ckpt to {save_path}!')
        return 

    def resume(self, ckpt_path):
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            print(f'load ckpt from {ckpt_path}!')
            assert os.path.exists(ckpt_path)
            ckpt_dict= torch.load(ckpt_path, map_location=self.device)
            self.actor_critic.load_state_dict(ckpt_dict["model_state_dict"])
            
            # # debug
            # cri_ckpt_path = '/home/jiayichen/RLManip/logs/ckpts/grasp_cube_ppo/reset5_seed9859/model_180000.pth'
            # cri_ckpt_dict= torch.load(cri_ckpt_path, map_location=self.device)
            # cri_dct = {}
            # for k in cri_ckpt_dict["model_state_dict"].keys():
            #     if 'critic' in k:
            #         kk = k.split('critic.')[-1]
            #         print(kk, k)
            #         cri_dct[kk] = cri_ckpt_dict["model_state_dict"][k]
            # self.actor_critic.critic.load_state_dict(cri_dct)
            # # debug

            self.optimizer_actor.load_state_dict(ckpt_dict["optimizer_actor"])
            self.optimizer_critic.load_state_dict(ckpt_dict["optimizer_critic"])
            self.curr_iter = ckpt_dict["iteration"]
            self.total_envsteps = ckpt_dict["total_steps"]

            # check tricks
            for k in self.tricks_keys:
                if self.tricks[k] != ckpt_dict['tricks'][k]:
                    print(f'WARNING: trick {k} is not consistent with ckpt! saved: {ckpt_dict[k]}, now: {self.tricks[k]}')
                    if k == 'use_state_norm':
                        print('this is not allowed')
                        exit(1)
            if self.tricks['use_state_norm']:
                self.state_norm.running_ms.load(ckpt_dict['state_running_ms'])
            assert self.obs_mode == ckpt_dict['obs_mode']
        return 

    def eval(self):
        self.actor_critic.eval()
        self.vec_env.train_test_flag = 'test'
        if self.test_only:
            self.log_dict = {}
        ep_infos = []

        with tqdm(total=self.eval_round) as pbar:
            pbar.set_description('Validating:')
            with torch.no_grad() :
                for r in range(self.eval_round):
                    save_dict_lst = []
                    curr_obs = self.vec_env.reset()[self.obs_mode]
                    tea_rew = []    # for dagger
                    save_ee_poses = []  # for real world exp
                    for i in range(self.max_episode_length) :
                        if self.tricks['use_state_norm']:   ### trick
                            curr_obs = self.state_norm(curr_obs, update=False)
                        
                        actions, value = self.actor_critic.act_cri(curr_obs)

                        if self.save_video:
                            save_image_path = pjoin(self.logger.save_video_dir,  f"Iter{self.curr_iter}", f"{i}.png")
                        else:
                            save_image_path = None 
                        next_obs, rews, _, infos = self.vec_env.step(actions, save_image_path=save_image_path)
                        # print(i, value[0], rews[0], actions[0])
                        # angle,axis = quat_to_angle_axis(self.vec_env.robot.tip_rb_tensor[0,3:7])
                        # save_ee_pose = torch.cat([self.vec_env.robot.tip_rb_tensor[0,:3], angle*axis, self.vec_env.gripper_length[0:1]]).cpu().numpy()
                        # print(save_ee_pose)
                        # save_ee_poses.append(save_ee_pose)
                        tea_rew.append(rews.cpu().numpy().mean())
                        # print(i, rews.min(),rews.argmin(), rews.mean())
                        infos['action_t'] = actions[:, :3].mean(dim=-1)
                        infos['action_r'] = actions[:, 3:6].mean(dim=-1)
                        infos['action_gripper'] = actions[:, -1]
                        infos['succ_rate'] = self.vec_env.success
                        ep_infos.append(deepcopy(infos))
                        if self.save_pose:
                            save_dict = self.vec_env.save_scene_pose(pjoin(self.logger.save_pose_dir, f"Iter{self.curr_iter}", f"{i}.npy"))
                            save_dict['state'] = curr_obs.cpu().numpy()
                            save_dict['action'] = actions.cpu().numpy()
                            save_dict_lst.append(deepcopy(save_dict))
                        curr_obs = next_obs[self.obs_mode]
                    # np.save('teacher_reward.npy', np.array(tea_rew))  # For Dagger!
                    # np.save('trajectory.npy', save_ee_poses)       # For real world exp
                    if self.save_pose:
                        print(ep_infos[-1]['obj_up_flag'].float().mean())
                        for i in range(self.max_episode_length) :
                            save_dict_lst[i]['success'] = ep_infos[-1]['obj_up_flag'].cpu().numpy()
                            np.save(pjoin(self.logger.save_pose_dir, f"Iter{self.curr_iter}", f"{i}.npy"), save_dict_lst[i])

                    if self.save_video:
                        save_image_path = pjoin(self.logger.save_video_dir,  f"Iter{self.curr_iter}")
                        path2video(save_image_path)

                    pbar.update(1)
        # log
        mode = 'Test' if self.test_only else 'Val'
        self.use_info_update_logdict(ep_infos, mode)   

        if self.log_dict[f'{mode}/succ_rate_max'] > 0.5 and self.update_RMS:
            self.update_RMS = False
        ep_infos.clear()
        return

    def run(self):
        if self.test_only:
            self.eval()
            self.logger.info(self.log_dict, self.curr_iter)
        else:
            curr_obs = self.vec_env.reset()[self.obs_mode]
            if self.tricks['use_state_norm']: ### trick
                curr_obs = self.state_norm(curr_obs, update=self.update_RMS)

            while self.curr_iter < self.max_iter:
                self.curr_iter += 1

                # intialization
                self.actor_critic.train()
                self.vec_env.train_test_flag = 'train'

                self.log_dict = {}
                ep_infos = []
                start = time.time()
                # Rollout
                for _ in range(self.n_steps): 

                    actions, actions_log_prob, values, mu, sigma = self.actor_critic.random_act_cri(curr_obs)
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    print('TrainIter: %d, SuccRate: %.3f, Finish: %d, Rew: %.3f, maxRew: %.3f' % (self.curr_iter, infos['succ_rate'].data, dones.int().sum(), self.vec_env.rew_buf.mean(), self.vec_env.rew_buf.max()))
                    self.storage.add_transitions(curr_obs, actions, rews, dones, self.vec_env.reset_succ, values, actions_log_prob, mu, sigma)

                    infos['action_t'] = actions[:, :3].abs().mean(dim=-1)
                    infos['action_r'] = actions[:, 3:6].abs().mean(dim=-1)
                    infos['action_gripper'] = actions[:, -1].abs()
                    infos['value_pred'] = values.squeeze(-1)

                    if self.tricks['use_state_norm']:
                        infos['RMS_state_mean'] = self.state_norm.running_ms.mean.mean().unsqueeze(0) 
                        infos['RMS_state_std'] = self.state_norm.running_ms.std.mean().unsqueeze(0) 
                        infos['RMS_state_1_std'] = self.state_norm.running_ms.std[:, :14].mean(dim=-1)
                        infos['RMS_state_2_std'] = self.state_norm.running_ms.std[:, 14:].mean(dim=-1)

                    curr_obs = next_obs[self.obs_mode]
                    if self.tricks['use_state_norm']: ### trick
                        curr_obs = self.state_norm(curr_obs, update=self.update_RMS)
                    ep_infos.append(deepcopy(infos))

                last_values = self.actor_critic.cri(curr_obs) 

                stop = time.time()
                collection_time = stop - start

                # mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step. NOTE self.storage.compute_returns will update advantage and self.update will use advantage
                start = time.time()
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                
                self.update(self.curr_iter)
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start

                # log
                self.total_envsteps += self.n_steps * self.vec_env.num_envs
                self.total_time += collection_time + learn_time
                action_std = self.actor_critic.log_std.exp()
                mean_std = self.actor_critic.log_std.exp().mean()
                fps = int(self.n_steps * self.vec_env.num_envs / (collection_time + learn_time))
                self.log_dict['Progress/total_steps'] = self.curr_iter
                self.log_dict['Progress/collection_time'] = collection_time
                self.log_dict['Progress/learn_time'] = learn_time
                self.log_dict['Progress/FPS'] = fps
                self.log_dict['Train/mean_action_noise_std'] = mean_std.item()
                self.log_dict['Train/mean_t_noise_std'] = action_std[:3].mean()
                self.log_dict['Train/mean_r_noise_std'] = action_std[3:-1].mean()
                self.log_dict['Train/mean_gripper_noise_std'] = action_std[-1]
                self.use_info_update_logdict(ep_infos, 'Train')            

                # evaluation and save. NOTE eval will update self.log_dict
                if self.curr_iter % self.eval_freq == 0:
                    self.eval()
                    # reset all envs after evaluation
                    curr_obs = self.vec_env.reset()[self.obs_mode]
                    if self.tricks['use_state_norm']: 
                        curr_obs = self.state_norm(curr_obs, update=self.update_RMS)
                if self.curr_iter % self.save_freq == 0:
                    self.save(self.curr_iter)
                self.logger.info(self.log_dict, self.curr_iter)

                # reset
                ep_infos.clear()
        return 

    def use_info_update_logdict(self, info_lst, mode):
        for key in info_lst[0]:
            assert len(info_lst[0][key].shape) == 1, f"{key}: {info_lst[0][key].shape}"
            for i, info in enumerate(info_lst):
                if i == 0:
                    all_info_train = info[key].unsqueeze(-1).float()
                else:
                    all_info_train = torch.cat((all_info_train, info[key].unsqueeze(-1).float()), dim=-1)
            self.log_dict[f'{mode}/{key}_mean'] = torch.mean(all_info_train)
            self.log_dict[f'{mode}/{key}_max'] = torch.mean(all_info_train.max(dim=-1)[0])
        return 

    def update(self, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl_mean = 0
        kl_max = 0
        count = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.n_updates):
            for indices in batch:
                obs_batch = self.storage.observations.view(-1, self.storage.observations.size(-1))[indices]  # (n_steps*n_env/n_minibatch, obs_dim)
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.update_act_cri(obs_batch, actions_batch)
                ###Trick 1 advantage normalization
                if self.tricks['mini_adv_norm']:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                # KL learning rate schedule
                kl = torch.sum(sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / 
                            (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)
                if kl_mean > kl_max:
                    kl_max = kl_mean
                if kl_mean > self.desired_kl:
                    continue
              
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Gradient step
                self.optimizer_actor.zero_grad()
                surrogate_loss.backward()

                if self.tricks['use_grad_clip']:   ###trick 7
                    nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.max_grad_norm)
                
                self.optimizer_actor.step()

                mean_surrogate_loss += surrogate_loss.item()
                mean_kl_mean += kl_mean
                count += 1

        for epoch in range(self.n_updates):
            for indices in batch:
                obs_batch = self.storage.observations.view(-1, self.storage.observations.size(-1))[indices]  # (n_steps*n_env/n_minibatch, obs_dim)
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_values_batch = self.storage.values.view(-1, 1)[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.update_act_cri(obs_batch, actions_batch)
               
                if self.tricks['use_clipped_value_loss']:
                    with torch.no_grad():
                        delta_value_clipped = (self.epsilon_clip*old_values_batch).abs().mean()
                        return_clipped = old_values_batch + (returns_batch - old_values_batch).clamp(-delta_value_clipped, delta_value_clipped)
                    value_loss = (value_batch - return_clipped).pow(2).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # Gradient step
                self.optimizer_critic.zero_grad()
                value_loss.backward()
                
                if self.tricks['use_grad_clip']:   ###trick 7
                    nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                mean_value_loss += value_loss.item()

        mean_value_loss /= self.n_updates * len(batch)
        mean_surrogate_loss /= count
        mean_kl_mean /= count

        if self.lr_schedule == 'linear_decay': 
            lr_now = max(self.lr * (1 - it / self.max_iter), 1e-5)
            for param_group in self.optimizer_actor.param_groups:
                param_group['lr'] = lr_now
        elif self.lr_schedule == 'step_decay': 
            if it > self.max_iter // 2:
                lr_now = 1e-5
            else:
                lr_now = self.lr
            for param_group in self.optimizer_actor.param_groups:
                param_group['lr'] = lr_now
        # log 
        self.log_dict['Train/value_gt_return_mean'] = self.storage.returns.mean()
        self.log_dict['Train/value_gt_return_max'] = self.storage.returns.max()
        self.log_dict['Train/learning_rate'] = self.optimizer_actor.param_groups[0]['lr']
        self.log_dict['Train/value_function_loss'] =  mean_value_loss
        self.log_dict['Train/surrogate_loss'] = mean_surrogate_loss
        self.log_dict['Train/kl'] = mean_kl_mean
        self.log_dict['Train/kl_max'] = kl_max 
        self.log_dict['Train/kl_update_count'] = count 

        return 
    
