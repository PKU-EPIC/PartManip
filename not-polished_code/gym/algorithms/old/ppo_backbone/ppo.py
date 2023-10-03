
# 0708 work version

def debug_print(object_name, object, only_shape = False, with_shape = False):
    if only_shape:
        print(f"debug: {object_name}", object.shape)
    elif with_shape:
        print(f"debug: {object_name}", object, object.shape)
    else:
        print(f"debug: {object_name}", object)

import os
import time
import ipdb
from gym.spaces import Space, Box

import numpy as np
import statistics
from collections import deque
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithms.ppo import RolloutStorage
from algorithms.ppo import RolloutStoragePC
from .module import Normalization, RewardScaling
from ..ppo_utils.io_util import load, save
from ..ppo_utils.misc_util import lr_decay
from ..ppo_utils.log_util import log, log_test

class PPO:
    def __init__(self,
                 vec_env,
                 actor_critic_class,
                #  use_pc,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 max_lr=1e-3,
                 min_lr=1e-7,
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 eval_round=1,
                 eval_freq = 50,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False,
                 
                 max_iterations = 50000,
                 wandb_writer = None,
                 checkpoint_path = ' ',

                 use_adv_norm = True,          ### trick 1
                 adv_norm_epsilon= 1e-8,
                 use_state_norm= False,         ### trick 2
                 use_reward_norm= False,        ### trick 3
                 use_reward_scaling= False,     ### trick 4
                 learning_rate_decay = False,   ### trick 6
                 use_grad_clip = True,          ###trick 7
                 use_orthogonal_init= True,     ### trick 8
                 adam_epsilon = 1e-8,           ### trick 9
                 ):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.use_pc = model_cfg["use_pc"]
        if self.use_pc:
            self.pc_space = vec_env.pc_space
        self.train_env_num = vec_env.task.task_meta["train_env_num"] 
        self.valIntra_env_num = vec_env.task.task_meta["valIntra_env_num"]
        self.valInter_env_num = vec_env.task.task_meta["valInter_env_num"]
        self.val_env_num = self.valIntra_env_num + self.valInter_env_num
        self.max_episode_length = vec_env.task.task_meta["max_episode_length"]
        self.eval_round = eval_round
        self.eval_freq = eval_freq

        self.device = device
        self.asymmetric = asymmetric

        self.desired_kl = desired_kl
        self.lr_upper = float(max_lr)
        self.lr_lower = float(min_lr)
        self.schedule = schedule
        self.step_size = learning_rate

        model_cfg["task_meta"].update(vec_env.task.task_meta)


        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.max_iterations = max_iterations

        # PPO components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, asymmetric=asymmetric)
        
        self.actor_critic.to(self.device)
        self.is_testing = is_testing
        if not self.is_testing: 
            if not self.use_pc:
                self.storage = RolloutStorage(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                                        self.state_space.shape, self.action_space.shape, self.device, sampler)
            else:
                self.storage = RolloutStoragePC(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                                        self.state_space.shape, self.action_space.shape, self.pc_space.shape, self.device, sampler)

            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps= adam_epsilon)   # , weight_decay=float(self.weight_decay), trick 9

        
        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10) 
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.apply_reset = apply_reset
        self.exp_name = self.vec_env.task.env_name
        self.wandb_writer = wandb_writer
        self.checkpoint_path = checkpoint_path
        if self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed':
            self.order_part_pos = 10
        elif self.vec_env.task.cfg["env"]["robotName"] == 'FrankaSliderXYZ':
            self.order_part_pos = 16
        else:
            self.order_part_pos = 14
        
        # Trick
        self.use_adv_norm = use_adv_norm                    ###trick 1
        self.adv_norm_epsilon = adv_norm_epsilon
        self.use_state_norm = use_state_norm                ###trick 2 
        self.state_norm = Normalization(shape = (self.train_env_num,)+self.observation_space.shape)
        self.use_reward_norm = use_reward_norm              ###trick 3
        self.reward_norm = Normalization(shape=self.train_env_num)
        self.use_reward_scaling = use_reward_scaling        ###trick 4
        self.reward_scaling = RewardScaling(shape =self.train_env_num, gamma= self.gamma)
        self.learning_rate_decay = learning_rate_decay      ###trick 6
        self.use_grad_clip = use_grad_clip                  ###trick 7
        self.use_orthogonal_init = use_orthogonal_init      ###trick 8 
        
    def test(self, path):
        #root_path, model_name = os.path.split(path)
        #self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        #self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        #self.actor_critic.eval()
        #if self.vec_env.task.task_meta["need_update"] :
        #    self.vec_env.task.load(path, self.current_learning_iteration)
        #    self.vec_env.task.eval()
        assert os.path.exists(self.checkpoint_path)
        checkpoint_dict= torch.load(self.checkpoint_path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
        self.current_learning_iteration = checkpoint_dict["iteration"]
        self.actor_critic.eval()

    def load(self, path):

        #root_path, model_name = os.path.split(path)
        #self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        #self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        #self.actor_critic.train()
        #if self.vec_env.task.task_meta["need_update"] :
        #    self.vec_env.task.load(root_path, self.current_learning_iteration)
        #    self.vec_env.task.train()
        assert os.path.exists(self.checkpoint_path)
        checkpoint_dict= torch.load(self.checkpoint_path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        self.current_learning_iteration = checkpoint_dict["iteration"]
        self.tot_timesteps = checkpoint_dict["total_steps"]

    def save(self, path, it):    
        torch.save({
                    'iteration': it + 1,
                    'model_state_dict': self.actor_critic.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'total_steps' : self.tot_timesteps,
                    }, path)
    
    def lr_decay(self, total_steps):
        lr_now = self.step_size * (1 - total_steps / self.max_iterations)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def eval(self, it) :

        self.vec_env.task.eval()
        self.actor_critic.eval()
        current_obs = self.vec_env.reset()
        
        total_reward = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num), device=self.device)
        total_success = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num, self.eval_round), device=self.device)
        
        with tqdm(total=self.eval_round) as pbar:
            pbar.set_description('Validating:')
            with torch.no_grad() :
                for r in range(self.eval_round) :
                    for i in range(self.max_episode_length) :
                        if self.apply_reset:
                            current_obs = self.vec_env.reset()
                        if self.use_state_norm and not self.use_pc : ### trick 2
                            current_obs = self.state_norm(current_obs)
                        
                        actions = self.actor_critic.act_inference(current_obs)
                        next_obs, rews, dones, infos = self.vec_env.step(actions)
                        # next_obs_clouds, next_obs_states, rews, dones, infos = self.vec_env.step(actions)
                        current_obs= next_obs.copy()
                      
                        total_reward += rews.to(self.device)
                        total_success[:, r] = torch.logical_or(infos["successes"].to(self.device), total_success[:, r])
                        # if infos["successes"].item() != 0 :
                        #     print("WIN")
                    pbar.update(1)
        
        train_reward = total_reward[:self.train_env_num].mean() / self.max_episode_length / self.eval_round
        valIntra_reward = total_reward[self.train_env_num:self.valIntra_env_num + self.train_env_num].mean() / self.max_episode_length / self.eval_round
        valInter_reward = total_reward[self.valIntra_env_num + self.train_env_num:].mean() / self.max_episode_length / self.eval_round
        train_success = total_success[:self.train_env_num].float().mean()
        valIntra_success = total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float().mean()
        valInter_success = total_success[self.valIntra_env_num + self.train_env_num:].float().mean()

        train_reward = train_reward.cpu().item()
        valIntra_reward = valIntra_reward.cpu().item()
        valInter_reward = valInter_reward.cpu().item()
        train_success = train_success.cpu().item()
        valIntra_success = valIntra_success.cpu().item()
        valInter_success = valInter_success.cpu().item()

        #if self.is_testing:
        print("Training set average reward:     ", train_reward)
        print("ValIntra set average reward:      ", valIntra_reward)
        print("ValInter set average reward:      ", valInter_reward)
        print("Training set average success:    ", train_success)
        print("ValIntra set average success:     ", valIntra_success)
        print("ValInter set average success:     ", valInter_success)
        print(torch.mean(total_success[:self.train_env_num].float(), dim = 0))
        print(torch.mean(total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float(), dim = 0))
        print(torch.mean(total_success[self.valIntra_env_num + self.train_env_num:].float(), dim = 0))
        print(torch.mean(total_success[:self.train_env_num].float(), dim = 1))
        print(torch.mean(total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float(), dim = 1))
        print(torch.mean(total_success[self.valIntra_env_num + self.train_env_num:].float(), dim = 1))
        #print("Training set success list:")
        #for x in total_success[:self.train_env_num] / self.eval_round :
        #    print(x.cpu().item(), end=' ')

        #print("\n\nTesting set success list:")
        #for x in total_success[self.train_env_num:] / self.eval_round :
            #print(x.cpu().item(), end=' ')
        
        #print('\n')

        self.writer.add_scalar('Test/' + 'TestSuccessRate/TrainSet', train_success, it)
        self.writer.add_scalar('Test/' + 'TestSuccessRate/ValIntraSet', valIntra_success, it)
        self.writer.add_scalar('Test/' + 'TestSuccessRate/ValInterSet', valInter_success, it)
        self.writer.add_scalar('Test/' + 'TestReward/TrainSet', train_reward, it)
        self.writer.add_scalar('Test/' + 'TestReward/ValIntraSet', valIntra_reward, it)
        self.writer.add_scalar('Test/' + 'TestReward/ValInterSet', valInter_reward, it)
        
        self.vec_env.task.train()
        self.actor_critic.train()
        return (train_reward, valIntra_reward, valInter_reward,  train_success, valIntra_success, valInter_success)

    def run(self, num_learning_iterations, log_interval=1):
        
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()   # size (16, 0) ?
        if self.is_testing:

            self.vec_env.task.eval()

            _ = self.eval(self.current_learning_iteration)

        else:

            #if self.checkpoint > 0:   #load checkpoint 
            #    assert os.path.exists(self.checkpoint_path)
            #    checkpoint_dict= torch.load(self.checkpoint_path, map_location=self.device)
            #    self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
            #    self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            #    self.current_learning_iteration = checkpoint_dict["iteration"]
            #    self.tot_timesteps = checkpoint_dict["total_steps"]
            self.vec_env.task.train()

            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            success_rate = []
            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                task_info = {}  
                #print(current_states)
                #if it % self.eval_interval == 0 and self.val_env_num>0:
                #    self.eval(it)

                # Rollout
                for _ in range(self.num_transitions_per_env): 

                    if self.use_state_norm : ### trick 2
                        current_obs = self.state_norm(current_obs)
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        #current_states = self.vec_env.get_state()

                    if self.valIntra_env_num or self.valInter_env_num :
                        if not self.use_pc:
                            current_train_obs = current_obs[:self.train_env_num]
                            current_valIntra_obs = current_obs[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                            current_valInter_obs = current_obs[self.valIntra_env_num + self.train_env_num:]
                            current_train_states = current_states[:self.train_env_num]
                            current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                            current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                        else:
                            current_train_obs = {"state": current_obs["state"][:self.train_env_num], "pc": current_obs["pc"][:self.train_env_num]}
                            current_train_obs = {"state": current_obs["state"][self.train_env_num:self.valIntra_env_num + self.train_env_num], "pc": current_obs["pc"][self.train_env_num:self.valIntra_env_num + self.train_env_num]}
                            current_train_obs = {"state": current_obs["state"][self.valIntra_env_num + self.train_env_num:], "pc": current_obs["pc"][self.valIntra_env_num + self.train_env_num:]}
                            current_train_states = current_states[:self.train_env_num]
                            current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                            current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                            # Compute the action
                        train_actions, train_actions_log_prob, train_values, train_mu, train_sigma = self.actor_critic.act(current_train_obs, current_train_states)
                        valIntra_actions = self.actor_critic.act_inference(current_valIntra_obs)
                        valInter_actions = self.actor_critic.act_inference(current_valInter_obs)
                        actions = torch.cat((train_actions, valIntra_actions, valInter_actions))
                        # Step the vec_environment
                        next_obs, rews, dones, infos = self.vec_env.step(actions)

                        # #!
                        # rews*=0.01
                        # trick 3, 4
                        if self.use_reward_norm:    
                            rews = self.reward_norm(rews)
                        elif self.use_reward_scaling:
                            rews = self.reward_scaling(rews)


                        next_states = self.vec_env.get_state()
                        train_rews = rews[:self.train_env_num]
                        train_dones = dones[:self.train_env_num]
                        # Record the transition
                        if not self.use_pc:
                            self.storage.add_transitions(
                                current_train_obs,
                                current_train_states,
                                train_actions,
                                train_rews, 
                                train_dones,
                                train_values, 
                                train_actions_log_prob,
                                train_mu,
                                train_sigma
                            )
                        else:
                            self.storage.add_transitions(
                                current_train_obs["state"],
                                current_train_obs["pc"],
                                current_train_states,
                                train_actions,
                                train_rews, 
                                train_dones,
                                train_values, 
                                train_actions_log_prob,
                                train_mu,
                                train_sigma
                            )
                    else :
                        actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
                        next_obs, rews, dones, infos = self.vec_env.step(actions)
                        # while(1):
                        #     actions = torch.zeros((256,11)).float()
                        #     next_obs, rews, dones, infos = self.vec_env.step(actions)
                        #     pass
                        # debug_print("actions", actions, only_shape=True)
                        # debug_print("next_obs",next_obs, only_shape=True)
                        # debug_print("rews", rews, only_shape=True)
                        # debug_print("dones", dones, only_shape=True)
                        # debug_print("infos", infos)
                        # exit(123)

                        # trick 3, 4
                        if self.use_reward_norm:    
                            rews = self.reward_norm(rews)
                        elif self.use_reward_scaling:
                            rews = self.reward_scaling(rews)

                        #next_states = self.vec_env.get_state()
                        if not self.use_pc:
                            self.storage.add_transitions(
                                current_obs,
                                current_states,
                                actions,
                                rews, 
                                dones,
                                values, 
                                actions_log_prob,
                                mu,
                                sigma
                            )
                        else:
                            self.storage.add_transitions(
                                current_obs["state"],
                                current_obs["pc"],
                                current_states,
                                actions,
                                rews, 
                                dones,
                                values, 
                                actions_log_prob,
                                mu,
                                sigma
                            )

                    current_obs= next_obs.copy()
                    #current_states.copy_(next_states)
                    # Book keeping

                    ep_infos.append(infos)

                    if self.print_log:
                        
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:

                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)
                #(self.storage.observations[:,:,14])
                mean_value_loss, mean_surrogate_loss = self.update(it)
                if self.vec_env.task.task_meta["need_update"] :
                    task_info = self.vec_env.task.update(it)

                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if it % self.eval_freq == 0:
                    train_reward, valIntra_reward, valInter_reward,  train_success,  valIntra_success, valInter_success = self.eval(it)
                    if self.wandb_writer is not None:
                        self.wandb_writer.log({
                            "Val/train_reward": train_reward,
                            "Val/valIntra_reward": valIntra_reward,
                            "Val/valInter_reward": valInter_reward,
                            "Val/train_success": train_success,
                            "Val/valIntra_success": valIntra_success,
                            "Val/valInter_success": valInter_success,
                        })
                    

                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.tar'.format(it)), it)
                    if self.vec_env.task.task_meta["need_update"] :
                        self.vec_env.task.save(self.log_dir, it)
                ep_infos.clear()

            if self.vec_env.task.task_meta["need_update"] :
                self.vec_env.task.save(self.log_dir, num_learning_iterations)

    def log_test(self, locs, width=80, pad=35) :
        ##debug

        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        # if locs['task_info']:
        #     for key in locs['task_info']:
        #         value = locs['task_info'][key]
        #         self.writer.add_scalar('Episode/' + key, value)
        #         ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f} \n"""
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Exp Name':>{pad}} {self.exp_name} \n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Exp Name':>{pad}} {self.exp_name} \n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if_val = self.train_env_num<self.vec_env.num_envs
        if locs['ep_infos']:

            for key in locs['ep_infos'][0]:
                
                if key == "successes" or key == "success_entropy":
                    continue
                    infotensor_train = torch.zeros((self.train_env_num), device= self.device)
                    infotensor_val = torch.zeros((self.val_env_num), device=self.device)
                    for ep_info in locs['ep_infos']:
                        infotensor_train = torch.logical_or(infotensor_train, ep_info[key].to(self.device)[:self.train_env_num])
                        infotensor_val = torch.logical_or(infotensor_val, ep_info[key].to(self.device)[self.train_env_num:])
                elif key == "part_pos" or key == "grasp_success_rate" :
                    infotensor_train = torch.zeros((self.train_env_num), device= self.device)
                    infotensor_valIntra = torch.zeros((self.valIntra_env_num), device=self.device)
                    infotensor_valInter = torch.zeros((self.valInter_env_num), device=self.device)
                    for ep_info in locs['ep_infos']:
                        infotensor_train = torch.max(infotensor_train, ep_info[key].to(self.device)[:self.train_env_num])
                        infotensor_valIntra = torch.max(infotensor_valIntra, ep_info[key].to(self.device)[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                        infotensor_valInter = torch.max(infotensor_valInter, ep_info[key].to(self.device)[self.valIntra_env_num + self.train_env_num:])
                else: 
                    infotensor_train = torch.tensor([], device=self.device)
                    infotensor_valIntra = torch.tensor([], device=self.device)
                    infotensor_valInter = torch.tensor([], device=self.device)
                    for ep_info in locs['ep_infos']:

                        infotensor_train = torch.cat((infotensor_train, ep_info[key].to(self.device)[:self.train_env_num].float()))
                        infotensor_valIntra = torch.cat((infotensor_valIntra, ep_info[key].to(self.device)[self.train_env_num:self.valIntra_env_num + self.train_env_num].float()))
                        infotensor_valInter = torch.cat((infotensor_valInter, ep_info[key].to(self.device)[self.valIntra_env_num + self.train_env_num:].float()))
                if key=="success_rate":
                    sorted_success_rate_train, _ = torch.sort(infotensor_train)
                    sorted_success_rate_valIntra , _ = torch.sort(infotensor_valIntra)
                    sorted_success_rate_valInter , _ = torch.sort(infotensor_valInter)
                    worst_rate = 0.5
                    num_worst_train = int(infotensor_train.shape[0]*worst_rate)
                    num_worst_valIntra = int(infotensor_valIntra.shape[0]*worst_rate)
                    num_worst_valInter = int(infotensor_valInter.shape[0]*worst_rate)
                    worst_success_rate_train = sorted_success_rate_train[:num_worst_train]
                    worst_success_rate_valIntra = sorted_success_rate_valIntra[:num_worst_valIntra]
                    worst_success_rate_valInter = sorted_success_rate_valInter[:num_worst_valInter]
                    worst_mean_train = worst_success_rate_train.mean()
                    worst_mean_valIntra = worst_success_rate_valIntra.mean()
                    worst_mean_valInter = worst_success_rate_valInter.mean()

                    self.writer.add_scalar(f"""Episode/worst_{worst_rate*100}%_success_rate_train""", worst_mean_train, locs['it'])
                    if self.wandb_writer is not None : self.wandb_writer.log({f"""Episode/worst_{worst_rate*100}%_success_rate_train""": worst_mean_train})
                    ep_string += f"""{f'Mean episode worst {worst_rate*100}% success rate train:':>{pad}} {worst_mean_train:.4f} \n"""
                    if if_val:
                        self.writer.add_scalar(f"""Episode/worst_{worst_rate*100}%_success_rate_valIntra""", worst_mean_valIntra, locs['it'])
                        self.writer.add_scalar(f"""Episode/worst_{worst_rate*100}%_success_rate_valInter""", worst_mean_valInter, locs['it'])
                        if self.wandb_writer is not None :
                            self.wandb_writer.log({f"""Episode/worst_{worst_rate*100}%_success_rate_valIntra""": worst_mean_valIntra,
                                                    f"""Episode/worst_{worst_rate*100}%_success_rate_valInter""": worst_mean_valInter})
                        ep_string += f"""{f'Mean episode worst {worst_rate*100}% success rate valIntra:':>{pad}} {worst_mean_valIntra:.4f} \n"""  
                        ep_string += f"""{f'Mean episode worst {worst_rate*100}% success rate valInter:':>{pad}} {worst_mean_valInter:.4f} \n"""  

                value_train = torch.mean(infotensor_train.float())
                value_valIntra = torch.mean(infotensor_valIntra.float())
                value_valInter = torch.mean(infotensor_valInter.float())

                if key == "dist_tip" or key == "part_pos" :
                    self.writer.add_scalar('Train/' + key + '_train', value_train, locs['it'])
                    if self.wandb_writer is not None : self.wandb_writer.log({'Train/' + key + '_train': value_train})
                    ep_string += f"""{f'{key} train:':>{pad}} {value_train:.4f}\n"""
                    if if_val:
                        self.writer.add_scalar('Train/' + key + '_valIntra', value_valIntra, locs['it'])
                        self.writer.add_scalar('Train/' + key + '_valInter', value_valInter, locs['it'])
                        if self.wandb_writer is not None : 
                            self.wandb_writer.log({'Train/' + key + '_valIntra': value_valIntra, 
                                                    'Train/' + key + '_valInter': value_valInter})
                        ep_string += f"""{f'{key} valIntra:':>{pad}} {value_valIntra:.4f}\n"""
                        ep_string += f"""{f'{key} valInter:':>{pad}} {value_valInter:.4f}\n"""
                else :
                    self.writer.add_scalar('Episode/' + key + '_train', value_train, locs['it'])
                    if self.wandb_writer is not None : self.wandb_writer.log({'Episode/' + key + '_train': value_train})
                    ep_string += f"""{f'Mean episode {key} train:':>{pad}} {value_train:.4f}\n"""
                    if if_val:
                        ep_string += f"""{f'Mean episode {key} valIntra:':>{pad}} {value_valIntra:.4f}\n"""
                        self.writer.add_scalar('Episode/' + key + '_valIntra', value_valIntra, locs['it'])
                        ep_string += f"""{f'Mean episode {key} valInter:':>{pad}} {value_valInter:.4f}\n"""
                        self.writer.add_scalar('Episode/' + key + '_val', value_valInter, locs['it'])
                        if self.wandb_writer is not None :
                            self.wandb_writer.log({'Episode/' + key + '_valIntra': value_valIntra, 
                                                    'Episode/' + key + '_val': value_valInter})
                            

        if locs['task_info']:
            for key in locs['task_info']:
                value = locs['task_info'][key]
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f} \n"""

        mean_std = self.actor_critic.log_std.exp().mean()
        mean_part_position_train, _ = torch.max(self.storage.observations[:self.train_env_num, :, self.order_part_pos], dim=0)
        mean_part_position_valIntra, _ = torch.max(self.storage.observations[self.train_env_num:self.valIntra_env_num + self.train_env_num, :, self.order_part_pos], dim=0) if self.valIntra_env_num > 0 else (0, 0)
        mean_part_position_valInter, _ = torch.max(self.storage.observations[self.valIntra_env_num + self.train_env_num:, :, self.order_part_pos], dim=0) if self.valInter_env_num > 0 else (0, 0)
        mean_part_position_train = mean_part_position_train.mean()
        mean_part_position_valIntra = mean_part_position_valIntra.mean() if self.valIntra_env_num > 0 else 0
        mean_part_position_valInter = mean_part_position_valInter.mean() if self.valInter_env_num > 0 else 0
        
        ###wandb###
        if self.wandb_writer is not None :
            self.wandb_writer.log({
                    "total_steps": locs['it'], 
                    'Loss/value_function': locs['mean_value_loss'],
                    'Loss/surrogate': locs['mean_surrogate_loss'],
                    'Policy/mean_noise_std': mean_std.item(),
                    'Policy/lr': self.step_size,
                    'Train/mean_reward/step': locs['mean_reward'],
                    #'Train_/mean_episode_length/episode': locs['mean_trajectory_length'],
                    'Train/part_pos_train' : mean_part_position_train,
                    'Train/part_pos_valIntra' : mean_part_position_valIntra,
                    'Train/part_pos_valInter' : mean_part_position_valInter,
                     })
        
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Policy/lr', self.step_size, locs['it'])
        if len(locs['rewbuffer']) > 0:
            if self.wandb_writer is not None : self.wandb_writer.log({
                    'Train/reward': torch.tensor(list(collections.deque(locs['rewbuffer']))),
                    #'Train/episode_length/time': torch.tensor(list(collections.deque(locs['lenbuffer']))),
                    })
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            #self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            #self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train/mean_reward/step', locs['mean_reward'], locs['it'])
        #self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Exp Name':>{pad}} {self.exp_name} \n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          #f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          #f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                          f"""{'Learning Rate:':>{pad}} {self.step_size}\n"""
                          f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                          f"""{'Mean_part_position_valIntra:':>{pad}} {mean_part_position_valIntra:.5f}\n"""\
                          f"""{'Mean_part_position_valInter:':>{pad}} {mean_part_position_valInter:.5f}\n"""\
                          )
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Exp Name':>{pad}} {self.exp_name} \n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}s\n"""
                          #f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                          f"""{'Learning Rate:':>{pad}} {self.step_size}\n"""
                          f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                          f"""{'Mean_part_position_valIntra:':>{pad}} {mean_part_position_valIntra:.5f}\n"""\
                          f"""{'Mean_part_position_valInter:':>{pad}} {mean_part_position_valInter:.5f}\n"""\
                          )

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                
                if self.use_pc:
                    observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                    pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
                    obs_batch = {"pc": pc_batch, "state": observations_batch}
                else:
                    obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]
                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch,
                                                                                                                       states_batch,
                                                                                                                       actions_batch)
                ###Trick 1 advantage normalization
                if self.use_adv_norm:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + self.adv_norm_epsilon)

                # KL learning rate schedule
                if self.desired_kl > 0 and self.schedule == 'adaptive':

                    kl = torch.sum(
                        sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(self.lr_lower, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(self.lr_upper, self.step_size * 1.5)
                    
                    # if it > 2000 :
                    #     self.step_size = max(min(self.step_size, 3e-4 - (it-2000)/1000*3e-4), 0.0)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size
                
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                
                
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                # #!
                # surrogate_loss = surrogate.mean()


                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                # ipdb.set_trace()
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                if self.use_grad_clip:   ###trick 7
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                # self.network_lr_scheduler.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        
        if self.learning_rate_decay: 
            self.lr_decay(it)

        return mean_value_loss, mean_surrogate_loss
    
def space_add(a, b):

    if len(a.shape) != 1 or len(b.shape) != 1 :
        
        raise TypeError("Shape of two spaces need to be 1d")
    
    elif not isinstance(a, Box) or not isinstance(b, Box) :

        raise TypeError("Type of two spaces need to be Box")
    
    else :

        low = np.concatenate((a.low, b.low))
        high = np.concatenate((a.high, b.high))
        return Box(low=low, high=high)

