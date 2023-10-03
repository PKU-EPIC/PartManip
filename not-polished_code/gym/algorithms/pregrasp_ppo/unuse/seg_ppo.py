
import time
from webbrowser import get
import random
from gym.spaces import Space, Box
import math
import numpy as np
import statistics
from collections import deque
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data_structure.observation import Observations

from algorithms.pregrasp_ppo import RolloutStorage, RolloutStoragePC, RolloutStoragePC_feature, RolloutStorage_expert
from .module import Normalization, RewardScaling, Discriminator
from envs.utils.misc import _draw_line
# from scipy.spatial.transform import Rotation as R
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
     quat_mul, tensor_clamp
import torch    
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert, quaternion_multiply, quaternion_apply, euler_angles_to_matrix
from isaacgym.gymtorch import wrap_tensor
from envs.utils.get_running_bbox import get_bbox_isaac_tensor, _draw_bbox_tensor, get_bbox_pt, _draw_bbox_pt
from envs.utils.get_reward import quat_axis
from ..ppo_utils.io_util import load, save
from ..ppo_utils.misc_util import lr_decay
from ..ppo_utils.log_util import log, log_test
from data_structure.observation import Observations
from .ppo import PREGRASP_PPO
import os

class imitation_learning(PREGRASP_PPO):
    def __init__(self,
                 vec_env,
                 actor_critic_class,
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
                 discriminator_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 eval_round=1,
                 eval_freq = 50,
                 print_log=True,
                 apply_reset=False,

                 max_iterations = 50000,
                 wandb_writer = None,
                 checkpoint_path = ' ',

                 canonical = False,
                 canon_space = 'part',
                 use_residual_traj = False,


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


        super().__init__(
            vec_env = vec_env,
            actor_critic_class = actor_critic_class,
            num_transitions_per_env = num_transitions_per_env,
            num_learning_epochs = num_learning_epochs,
            num_mini_batches = num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            init_noise_std=init_noise_std,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            max_lr=max_lr,
            min_lr=min_lr,
            model_cfg=model_cfg,
            device=device,
            sampler=sampler,
            log_dir=log_dir,
            is_testing=is_testing,
            eval_round=eval_round,
            eval_freq = eval_freq,
            print_log=print_log,
            apply_reset=apply_reset,
            max_iterations = max_iterations,
            wandb_writer =wandb_writer,
            checkpoint_path = checkpoint_path,
            canonical = canonical,
            canon_space = canon_space,
            use_residual_traj = use_residual_traj,
            use_adv_norm = use_adv_norm,          ### trick 1
            adv_norm_epsilon= adv_norm_epsilon,
            use_state_norm= use_state_norm,         ### trick 2
            use_reward_norm= use_reward_norm,        ### trick 3
            use_reward_scaling= use_reward_scaling,     ### trick 4
            learning_rate_decay = learning_rate_decay,   ### trick 6
            use_grad_clip = use_grad_clip,          ###trick 7
            use_orthogonal_init= use_orthogonal_init,     ### trick 8
            adam_epsilon = adam_epsilon,           ### trick 9
        )
        print("finish ppo initialization~")

        self.debug = model_cfg["GAIL_debug"]
        self.backbone_freeze = model_cfg["Spconv"]["freeze"]
        
        # PPO components
        self.vec_env = vec_env
        self.use_qpose_state = self.vec_env.task.cfg["task"]["use_qpose_state"]
        self.pc_feature_shape = model_cfg["feature_dim"]
        self.model_cfg = model_cfg
        self.train_asset_num = self.train_env_num // self.vec_env.task.env_per_asset
        if model_cfg["use_discriminator"]:
            self.discriminator = Discriminator(discriminator_cfg, self.state_space.shape, self.action_space.shape, feature_dim=self.pc_feature_shape ,device = self.device)
            print(self.discriminator)
            self.discriminator_cfg = discriminator_cfg
            self.rew_clip_magnitude =  discriminator_cfg["rew_clip_magnitude"]
            self.mean_rew_policy = 0
            self.mean_rew_expert = 0
            self.use_self_imitation = model_cfg["use_self_imitation"]

        if not self.is_testing: 
            if not self.debug:
                if self.backbone_freeze:
                    self.storage = RolloutStoragePC_feature(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                                            self.state_space.shape, self.action_space.shape, self.pc_feature_shape, self.device, sampler)
                else:
                    # self.storage = RolloutStoragePC_feature(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                    #                         self.state_space.shape, self.action_space.shape, self.pc_feature_shape, self.device, sampler)
                    self.storage = RolloutStoragePC(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                                        self.state_space.shape, self.action_space.shape, self.pc_space.shape, self.device, sampler)
            else:
                self.storage = RolloutStorage(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                    self.state_space.shape, self.action_space.shape, self.device, sampler)

            if model_cfg["use_discriminator"]:
                self.demo_num = model_cfg["demo_num"]
                self.demo_states = torch.zeros((self.demo_num, self.train_asset_num, self.max_episode_length, *self.state_space.shape), device = self.device)
                
                if self.backbone_freeze:
                    self.demo_pcs = torch.zeros((self.demo_num, self.train_asset_num, self.max_episode_length, self.pc_feature_shape), device = self.device)
                else:
                    self.demo_pcs = torch.zeros((self.demo_num, self.train_asset_num, self.max_episode_length, *self.pc_space.shape), device = self.device)
                ############only for debug##########
                if self.debug:
                    self.demo_pcs =None
                self.demo_total_num = torch.zeros(self.train_asset_num, device = self.device)
                self.demo_self_num = torch.zeros(self.train_asset_num, device=self.device)

                self.demo_actions = torch.zeros((self.demo_num, self.train_asset_num, self.max_episode_length, *self.action_space.shape), device = self.device)
                self.demo_step =  torch.zeros((self.train_asset_num), device=self.device, dtype=torch.int)
               
                ############TODO############
                #setup deque for each object trajectory list()


                # self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps= adam_epsilon)   # , weight_decay=float(self.weight_decay), trick 9
                self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_cfg["lr"], weight_decay=discriminator_cfg["weight_reg"])
                self.bce = nn.BCEWithLogitsLoss().to(self.device)
                self.disc_gt = torch.cat(
                    [
                        torch.zeros((num_transitions_per_env * self.train_env_num, 1), device=self.device),
                        torch.ones((num_transitions_per_env * self.train_env_num, 1), device=self.device),
                    ],
                    dim=0,
                )


        # if self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed' or self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed_newtips' :
        #     self.order_part_pos = 10
        # elif self.vec_env.task.cfg["env"]["robotName"] == 'FrankaSliderXYZ' or self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed_SliderXYZ':
        #     self.order_part_pos = 16
        # else:
        #     self.order_part_pos = 14
        self.order_part_pos = 22
        

    def run(self, num_learning_iterations, log_interval=1):

        if self.model_cfg["use_discriminator"] and self.expert_cfg["canonicalize"]:
            self.canonicalize_init()
        if self.canonical :
            self.canonicalize_init()
            

        if self.vec_env.task.cfg["env"]["usePreGraspPose"]:
            self.pre_grasp()
        current_obs = self.vec_env.reset()
        self.episode_step_for_now = 0

        # if self.use_first_bbox is not None:
        #     if self.use_first_bbox == 'gt': 
        #         if not self.use_pc:
        #             self.init_obs = current_obs.clone()
        #         else:
        #             self.init_obs = current_obs.copy()
        #         # self.init_obs = current_obs.clone()
        #     elif self.use_first_bbox == 'pt':
        #         part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task) 
        #         #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
        #         #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
        #         #print(handle_bbox_tensor.shape)
        #         #print(part_bbox_tensor.shape)
        #         if not self.use_pc:
        #             self.init_obs = self.change_obs(part_bbox_tensor, handle_bbox_tensor, current_obs.clone())
        #         else:
        #             self.init_obs = self.change_obs(part_bbox_tensor, handle_bbox_tensor, current_obs.copy())
        

        # current_states = self.vec_env.get_state()   # size (16, 0) ?
        
        if self.is_testing:
            self.vec_env.task.eval()
            _ = self.eval(self.current_learning_iteration)
        else:
            self.vec_env.task.train()
            
            rewbuffer = deque(maxlen=200)
            if self.model_cfg["use_discriminator"]:
                gt_rewbuffer = deque(maxlen=200)
                disc_loss_all = deque(maxlen=10)
                lenbuffer = deque(maxlen=200)
                cur_gt_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
                cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            
            success_rate = []
            reward_sum = []
            gt_reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                task_info = {}  
               
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
                    
                total_success = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num), device=self.device)

                for i in range(self.max_episode_length): 
                    self.episode_step_for_now = i

                    if self.use_state_norm : ### trick 2
                        current_obs = self.state_norm(current_obs)
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        self.episode_step_for_now = 0
                        #current_states = self.vec_env.get_state()
                    #print(self.vec_env.task.progress_buf)
                    ####canonicalize
                    if self.use_residual_traj:
                        current_obs = self.residual_actions(current_obs, i)
                        
                    # if self.canonical: 
                    #     current_obs = self.canonicalize(current_obs)

                        
                    # if self.valIntra_env_num or self.valInter_env_num :
                    # if not self.use_pc:
                    #     current_train_obs = current_obs[:self.train_env_num]
                    #     current_valIntra_obs = current_obs[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                    #     current_valInter_obs = current_obs[self.valIntra_env_num + self.train_env_num:]
                    #     current_train_states = current_states[:self.train_env_num]
                    #     current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                    #     current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                    # else:
                    current_train_obs = Observations(state=current_obs.state[:self.train_env_num], points= current_obs.points[:self.train_env_num], obs=current_obs.obs[:self.train_env_num])
                    if self.valIntra_env_num:
                        current_valIntra_obs = Observations(state=current_obs.state[self.train_env_num:self.valIntra_env_num + self.train_env_num],obs=current_obs.obs[self.train_env_num:self.valIntra_env_num + self.train_env_num], points= current_obs.points[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                        # current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                    if self.valIntra_env_num:
                        current_valInter_obs = Observations(state=current_obs.state[self.valIntra_env_num + self.train_env_num:],obs=current_obs.obs[self.valIntra_env_num + self.train_env_num:], points=current_obs.points[self.valIntra_env_num + self.train_env_num:])
                        # current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                    
                    #####use noqpose and no boundingbox observations for actor

                    # Compute the action
                    train_actions, train_actions_log_prob, train_values, train_mu, train_sigma , train_features = self.actor_critic.act(current_train_obs)
                    actions = train_actions
                    features = train_features
                    if self.valIntra_env_num:
                        valIntra_actions, valIntra_features = self.actor_critic.act_inference(current_valIntra_obs)
                        actions = torch.cat((actions, valIntra_actions))
                        features = torch.cat((features, valIntra_features))
                    if self.valIntra_env_num:
                        valInter_actions, valInter_features = self.actor_critic.act_inference(current_valInter_obs)
                        actions = torch.cat((actions, valInter_actions))
                        features = torch.cat((features, valInter_features))
                    
                    ######actions un-canonlicalize#####
                    # if self.canonical:
                    #     actions = self.uncanonicalize(actions)



                    next_obs, gt_rews, dones, infos = self.vec_env.step(actions)
                    total_success = torch.logical_or(infos["successes"].to(self.device), total_success)
                    with torch.no_grad():
                        # disc_logits = self.discriminator(current_train_obs.state, train_actions, train_features).view(-1)
                        disc_logits = self.discriminator(current_obs.state, actions, features).view(-1)
                        rews = F.softplus(disc_logits, beta=1)
                        if self.rew_clip_magnitude is not None:
                            rews = torch.clamp(
                                rews, max=self.rew_clip_magnitude, min= -1 * self.rew_clip_magnitude
                            )


                    train_rews = rews[:self.train_env_num] * self.model_cfg["disc_reward_perportion"] + gt_rews[:self.train_env_num] 

                    #####ppo trick 3, 4#########
                    if self.use_reward_norm:    
                        rews = self.reward_norm(rews)
                    elif self.use_reward_scaling:
                        rews = self.reward_scaling(rews)

                    next_states = self.vec_env.get_state()
                    dones = dones[:self.train_env_num]

                    # Record the transition
                    if not self.debug:
                        if self.backbone_freeze:
                            self.storage.add_transitions(
                                current_train_obs.obs,
                                train_features,
                                current_train_obs.state,
                                train_actions,
                                train_rews.clone(), 
                                dones,
                                train_values, 
                                train_actions_log_prob,
                                train_mu,
                                train_sigma
                            )
                        else:
                            self.storage.add_transitions(
                                current_train_obs.obs,
                                current_train_obs.points[...,:6],
                                current_train_obs.state,
                                train_actions,
                                train_rews.clone(), 
                                dones,
                                train_values, 
                                train_actions_log_prob,
                                train_mu,
                                train_sigma
                            )
                    else:
                        self.storage.add_transitions(
                            current_train_obs.obs,
                            current_train_obs.state,
                            actions,
                            rews.clone(), 
                            dones,
                            train_values, 
                            train_actions_log_prob,
                            train_mu,
                            train_sigma
                        )
                    # else :

                    #     current_obs_new = current_obs
                    #     actions, actions_log_prob, values, mu, sigma, features = self.actor_critic.act(current_obs_new)
                    #     ######actions un-canonlicalize#####
                    #     if self.canonical:
                    #         actions = self.uncanonicalize(actions)
                    #     # if self.use_residual_traj:
                    #     #     actions = self.residual_actions(actions, i)
                    #     ######actions un-canonlicalize end##### 
                        
                    #     # print(actions.shape)
                    #     # print(self.demo_actions.shape)
                    #     next_obs, gt_rews, dones, infos = self.vec_env.step(actions)
                    #     # next_obs, gt_rews, dones, infos = self.vec_env.step(self.demo_actions[0, :, i])

                    #     total_success = torch.logical_or(infos["successes"].to(self.device), total_success)

                    #     with torch.no_grad():
                    #         disc_logits = self.discriminator(current_obs.state, actions, features).view(-1)
                    #         rews = F.softplus(disc_logits, beta=1)
                    #         if self.rew_clip_magnitude is not None:
                    #             rews = torch.clamp(
                    #                 rews, max=self.rew_clip_magnitude, min= -1 * self.rew_clip_magnitude
                    #             )

                        
                    #     rews = gt_rews + rews * self.model_cfg["disc_reward_perportion"]

                    #     # trick 3, 4
                    #     if self.use_reward_norm:    
                    #         rews = self.reward_norm(rews)
                    #     elif self.use_reward_scaling:
                    #         rews = self.reward_scaling(rews)
                            
                    #     #print(rews)
                    #     #next_states = self.vec_env.get_state()
                    #     if not self.debug:
                    #         if self.backbone_freeze: ####如果freeze backbone , storage里面直接存feature#######
                    #             self.storage.add_transitions(
                    #                 current_obs.state,
                    #                 features,
                    #                 current_states,
                    #                 actions,
                    #                 rews.clone(), 
                    #                 dones,
                    #                 values, 
                    #                 actions_log_prob,
                    #                 mu,
                    #                 sigma
                    #             )
                    #         else:
                    #             self.storage.add_transitions(
                    #                 current_obs.state,
                    #                 current_obs.points,
                    #                 current_states,
                    #                 actions,
                    #                 rews.clone(), 
                    #                 dones,
                    #                 values, 
                    #                 actions_log_prob,
                    #                 mu,
                    #                 sigma
                    #             )
                    #     else:
                    #         self.storage.add_transitions(
                    #             current_obs.state,
                    #             current_states,
                    #             actions,
                    #             rews.clone(), 
                    #             dones,
                    #             values, 
                    #             actions_log_prob,
                    #             mu,
                    #             sigma
                    #         )

                    current_obs= Observations(state=next_obs.state,obs=next_obs.obs, points=next_obs.points)
                    current_obs= Observations(state=next_obs.state,obs=next_obs.obs, points=next_obs.points)
                    #current_states.copy_(next_states)
                    # Book keeping

                    ep_infos.append(infos)

                    if self.print_log:
                        
                        cur_reward_sum[:] += rews
                        cur_gt_reward_sum[:] += gt_rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                        gt_reward_sum.extend(cur_gt_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].detach().cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_gt_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                     # Learning step
                    if (i + 1) % self.num_transitions_per_env == 0 and (i + 1) != self.max_episode_length:

                        current_obs_new = current_obs
                        
                        # import pdb
                        # pdb.set_trace()
                        _, _, last_values, _, _ , _= self.actor_critic.act(current_obs_new)
                        stop = time.time()

                        mean_trajectory_length, mean_reward = self.storage.get_statistics()

                        # Learning step
                                
                        disc_loss = self.discriminator_update(it)
                        disc_loss_all.extend([disc_loss.item()])

                        self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)

                        mean_value_loss, mean_surrogate_loss = self.update(it)

                        if self.vec_env.task.task_meta["need_update"] :
                            task_info = self.vec_env.task.update(it)


                        self.storage.clear()
                    
                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    gt_rewbuffer.extend(gt_reward_sum)
                    lenbuffer.extend(episode_length)
                current_obs_new = current_obs
                _, _, last_values, _, _ , _= self.actor_critic.act(current_obs_new)
                stop = time.time()
                collection_time = stop - start

                ##############test_demo_states############
                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                 ####update_discriminator
                
                disc_loss = self.discriminator_update(it)
                disc_loss_all.extend([disc_loss.item()])

                # Learning step
                start = stop

                self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)

                mean_value_loss, mean_surrogate_loss = self.update(it)

                if self.vec_env.task.task_meta["need_update"] :
                    task_info = self.vec_env.task.update(it)

                self.storage.clear()
                stop = time.time()

                learn_time = stop - start

                #################self imitate, update the demostration buffer##################
                ###############borrowed from Hao Shen, Weikang Wan###################
                if self.use_self_imitation:
                    for i in range(self.train_asset_num):
                        for j in range(self.vec_env.task.env_per_asset):
                            if total_success[i*self.vec_env.task.env_per_asset+j]:
                                # import pdb
                                # pdb.set_trace()
                                self.demo_states[self.demo_step[i], i] = self.storage.states[:, i]
                                self.demo_actions[self.demo_step[i], i] = self.storage.actions[:, i]
                                if not self.debug:
                                    if self.backbone_freeze:
                                        self.demo_pcs[self.demo_step[i], i] = self.storage.pointcloud_features[:, i]
                                    else:
                                        self.demo_pcs[self.demo_step[i], i] = self.storage.pointcloud[:, i]
                    
                            self.demo_step[i] += 1
                            self.demo_step = self.demo_step % self.demo_num

                #####################END################

            
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.tar'.format(it)), it)
                    if self.vec_env.task.task_meta["need_update"] :
                        self.vec_env.task.save(self.log_dir, it)
                ep_infos.clear()

            if self.vec_env.task.task_meta["need_update"] :
                self.vec_env.task.save(self.log_dir, num_learning_iterations)

    # def imit_log(self, locs, width=80, pad=35):
    #     self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
    #     self.tot_time += locs['collection_time'] + locs['learn_time']
    #     iteration_time = locs['collection_time'] + locs['learn_time']

    def load_expert(self, path):
        #root_path, model_name = os.path.split(path)
        #self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        #self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        #self.actor_critic.train()
        #if self.vec_env.task.task_meta["need_update"] :
        #    self.vec_env.task.load(root_path, self.current_learning_iteration)
        #    self.vec_env.task.train()
        assert os.path.exists(path)

        path = os.path.join(path, '{}'.format(self.vec_env.task.cfg["task"]["target_part_category"]))
        for i in range(self.train_env_num // self.vec_env.task.env_per_asset):
            path_i_new = path + '/{}.tar'.format(str(self.vec_env.task.selected_asset_path_list[i]).split("/")[-1])
            if os.path.exists(path_i_new):
                print("load demo_{} from ".format(i), path_i_new)
                
                dict = torch.load(path_i_new, map_location=self.device)
            
                assert(self.demo_num <= dict["traj_num"])
                self.demo_states[:, i] = dict["observations"][:self.demo_num]
                self.demo_actions[:, i] = dict["actions"][:self.demo_num]
                
                if self.debug:
                    continue

                if self.backbone_freeze:
                    self.demo_pcs[:, i] = self.actor_critic.get_features(dict["pcs"][:self.demo_num, :, :, :6].reshape(-1, dict["pcs"].shape[-2], dict["pcs"].shape[-1])).reshape(self.demo_num, self.num_transitions_per_env, -1)
                else:
                    self.demo_pcs[:, i] = dict["pcs"][:self.demo_num, :, :, :6]
            else:
                print(f"{path_i_new} is not found !")

    #load expert demostration and put it into deque

    def log(self, locs, width=80, pad=35):
        #mean_vel = torch.mean(torch.mean(self.storage.observations[:, :, 7:13], dim = 0), dim = 1)
        #print(mean_vel)
        #print((torch.abs(mean_vel) < 0.005).sum().item())
        self.tot_timesteps += self.max_episode_length * self.vec_env.num_envs
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
                        #print(ep_info[key])
                elif key == "part_pos" or key == "grasp_success_rate" :
                    infotensor_train = torch.zeros((self.train_env_num), device= self.device)
                    infotensor_valIntra = torch.zeros((self.valIntra_env_num), device=self.device)
                    infotensor_valInter = torch.zeros((self.valInter_env_num), device=self.device)
                    for ep_info in locs['ep_infos']:
                        infotensor_train = torch.max(infotensor_train, ep_info[key].to(self.device)[:self.train_env_num])
                        infotensor_valIntra = torch.max(infotensor_valIntra, ep_info[key].to(self.device)[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                        infotensor_valInter = torch.max(infotensor_valInter, ep_info[key].to(self.device)[self.valIntra_env_num + self.train_env_num:])
                    #print(infotensor_train , infotensor_val)
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

        mean_part_position_train, _ = torch.max(self.storage.observations[:, :self.train_env_num, self.order_part_pos], dim=0)
        #mean_part_position_valIntra, _ = torch.max(self.storage.observations[self.train_env_num:self.valIntra_env_num + self.train_env_num, :, self.order_part_pos], dim=0) if self.valIntra_env_num > 0 else (0, 0)
        #mean_part_position_valInter, _ = torch.max(self.storage.observations[self.valIntra_env_num + self.train_env_num:, :, self.order_part_pos], dim=0) if self.valInter_env_num > 0 else (0, 0)
        mean_part_position_train = mean_part_position_train.mean()
        #mean_part_position_valIntra = mean_part_position_valIntra.mean() if self.valIntra_env_num > 0 else 0
        #mean_part_position_valInter = mean_part_position_valInter.mean() if self.valInter_env_num > 0 else 0
        ###wandb###
        if self.wandb_writer is not None :
            self.wandb_writer.log({
                    "total_steps": locs['it'], 
                    'Loss/value_function': locs['mean_value_loss'],
                    'Loss/surrogate': locs['mean_surrogate_loss'],
                    'Policy/mean_noise_std': mean_std.item(),
                    'Policy/lr': self.step_size,
                    'Train/mean_reward/step': locs['mean_reward'],
                    'Train/Mean gt reward/step:': statistics.mean(locs['gt_rewbuffer']) / self.max_episode_length, 
                    #'Train_/mean_episode_length/episode': locs['mean_trajectory_length'],
                    'Train/part_pos_train' : mean_part_position_train,
                    #'Train/part_pos_valIntra' : mean_part_position_valIntra,
                    #'Train/part_pos_valInter' : mean_part_position_valInter,
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

        fps = int(self.max_episode_length * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

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
                        #   f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.3f}\n"""
                          f"""{'Mean gt reward/step:':>{pad}} {(statistics.mean(locs['gt_rewbuffer']) / self.max_episode_length):.2f}\n"""
                          #f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean expert reward/step:':>{pad}} {self.mean_rew_expert:.3f}\n"""
                          f"""{'Mean policy reward/step:':>{pad}} {self.mean_rew_policy:.3f}\n"""
                          #f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                          f"""{'Learning Rate:':>{pad}} {self.step_size}\n"""
                          f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                          #f"""{'Mean_part_position_valIntra:':>{pad}} {mean_part_position_valIntra:.5f}\n"""\
                          #f"""{'Mean_part_position_valInter:':>{pad}} {mean_part_position_valInter:.5f}\n"""\
                          )
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Exp Name':>{pad}} {self.exp_name} \n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'discriminator loss:':>{pad}} {statistics.mean(locs['disc_loss_all']):.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}s\n"""
                          f"""{'Mean expert reward/step:':>{pad}} {self.mean_rew_expert:.3f}\n"""
                          f"""{'Mean policy reward/step:':>{pad}} {self.mean_rew_policy:.3f}\n"""
                        #   f"""{'Mean gt reward:':>{pad}} {locs['gt_rewbuffer']:.2f}\n"""
                          #f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                          f"""{'Learning Rate:':>{pad}} {self.step_size}\n"""
                          f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                          #f"""{'Mean_part_position_valIntra:':>{pad}} {mean_part_position_valIntra:.5f}\n"""\
                          #f"""{'Mean_part_position_valInter:':>{pad}} {mean_part_position_valInter:.5f}\n"""\
                          )

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def discriminator_update(self, it):
        ####expert_batch####
        states_batch = self.storage.states.reshape(-1, self.storage.states.size(-1))
        actions_batch = self.storage.actions.reshape(-1, self.storage.actions.size(-1))
        pcs_batch = None
        expert_pcs_batch = None
        if not self.debug:
            if self.backbone_freeze:
                pcs_batch = self.storage.pointcloud_features.reshape(-1, self.storage.pointcloud_features.size(-1))
            else:
                #pcs_batch = self.storage.pointcloud_features.reshape(-1, self.storage.pointcloud.size(-1))
                pcs_batch = self.storage.pointcloud.reshape(-1, self.storage.pointcloud.size(-2), self.storage.pointcloud.size(-1))


        # import pdb
        # pdb.set_trace()
        rand_num = random.randint(0, self.demo_num - 1)
        if self.discriminator_cfg["use_random_transitions"]:
            rand_num_ = torch.randint(0, self.max_episode_length, (self.num_transitions_per_env,), device=self.device)
            expert_states_batch = self.demo_states[rand_num][:, rand_num_].reshape(-1, self.storage.states.size(-1))
            expert_actions_batch = self.demo_actions[rand_num][:, rand_num_].reshape(-1, self.storage.actions.size(-1))
            if not self.debug:
                if self.backbone_freeze:
                    expert_pcs_batch = self.demo_pcs[rand_num][:, rand_num_].reshape(-1, self.demo_pcs[rand_num].size(-1))
                else:
                    expert_pcs_batch = self.demo_pcs[rand_num][:, rand_num_].reshape(-1, self.demo_pcs[rand_num].size(-2), self.demo_pcs[rand_num].size(-1))
        else:
            expert_states_batch = self.demo_states[rand_num][:, self.episode_step_for_now+1-self.num_transitions_per_env:self.episode_step_for_now+1].reshape(-1, self.storage.states.size(-1))
            expert_actions_batch = self.demo_actions[rand_num][:, self.episode_step_for_now+1-self.num_transitions_per_env:self.episode_step_for_now+1].reshape(-1, self.storage.actions.size(-1))

            if not self.debug:
                if self.backbone_freeze:
                    expert_pcs_batch = self.demo_pcs[rand_num][:, self.episode_step_for_now+1-self.num_transitions_per_env:self.episode_step_for_now+1].reshape(-1, self.demo_pcs[rand_num].size(-1))
                else:
                    expert_pcs_batch = self.demo_pcs[rand_num][:, self.episode_step_for_now+1-self.num_transitions_per_env:self.episode_step_for_now+1].reshape(-1, self.demo_pcs[rand_num].size(-2), self.demo_pcs[rand_num].size(-1))
        
        # print(self.storage.pointcloud.size(-2), self.storage.pointcloud_features.size(-1))
        
        # print(expert_pcs_batch.shape)
        # print(pcs_batch.shape)

        ############not freeze##############
        if not self.backbone_freeze and not self.debug:
            expert_pcs_batch_new = torch.zeros((self.train_asset_num*self.num_transitions_per_env, self.pc_feature_shape), device = self.device)
            pcs_batch_new = torch.zeros((self.train_env_num*self.num_transitions_per_env, self.pc_feature_shape), device = self.device)
            with torch.no_grad():
                expert_pcs_batch_new = self.actor_critic.get_features(expert_pcs_batch)
                pcs_batch_new = self.actor_critic.get_features(pcs_batch)
            # for i in range(self.train_asset_num):
            #     with torch.no_grad():
            #         expert_pcs_batch_new[i*self.num_transitions_per_env:(i+1)*self.num_transitions_per_env] = self.actor_critic.get_features(expert_pcs_batch[i*self.num_transitions_per_env:(i+1)*self.num_transitions_per_env] )####变成feature
            
            # for i in range(self.train_env_num):
            #     with torch.no_grad():
            #         pcs_batch_new[i*self.num_transitions_per_env:(i+1)*self.num_transitions_per_env]  = self.actor_critic.get_features(pcs_batch[i*self.num_transitions_per_env:(i+1)*self.num_transitions_per_env] )
        else:
            expert_pcs_batch_new = expert_pcs_batch
            pcs_batch_new = pcs_batch
        # import pdb
        # pdb.set_trace()
        disc_logits_policy = self.discriminator(states_batch, actions_batch, pcs_batch_new)
        disc_logits_expert = self.discriminator(expert_states_batch, expert_actions_batch, expert_pcs_batch_new)
        disc_logits_expert = disc_logits_expert.repeat(self.vec_env.task.env_per_asset, 1)
        disc_logits = torch.cat((disc_logits_policy, disc_logits_expert), dim = 0)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())

        if self.discriminator_cfg["use_grad_pen"]:
            
            expert_input = expert_actions_batch
            policy_input = actions_batch

            if self.discriminator_cfg["use_feature"]:
                expert_input = torch.cat((expert_pcs_batch_new, expert_input), dim = 1)
                policy_input = torch.cat((pcs_batch_new, policy_input), dim = 1)
            if self.discriminator_cfg["use_gt_obs"]:
                expert_input = torch.cat((expert_states_batch, expert_input), dim = 1)
                policy_input = torch.cat((states_batch, policy_input), dim = 1)
            # import pdb
            # pdb.set_trace()

            expert_input = expert_input.repeat(self.vec_env.task.env_per_asset, 1)

            eps = torch.rand(expert_input.size(0), 1, device=self.device)
            # print(eps.shape)
            # print(expert_input.shape)
            interp_states = eps * expert_input + (1 - eps) * policy_input
            interp_states = interp_states.detach()
            interp_states.requires_grad_(True)

            gradients = torch.autograd.grad(
                outputs=self.discriminator.disc(interp_states).sum(),
                inputs=[interp_states],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.discriminator_cfg["grad_pen_weight"]
        else:
            disc_grad_pen_loss = 0

        disc_ce_loss = self.bce(disc_logits, self.disc_gt) + disc_grad_pen_loss

        accuracy = (disc_preds == self.disc_gt).type(torch.FloatTensor).mean()
       
        self.mean_rew_policy = F.softplus(disc_logits_policy, beta=1).detach().mean()
        self.mean_rew_expert = F.softplus(disc_logits_expert, beta=1).detach().mean()
        
        #print(length)
        if self.wandb_writer is not None:
            self.wandb_writer.log({
                "discriminator/loss": disc_ce_loss, 
                "discriminator/accuracy": accuracy, 
                "discriminator/policy_rews": self.mean_rew_policy, 
                "discriminator/expert_rews": self.mean_rew_expert, 
            })
        
        self.writer.add_scalar('discriminator/loss', disc_ce_loss, it)
        self.writer.add_scalar('discriminator/accuracy', accuracy, it)
        self.writer.add_scalar('discriminator/policy_rews', self.mean_rew_policy, it)
        self.writer.add_scalar('discriminator/expert_rews', self.mean_rew_expert, it)        


        #########update discriminator#############
        if it % self.discriminator_cfg["update_freq"] == 0:
            self.disc_optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            disc_ce_loss.backward() ####防止gradient graph中断
            self.disc_optimizer.step()

        return disc_ce_loss.detach()

    def update(self, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                pc_batch = None
                observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                if not self.debug: ############Only for debug##############
                    if not self.backbone_freeze:    
                        #pc_batch = self.storage.pointcloud_features.view(-1, *self.storage.pointcloud_features.size()[2:])[indices]
                        pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
                    else:
                        pc_batch = self.storage.pointcloud_features.view(-1, *self.storage.pointcloud_features.size()[2:])[indices]
                    
                    
                obs_batch = Observations(points=pc_batch, state=states_batch, obs=observations_batch)

                # if self.asymmetric:
                #     states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                # else:
                #     states_batch = None
                
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]
                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch, actions_batch)
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

