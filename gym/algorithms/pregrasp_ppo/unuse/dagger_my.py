import os
import time
from webbrowser import get
import random
from data_structure.observation import Observations
import ipdb
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

# from algorithms.imitation_learning import RolloutStorage, RolloutStoragePC, RolloutStoragePC_feature, RolloutStorage_expert
from .module import Normalization, RewardScaling, Discriminator
from envs.utils.misc import _draw_line
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
     quat_mul, tensor_clamp
import torch    
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert, quaternion_multiply, quaternion_apply, euler_angles_to_matrix
from isaacgym.gymtorch import wrap_tensor
from envs.utils.get_running_bbox import get_bbox_isaac_tensor, _draw_bbox_tensor, get_bbox_pt, _draw_bbox_pt
from envs.utils.get_reward import quat_axis
from .ppo import PREGRASP_PPO
from .storage import RolloutStorage, RolloutStorage_expert, RolloutStoragePC_feature, RolloutStoragePC

class DAGGER_PPO(PREGRASP_PPO):
    def __init__(self,
                 vec_env,
                 actor_critic_class,
                 actor_critic_expert_class,
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
                 expert_cfg=None,
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
        self.pc_feature_shape = model_cfg["feature_dim"]

        self.backbone_freeze = model_cfg["Spconv"]["freeze"]

        self.actor_critic_expert = actor_critic_expert_class(self.state_space.shape, self.action_space.shape,
                                               init_noise_std, expert_cfg, device = self.device)
       
        self.discriminator = Discriminator(discriminator_cfg, self.state_space.shape,  self.action_space.shape, feature_dim=self.pc_feature_shape ,device = self.device)
        self.actor_critic_expert.to(self.device)
        self.discriminator.to(self.device)
        self.use_expert = True # use expert policy network or use demostration
        self.model_cfg = model_cfg
        self.expert_cfg = expert_cfg
        self.discriminator_cfg = discriminator_cfg
        self.rew_clip_magnitude =  discriminator_cfg["rew_clip_magnitude"]
        self.mean_rew_policy = 0
        self.mean_rew_expert = 0
        self.use_self_imitation = model_cfg["use_self_imitation"]
        
        self.qpose_num = (self.vec_env.task.franka_num_dofs)*2 + 2 # robot qpose qvel + object qpos qvel
        
        if not self.is_testing:
            if self.use_expert:
                self.expert_storage = RolloutStorage_expert(self.train_env_num, num_transitions_per_env, self.state_space.shape,
                                        self.state_space.shape, self.action_space.shape, self.device, sampler)
            else:
                self.demo_num = model_cfg["demo_num"]
                self.demo_obs = torch.zeros((self.demo_num, self.train_env_num, self.num_transitions_per_env, *self.observation_space.shape), device = self.device)
                if self.backbone_freeze:
                    self.demo_pcs = torch.zeros((self.demo_num, self.train_env_num, self.num_transitions_per_env, self.pc_feature_shape), device = self.device)
                else:
                    self.demo_pcs = torch.zeros((self.demo_num, self.train_env_num, self.num_transitions_per_env, *self.pc_space.shape), device = self.device)
            
                self.demo_actions = torch.zeros((self.demo_num, self.train_env_num, self.num_transitions_per_env, *self.action_space.shape), device = self.device)
                self.demo_step =  torch.zeros((self.train_env_num), device=self.device, dtype=torch.float)
                ############TODO############
                #setup deque for each object trajectory list()


            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps= adam_epsilon)   # , weight_decay=float(self.weight_decay), trick 9
            self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_cfg["lr"])
            self.bce = nn.BCEWithLogitsLoss().to(self.device)
            self.l2 = nn.MSELoss().to(self.device)
            self.disc_gt = torch.cat(
            [
                torch.zeros((self.discriminator_cfg["update_freq"]  * self.train_env_num, 1), device=self.device),
                torch.ones((self.discriminator_cfg["update_freq"]  * self.train_env_num, 1), device=self.device),
            ],
            dim=0,
        )
            
            
    def load(self, path):
        assert os.path.exists(self.checkpoint_path)
        checkpoint_dict= torch.load(self.checkpoint_path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        self.discriminator.load_state_dict(checkpoint_dict["discriminator_dict"])
        self.disc_optimizer.load_state_dict(checkpoint_dict["discriminator_dict"])
        self.current_learning_iteration = checkpoint_dict["iteration"]
        self.tot_timesteps = checkpoint_dict["total_steps"]

    def load_expert(self, path):
        assert os.path.exists(path)
        if self.use_expert:
            checkpoint_dict= torch.load(path, map_location=self.device)
            self.actor_critic_expert.load_state_dict(checkpoint_dict["model_state_dict"])
        else:
            path = os.path.join(path, 'pc', '{}'.format(self.vec_env.task.cfg["task"]["target_part_category"]))
            for i in range(self.train_env_num):
                path_i = path + '/{}.tar'.format(i)
                print("load demo from ", path_i)
                assert os.path.exists(path_i)
                dict= torch.load(path_i, map_location=self.device)
                self.demo_obs[:, i] = dict["observations"][:self.demo_num]
                
                self.demo_actions[:, i] = dict["actions"][:self.demo_num]
                assert(self.demo_num <= dict["traj_num"])
                if self.backbone_freeze:
                    for j in range(self.demo_num):
                        self.demo_pcs[j] = self.actor_critic.get_features(dict["pcs"][j])
                else:
                    self.demo_pcs[j] = dict["pcs"]
    
    def save(self, path, it):    
        torch.save({
                    'iteration': it + 1,
                    'model_state_dict': self.actor_critic.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'total_steps' : self.tot_timesteps,
                    'discriminator_dict': self.discriminator.state_dict(),
                    "discriminator_optimizer": self.disc_optimizer.state_dict(),
                    }, path)

    def change_obs(self, part_bbox_tensor, handle_bbox_tensor, current_obs):
        # By default, the expert use edge bbox and need to change observations
        
        # if self.vec_env.task.cfg["obs"]["state"]["use_bbox_type"] != "None" and self.vec_env.task.cfg["obs"]["state"]["use_bbox_form"] == "vertex":
        #     # if not self.use_pc:
        #     current_obs.state[:, self.qpose_num+16:self.qpose_num+40]\
        #         .copy_(handle_bbox_tensor.view(-1, 24))
        #     current_obs.state[:, self.qpose_num+40:self.qpose_num+16+64]\
        #         .copy_(part_bbox_tensor.view(-1, 24))
            # else:
            #     current_obs["state"][:, self.qpose_num+16:self.qpose_num+40]\
            #         .copy_(handle_bbox_tensor.reshape(-1, 24))
            #     current_obs["state"][:, self.qpose_num+40:self.qpose_num+64]\
            #         .copy_(part_bbox_tensor.reshape(-1, 24))

        # elif self.vec_env.task.cfg["obs"]["state"]["use_bbox_type"] != "None" and self.vec_env.task.cfg["obs"]["state"]["use_bbox_form"] == "edge":

        handle_out_canon = handle_bbox_tensor[:, 0] -  handle_bbox_tensor[:, 4]
        handle_long_canon =  handle_bbox_tensor[:, 1] -  handle_bbox_tensor[:, 0]
        handle_short_canon =  handle_bbox_tensor[:, 3] -  handle_bbox_tensor[:, 0]
        handle_center_canon = (handle_bbox_tensor[:, 0] + handle_bbox_tensor[:, 6]) / 2
        part_out_canon = part_bbox_tensor[:, 0] - part_bbox_tensor[:, 4]
        part_long_canon =  part_bbox_tensor[:, 1] - part_bbox_tensor[:, 0]
        part_short_canon =  part_bbox_tensor[:, 3] - part_bbox_tensor[:, 0]
        part_center_canon = (part_bbox_tensor[:, 0] + part_bbox_tensor[:, 6]) / 2
        #34-46: part bounding box
        # if not self.use_pc:
        
        current_obs.state[:, self.qpose_num+16:self.qpose_num+19]\
            .copy_(handle_out_canon)
        current_obs.state[:, self.qpose_num+19:self.qpose_num+22]\
            .copy_(handle_long_canon)
        current_obs.state[:, self.qpose_num+22:self.qpose_num+25]\
            .copy_(handle_short_canon)
        current_obs.state[:, self.qpose_num+25:self.qpose_num+28]\
            .copy_(handle_center_canon)
        #46-57: handle bounding box
        current_obs.state[:, self.qpose_num+28:self.qpose_num+31]\
            .copy_(part_out_canon)
        current_obs.state[:, self.qpose_num+31:self.qpose_num+34]\
            .copy_(part_long_canon)
        current_obs.state[:, self.qpose_num+34:self.qpose_num+37]\
            .copy_(part_short_canon)
        current_obs.state[:, self.qpose_num+37:self.qpose_num+40]\
            .copy_(part_center_canon)

            # else:
            #     current_obs["state"][:, self.qpose_num+16:self.qpose_num+19]\
            #         .copy_(handle_out_canon)
            #     current_obs["state"][:, self.qpose_num+19:self.qpose_num+22]\
            #         .copy_(handle_long_canon)
            #     current_obs["state"][:, self.qpose_num+22:self.qpose_num+25]\
            #         .copy_(handle_short_canon)
            #     current_obs["state"][:, self.qpose_num+25:self.qpose_num+28]\
            #         .copy_(handle_center_canon)
            #     #46-57: handle bounding box
            #     current_obs["state"][:, self.qpose_num+28:self.qpose_num+31]\
            #         .copy_(part_out_canon)
            #     current_obs["state"][:, self.qpose_num+31:self.qpose_num+34]\
            #         .copy_(part_long_canon)
            #     current_obs["state"][:, self.qpose_num+34:self.qpose_num+37]\
            #         .copy_(part_short_canon)
            #     current_obs["state"][:, self.qpose_num+37:self.qpose_num+40]\
            #         .copy_(part_center_canon)
        # else:
        #     raise NotImplementedError

        return current_obs

    def canonicalize(self, current_obs_old):
        # if not self.use_pc:
        #     current_obs = current_obs_old.clone()
        # else:
        current_obs = Observations(state = current_obs_old.state.clone(), obs=current_obs_old.obs.clone(), points = current_obs_old.points.clone())

        if self.use_first_bbox is not None:
            part_bbox_tensor = self.part_bbox_tensor
            handle_bbox_tensor = self.handle_bbox_tensor
        else:
            if self.vec_env.task.target_part == "door":
                part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
            elif self.vec_env.task.target_part == "drawer":
                part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 1)
        
        
        part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
        # _draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
        # _draw_line(self.vec_env.task, handle_bbox_tensor[0, 0].cpu(), (handle_bbox_tensor[0, 0]).cpu() + torch.tensor([1, 0, 0]), clear=False)
        # _draw_line(self.vec_env.task, handle_bbox_tensor[0, 0].cpu(), (handle_bbox_tensor[0, 0]).cpu() + torch.tensor([0, 1, 0]), clear=False)
        # _draw_line(self.vec_env.task, handle_bbox_tensor[0, 0].cpu(), (handle_bbox_tensor[0, 0]).cpu() + torch.tensor([0, 0, 1]), clear=False)
    
        ########hand_pose
        # if not self.use_pc:
        #     hand_pose = current_obs[:,self.qpose_num:self.qpose_num+7]   #N * 7
        # else:
        # import pdb
        # pdb.set_trace()
        hand_pose = current_obs.state[:,self.qpose_num:self.qpose_num+7]   #N * 7

        hand_pose[:, :3] += self.vec_env.task.franka_root_tensor[:, :3]  
        hand_pose[:, :3] -= self.center
        #print(hand_pose.dtype, rot_matrix.float().dtype)
        #print(hand_pose[:, :3].shape, rot_matrix.shape)
        #hand_pose[:, :3] = (hand_pose[:, :3].view(-1, 1, 3) @ self.rot_matrix.float()).view(-1, 3)
        hand_pose[:, :3] = quaternion_apply(self.quaternion_rot, hand_pose[:, :3])
        #root_new = (self.vec_env.task.franka_root_tensor[:, :3] .view(-1, 1, 3) @ rot_matrix.float()).view(-1, 3)
        #hand_pose[:, :3] -= root_new
        #_draw_line(self.vec_env.task, hand_pose[0, :3].cpu(), (hand_grip_dir[0]@rot_matrix[0].float() + hand_pose[0, :3]).cpu(), clear=False)               
        
        hand_pose[:, 3:7] = quaternion_multiply(self.quaternion_rot, hand_pose[:, 3:7])
        #######bbox
        handle_bbox_tensor_canon = handle_bbox_tensor - self.center.view(-1, 1, 3)
        part_bbox_tensor_canon = part_bbox_tensor - self.center.view(-1, 1, 3)
        handle_bbox_tensor_canon = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), handle_bbox_tensor_canon) #@self.rot_matrix
        part_bbox_tensor_canon = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), part_bbox_tensor_canon) #@self.rot_matrix
        # hand_grip_dir = quat_axis(hand_pose[:, 3:7], 2)
        # hand_sep_dir = quat_axis(hand_pose[:, 3:7], 1)
        # hand_down_dir = quat_axis(hand_pose[:, 3:7], 0)
        #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
        #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
        #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor_canon)
        #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor_canon)
        #_draw_line(self.vec_env.task, hand_pose[0, :3].cpu(), torch.tensor([0, 0, 0]), clear=False)
        #_draw_line(self.vec_env.task, hand_pose[0, :3].cpu(), (hand_grip_dir[0] + hand_pose[0, :3]).cpu(), clear=False)
        #time.sleep(1)
        if self.use_pc:
            # current_obs["state"][:,self.qpose_num:self.qpose_num+7] = hand_pose
            # current_obs["state"][:,self.qpose_num+13:self.qpose_num+16] =quaternion_apply(self.quaternion_rot.view(-1, 1, 4), (self.vec_env.task.franka_root_tensor[:, :3] - self.center).reshape(-1,1,3)).view(-1, 3)
            current_obs.points[:,:, :3] = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), current_obs.points[:,:, :3]-self.center.view(-1,1,3))
        
        else:
            current_obs.state[:,self.qpose_num:self.qpose_num+7] = hand_pose
            #print(current_obs.shape)
            #print(self.vec_env.task.franka_root_tensor.shape)
            #print(self.quaternion_rot.shape)
            #print(self.center.shape)
            #print(quaternion_apply(self.quaternion_rot.view(-1, 4), self.vec_env.task.franka_root_tensor[:, :3] - self.center).shape)
            current_obs.state[:,self.qpose_num+13:self.qpose_num+16] = quaternion_apply(self.quaternion_rot.view(-1, 4), self.vec_env.task.franka_root_tensor[:, :3] - self.center)
            #exit()

        return self.change_obs(part_bbox_tensor_canon, handle_bbox_tensor_canon, current_obs)
    
    

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

        mean_part_position_train, _ = torch.max(self.storage.observations[:self.train_env_num, :, self.order_part_pos], dim=0)
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
                    'Mean gt reward/step:': statistics.mean(locs['gt_rewbuffer']) / self.max_episode_length, 
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
                        #   f"""{'discriminator loss:':>{pad}} {statistics.mean(locs['disc_loss_all']):.4f}\n"""
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

    def run(self, num_learning_iterations, log_interval = 1):
        if self.canonical or self.expert_cfg["canonicalize"]:
            self.canonicalize_init()
            
        if self.is_testing:
            self.vec_env.task.eval()
            _ = self.eval(self.current_learning_iteration)
        else:
            self.vec_env.task.train()
            
            # buffers
            
            rewbuffer = deque(maxlen=200)
            gt_rewbuffer = deque(maxlen=200)
            # disc_loss_all = deque(maxlen=10)
            lenbuffer = deque(maxlen=200)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_gt_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            success_rate = []
            reward_sum = []
            gt_reward_sum = []
            episode_length = []
            
            for it in range(self.current_learning_iteration, num_learning_iterations):
                current_obs = self.vec_env.reset()
                start = time.time()
                ep_infos = []
                task_info = {}
                
                # evaluation
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
                    
                    # reset !!!
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        
                    # tricks
                    if self.use_state_norm : ### trick 2
                        current_obs = self.state_norm(current_obs)
                    if self.use_residual_traj:
                        current_obs = self.residual_actions(current_obs, i)
                        
                    if not self.use_pc:
                        current_train_obs = Observations(state=current_obs.state[:self.train_env_num], obs=current_obs.obs[:self.train_env_num])
                        if self.valIntra_env_num:
                            current_valIntra_obs = Observations(state=current_obs.state[self.train_env_num:self.valIntra_env_num + self.train_env_num], obs=current_obs.obs[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                        if self.valIntra_env_num:
                            current_valInter_obs = Observations(state=current_obs.state[self.valIntra_env_num + self.train_env_num:], obs=current_obs.obs[self.valIntra_env_num + self.train_env_num:])

                    else:
                        current_train_obs = Observations(state=current_obs.state[:self.train_env_num], points= current_obs.points[:self.train_env_num], obs=current_obs.obs[:self.train_env_num])
                        if self.valIntra_env_num:
                            current_valIntra_obs = Observations(state=current_obs.state[self.train_env_num:self.valIntra_env_num + self.train_env_num],obs=current_obs.obs[self.train_env_num:self.valIntra_env_num + self.train_env_num], points= current_obs.points[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                        if self.valIntra_env_num:
                            current_valInter_obs = Observations(state=current_obs.state[self.valIntra_env_num + self.train_env_num:],obs=current_obs.obs[self.valIntra_env_num + self.train_env_num:], points=current_obs.points[self.valIntra_env_num + self.train_env_num:])

                    
                    train_actions, train_actions_log_prob, train_values, train_mu, train_sigma, others = self.actor_critic.act(current_train_obs, require_grad = False)
                    actions = train_actions
                    features = others["feature"]
                    if self.valIntra_env_num:
                        valIntra_actions, others = self.actor_critic.act_inference(current_valIntra_obs)
                        actions = torch.cat((actions, valIntra_actions))
                        valIntra_features = others["feature"]
                        if valIntra_features is not None:
                            features = torch.cat((features, valIntra_features)) 
                    if self.valIntra_env_num:
                        valInter_actions, others = self.actor_critic.act_inference(current_valInter_obs)
                        actions = torch.cat((actions, valInter_actions))
                        valInter_features = others["feature"]
                        if valInter_features is not None:
                            features = torch.cat((features, valInter_features)) 
                    if self.use_expert:
                        if self.expert_cfg["canonicalize"]:
                            current_obs_expert = self.canonicalize(current_obs)
                        else:
                            current_obs_expert = current_obs

                        expert_actions, _ = self.actor_critic_expert.act_inference(current_obs_expert)
                        
                        if self.expert_cfg["canonicalize"]:
                            expert_actions = self.uncanonicalize(expert_actions)
                    next_obs, gt_rews, dones, infos = self.vec_env.step(actions)
                    total_success = torch.logical_or(infos["successes"].to(self.device), total_success)

                    train_rews = gt_rews[:self.train_env_num]

                    if self.use_expert:
                        self.expert_storage.add_transitions(
                            current_train_obs.state,
                            train_actions,
                            expert_actions[:self.train_env_num],
                        )
                    
                    # tricks
                    if self.use_reward_norm:    
                        rews = self.reward_norm(rews)
                    elif self.use_reward_scaling:
                        rews = self.reward_scaling(rews)
                        
                        
                    train_dones = dones[:self.train_env_num]
                    # Record the transition
                    if not self.use_pc:
                        self.storage.add_transitions(
                            observations = current_train_obs.obs,
                            states = current_train_obs.state,
                            actions = train_actions,
                            rewards = train_rews, 
                            dones = train_dones,
                            values = train_values, 
                            actions_log_prob = train_actions_log_prob,
                            mu = train_mu,
                            sigma = train_sigma,
                        )
                    else:
                        self.storage.add_transitions(
                            observations = current_train_obs.obs, 
                            points = current_train_obs.points[...,:self.numPCObservations],
                            states = current_train_obs.state,
                            actions = train_actions,
                            rewards = train_rews, 
                            dones = train_dones,
                            values = train_values, 
                            actions_log_prob = train_actions_log_prob,
                            mu = train_mu,
                            sigma = train_sigma
                        )

                    current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)

                    ep_infos.append(infos)
                    torch.cuda.synchronize()

                    if self.print_log:
                        
                        cur_reward_sum[:] += gt_rews
                        cur_gt_reward_sum[:] += gt_rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                        gt_reward_sum.extend(cur_gt_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].detach().cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_gt_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        
                    # update    
                    self.actor_critic.train() 
                    if (i + 1) % self.num_transitions_per_env == 0 and (i + 1) != self.max_episode_length:

                        _, _, last_values, _, _, _ = self.actor_critic.act(current_obs)
                        
                        stop = time.time()
                        collection_time = stop - start

                        # mean_trajectory_length, mean_reward = self.storage.get_statistics()
                        mean_reward = self.storage.rewards.mean()
                        # Learning step
                        start = stop
                        self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)
                        #(self.storage.observations[:,:,14])
                        mean_value_loss, mean_surrogate_loss = self.update(it)
                        self.storage.clear()
                        self.expert_storage.clear()
                        torch.cuda.synchronize()
                        stop = time.time()
                        learn_time = stop - start
                        start = stop
                        if self.print_log:
                            self.log(locals())
                    
                    if self.print_log:
                        rewbuffer.extend(reward_sum)
                        gt_rewbuffer.extend(gt_reward_sum)
                        lenbuffer.extend(episode_length)

                # current_obs_new = {"state": current_obs["state"][:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
                _, _, last_values, _, _ , _= self.actor_critic.act(current_obs)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop

                self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)

                mean_value_loss, mean_surrogate_loss = self.update(it)

                if self.vec_env.task.task_meta["need_update"] :
                    task_info = self.vec_env.task.update(it)

                self.storage.clear()
                self.expert_storage.clear()
                stop = time.time()

                learn_time = stop - start
                
                if self.print_log:
                    self.log(locals())
                    
                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.tar'.format(it)), it)
                    if self.vec_env.task.task_meta["need_update"] :
                        self.vec_env.task.save(self.log_dir, it)
                ep_infos.clear()

            if self.vec_env.task.task_meta["need_update"] :
                self.vec_env.task.save(self.log_dir, num_learning_iterations)


    def update(self,it):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for indices in batch:
            if not self.backbone_freeze:
                observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
                # observations_batch = observations_batch[:, self.qpose_num:self.qpose_num+16]
                # obs_batch = {"pc": pc_batch, "state": observations_batch}
            else:
                observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                pc_batch = self.storage.pointcloud_features.view(-1, *self.storage.pointcloud_features.size()[2:])[indices]
                # observations_batch = observations_batch[:, self.qpose_num:self.qpose_num+16]
                # obs_batch = {"pc": pc_batch, "state": observations_batch}
            
            # if self.asymmetric:
            states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
            # else:
            #     states_batch = None

            expert_actions_batch = self.expert_storage.expert_actions.reshape(-1, self.expert_storage.actions.size(-1))[indices]
            actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
            # import pdb
            # pdb.set_trace()
            surrogate_loss = self.l2(actions_batch, expert_actions_batch)



            target_values_batch = self.storage.values.view(-1, 1)[indices]
            returns_batch = self.storage.returns.view(-1, 1)[indices]
            old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
            advantages_batch = self.storage.advantages.view(-1, 1)[indices]
            old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
            old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]
            observations = Observations(state = states_batch, obs= observations_batch, points= pc_batch)
            actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(observations, actions_batch)

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = value_loss + surrogate_loss
            # self.network_lr_scheduler.step()
            self.optimizer.zero_grad()
            loss.backward()
            if self.use_grad_clip:   ###trick 7
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_value_loss /= self.num_mini_batches
            mean_surrogate_loss /= self.num_mini_batches
        return mean_value_loss, mean_surrogate_loss

        