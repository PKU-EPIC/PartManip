import os
from pickletools import long4
import time
from webbrowser import get
import ipdb
from gym.spaces import Space, Box
import math
import numpy as np
import statistics
from collections import deque
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithms.imitation_learning import RolloutStorage, RolloutStoragePC, RolloutStoragePC_feature
from .module import Normalization, RewardScaling

from envs.utils.misc import _draw_line
from envs.utils.compute import *

# from scipy.spatial.transform import Rotation as R
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
     quat_mul, tensor_clamp
import torch    
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert, quaternion_multiply, quaternion_apply, euler_angles_to_matrix
from isaacgym.gymtorch import wrap_tensor
from envs.utils.get_running_bbox import get_bbox_isaac_tensor, _draw_bbox_tensor, get_bbox_pt, _draw_bbox_pt
from envs.utils.get_reward import quat_axis
from data_structure.observation import Observations
from ..ppo_utils.io_util import load, save
from ..ppo_utils.misc_util import lr_decay
from ..ppo_utils.log_util import log, log_test

class collect_data:
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
        self.pregrasp_steps = vec_env.task.cfg["task"]["pregrasp_steps"]
        self.use_part_pregrasp= vec_env.task.cfg["task"]["use_part_pregrasp"]
        self.init_obs = None
        self.use_first_bbox = vec_env.task.cfg["task"]["use_first_bbox"]

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
        self.use_qpose_state = self.vec_env.task.cfg["task"]["use_qpose_state"]
        # self.qpose_num = (self.vec_env.task.franka_num_dofs - 4)*2+2 
        
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, device= self.device)
        self.pc_feature_shape = model_cfg["feature_dim"]
        print(self.actor_critic)
        self.actor_critic.to(self.device)
        self.is_testing = is_testing
        self.save_path = model_cfg["save_path"]
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
        if not self.is_testing: 
            if not self.use_pc:
                self.storage = RolloutStoragePC_feature(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                                        self.state_space.shape, self.action_space.shape, self.pc_feature_shape, self.device, sampler)
            else:
                self.storage = RolloutStoragePC(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                                        self.state_space.shape, self.action_space.shape, self.pc_space.shape, self.device, sampler)

            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps= adam_epsilon)   # , weight_decay=float(self.weight_decay), trick 9

        self.demo_num = model_cfg["demo_num"]
       
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
        if self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed' or self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed_newtips' :
            self.order_part_pos = 10
        elif self.vec_env.task.cfg["env"]["robotName"] == 'FrankaSliderXYZ' or self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed_SliderXYZ':
            self.order_part_pos = 16
        else:
            self.order_part_pos = 14
        
        # Trick
        self.use_residual_traj = use_residual_traj
        self.use_adv_norm = use_adv_norm                    ###trick 1
        self.adv_norm_epsilon = adv_norm_epsilon
        self.use_state_norm = use_state_norm                ###trick 2 
        #print((self.train_env_num,)+self.observation_space.shape)
        self.state_norm = Normalization(shape = (self.train_env_num,)+self.observation_space.shape)
        self.use_reward_norm = use_reward_norm              ###trick 3
        self.reward_norm = Normalization(shape=self.train_env_num)
        self.use_reward_scaling = use_reward_scaling        ###trick 4
        self.reward_scaling = RewardScaling(shape =self.train_env_num, gamma= self.gamma)
        self.learning_rate_decay = learning_rate_decay      ###trick 6
        self.use_grad_clip = use_grad_clip                  ###trick 7
        self.use_orthogonal_init = use_orthogonal_init      ###trick 8 
        
        self.canonical = canonical
        self.canon_space = canon_space
        #print(self.use_residual_traj)
        #exit()
        #print(self.use_first_bbox)
        if self.use_first_bbox == "gt" or self.use_first_bbox is None: 
            self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
        elif self.use_first_bbox == "pt":
            self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_pt(self.vec_env.task)

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

    def load_expert(self, expert_path):
        # import pdb
        # pdb.set_trace()
        assert os.path.exists(expert_path)
        checkpoint_dict= torch.load(expert_path, map_location=self.device)
        actor_dict = {}
        critic_dict = {}
        log_std = None
        for key in checkpoint_dict["model_state_dict"].keys():
            key_split = key.split('.')
            # print(key_split)
            if key_split[0] == 'actor_mlp':
                actor_dict['.'.join(key_split[1:])] = checkpoint_dict["model_state_dict"][key]
            elif key_split[0] == 'critic_mlp':
                critic_dict['.'.join(key_split[1:])] = checkpoint_dict["model_state_dict"][key]
            else:
                log_std= checkpoint_dict["model_state_dict"][key]

        self.actor_critic.actor_mlp.load_state_dict(actor_dict)
        self.actor_critic.critic_mlp.load_state_dict(critic_dict)
        self.actor_critic.log_std = torch.nn.Parameter(log_std)
        #self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        # self.current_learning_iteration = checkpoint_dict["iteration"]
        # self.tot_timesteps = checkpoint_dict["total_steps"]
        
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

    def pre_grasp(self):
        print("start pregrasp")
        actions = torch.zeros((self.train_env_num+self.valIntra_env_num+self.valInter_env_num, self.action_space.shape[0]), device = self.device)
        #print(actions.shape)
        #先向后移动, 防止碰撞

        if self.use_part_pregrasp:
            print("use_part_pregrasp")
            tar0 = torch.tensor([0.8, 0, 0], device=self.device)+ self.vec_env.task.hand_rigid_body_tensor[:, :3]
            for i in range(200):
                actions[:, :3] = tar0 - self.vec_env.task.hand_rigid_body_tensor[:, :3]
                self.vec_env.control_step(actions)
            if self.vec_env.task.target_part == "door":
                pos_pre = (self.part_bbox_tensor[:,1,:] + self.part_bbox_tensor[:,3,:])/2 + torch.tensor([0.9, 0, 0], device = self.vec_env.task.device)
                pos_tar = (self.part_bbox_tensor[:,1,:] + self.part_bbox_tensor[:,3,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
            elif self.vec_env.task.target_part =="drawer":
                pos_pre = (self.part_bbox_tensor[:,2,:] + self.part_bbox_tensor[:,5,:])/2 + torch.tensor([0.9, 0, 0], device = self.vec_env.task.device)
                pos_tar = (self.part_bbox_tensor[:,2,:] + self.part_bbox_tensor[:,5,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
            else:
                raise NotImplementedError
            actions[:, -2:] = torch.tensor([1, 1], device=self.device)
            handle_out = torch.tensor([1.0, 0, 0], device=self.device)
            handle_out = handle_out.repeat(self.train_env_num + self.val_env_num, 1)
            handle_long = torch.tensor([0, 0, 1.0], device=self.device)
            handle_long = handle_long.repeat(self.train_env_num + self.val_env_num, 1)
            handle_short = torch.tensor([0, 1.0, 0], device=self.device)
            handle_short = handle_short.repeat(self.train_env_num + self.val_env_num, 1)
        else:
            pos_pre = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
            pos_tar = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.35, 0, 0], device = self.vec_env.task.device)
            actions[:, -2:] = (torch.norm(self.handle_bbox_tensor[:,0,:] - self.handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.04)/2
            handle_out = self.handle_bbox_tensor[:, 0] - self.handle_bbox_tensor[:, 4]
            handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
            handle_long = self.handle_bbox_tensor[:, 1] - self.handle_bbox_tensor[:, 0]
            handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
            handle_short = self.handle_bbox_tensor[:, 3] - self.handle_bbox_tensor[:, 0]
            handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
        
        rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
        actions[:,3:7] = rotations

        total_step = 3
        for step in range(total_step):
            print("pregrasp_step{}".format(step))
            for i in range(self.pregrasp_steps):
                #print(i)    
                # if i == 1:
                #     print(wrap_tensor(self.vec_env.task.gym.acquire_dof_state_tensor(self.vec_env.task.sim)))
                
                #part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
                #part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
                #hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]

                actions[:,:3] = (step/(total_step-1))*(pos_tar - pos_pre)+pos_pre -  self.vec_env.task.hand_rigid_body_tensor[:, :3]
                #actions[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
                actions[:, :3] *= math.exp(-0.002 * i)
            
                actions[:, :3] = torch.clamp(actions[:, :3], max=1, min=-1)
                self.vec_env.control_step(actions)
        print(self.vec_env.task.cabinet_dof_tensor[:,0])
        print(torch.norm(pos_tar - self.vec_env.task.hand_rigid_body_tensor[:, :3], dim = 1))
        print("end pregrasp")
        
        self.vec_env.task.initial_dof_states = wrap_tensor(self.vec_env.task.gym.acquire_dof_state_tensor(self.vec_env.task.sim)).clone()

        
    def canonicalize_init(self):

        if self.canon_space == 'part': ######???????
            self.center = (self.part_bbox_tensor[:, 0, :] + self.part_bbox_tensor[:, 6, :]) / 2
            out = self.part_bbox_tensor[:, 0] - self.part_bbox_tensor[:, 4]
            long = self.part_bbox_tensor[:, 1] - self.part_bbox_tensor[:, 0]
            short = self.part_bbox_tensor[:, 3] - self.part_bbox_tensor[:, 0]
        elif self.canon_space == 'handle':
            self.center = (self.handle_bbox_tensor[:, 0, :] + self.handle_bbox_tensor[:, 6, :]) / 2
            out = self.handle_bbox_tensor[:, 0] - self.handle_bbox_tensor[:, 4]
            long = self.handle_bbox_tensor[:, 1] - self.handle_bbox_tensor[:, 0]
            short = self.handle_bbox_tensor[:, 3] - self.handle_bbox_tensor[:, 0]
        else:
            raise NotImplementedError

        out_dir = out / torch.norm(out, dim = 1, keepdim=True)
        long_dir = long / torch.norm(long, dim = 1, keepdim=True)
        short_dir = short /torch.norm(short, dim = 1, keepdim=True)
        self.rot_matrix = torch.cat([out_dir.view(-1, 1, 3), short_dir.view(-1, 1, 3), long_dir.view(-1, 1, 3)], dim = 1)
        self.quaternion_rot = matrix_to_quaternion(self.rot_matrix)
        self.quaternion_rot_invert = quaternion_invert(self.quaternion_rot)

    def canonicalize(self, current_obs):

        #if self.use_pc:
            #current_obs.points[:,:, :3] = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), current_obs.points[:,:, :3]-self.center.view(-1,1,3))

        return current_obs #self.change_obs(part_bbox_tensor_canon, handle_bbox_tensor_canon, current_obs)

    def uncanonicalize(self, actions, current_obs):
        ###增加了observation 的uncanonicalize, 这样储存的observation都是世界坐标系下的坐标
        #actions[:, :3] = (actions[:, :3].view(-1, 1, 3) @ self.rot_matrix.float().permute(0, 2, 1)).view(-1, 3)
        # if self.use_pc:
        #     state = current_obs.state
        #     #current_obs.points[:,:, :3] = quaternion_apply(self.quaternion_rot_invert.view(-1, 1, 4), current_obs.points[:,:, :3])+self.center.view(-1,1,3)
        # else:
        #     state = current_obs

        root_tensor = self.vec_env.task.franka_root_tensor[:, :3]
        hand_pose = relative_pose(self.vec_env.task.franka_root_tensor, self.vec_env.task.hand_rigid_body_tensor).view(self.vec_env.task.env_num, -1)
        # 18-30 hand root relative pose
        # state[:,self.qpose_num:self.qpose_num+13]\
        #     .copy_(hand_pose_)
        # # 31-33 franka root
        # state[:,self.qpose_num+13:self.qpose_num+16]\
        #     .copy_(root_tensor)

        # handle_bbox_tensor = quaternion_apply(self.quaternion_rot_invert.view(-1, 1, 4), handle_bbox_tensor) #@task.rot_matrix
        # part_bbox_tensor = quaternion_apply(self.quaternion_rot_invert.view(-1, 1, 4), part_bbox_tensor) #@task.rot_matrix
        
        if self.vec_env.task.target_part == "door":
            part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
        elif self.vec_env.task.target_part == "drawer":
            part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 1)

        handle_out = handle_bbox_tensor[:, 0] - handle_bbox_tensor[:, 4]
        handle_long = handle_bbox_tensor[:, 1] - handle_bbox_tensor[:, 0]
        handle_short = handle_bbox_tensor[:, 3] - handle_bbox_tensor[:, 0]
        handle_center = (handle_bbox_tensor[:, 0] + handle_bbox_tensor[:, 6]) / 2
        part_out = part_bbox_tensor[:, 0] - part_bbox_tensor[:, 4]
        part_long = part_bbox_tensor[:, 1] - part_bbox_tensor[:, 0]
        part_short = part_bbox_tensor[:, 3] - part_bbox_tensor[:, 0]
        part_center = (part_bbox_tensor[:, 0] + part_bbox_tensor[:, 6]) / 2

        state = torch.zeros((self.vec_env.task.num_envs, 0), device = self.vec_env.task.device)
        if self.vec_env.task.cfg["obs"]["state"]["use_robot_qpose"]: # numActions * 2 # 22 for slider
            robot_qpose = (2 * (self.vec_env.task.franka_dof_tensor[:, :, 0]-self.vec_env.task.franka_dof_lower_limits_tensor[:])/(self.vec_env.task.franka_dof_upper_limits_tensor[:] - self.vec_env.task.franka_dof_lower_limits_tensor[:])) - 1
            robot_qvel = self.vec_env.task.franka_dof_tensor[:, :, 1]
            state = torch.cat((state, robot_qpose, robot_qvel), dim = 1)
        if self.vec_env.task.cfg["obs"]["state"]["use_cabinet_qpose"]: # 2
            cabinet_qpose = self.vec_env.task.cabinet_dof_tensor
            state = torch.cat((state, cabinet_qpose), dim = 1)
        if self.vec_env.task.cfg["obs"]["state"]["use_hand_root_pose"]: # 16
            hand_root_pose = torch.cat((root_tensor, hand_pose), dim=1)
            state = torch.cat((state, hand_root_pose), dim = 1)
        if self.vec_env.task.cfg["obs"]["state"]["use_cabinet_bbox"]:
            self.vec_env.task.part_bbox_tensor, self.vec_env.task.handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0])
            # if self.vec_env.task.target_part == "door":
            #     self.vec_env.task.part_bbox_tensor, self.vec_env.task.handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
            # elif self.vec_env.task.target_part == "drawer":
            #     self.vec_env.task.part_bbox_tensor, self.vec_env.task.handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 1)
            if self.vec_env.task.cfg["obs"]["state"]["use_bbox_form"] == "vertex": # 48
                handle_bbox = handle_bbox_tensor.view(-1, 24)
                part_bbox = part_bbox_tensor.view(-1, 24)
                state = torch.cat((state, handle_bbox, part_bbox), dim = 1)
            elif self.vec_env.task.cfg["obs"]["state"]["use_bbox_form"] == "edges": # 24
                state = torch.cat((state, handle_out, handle_long, handle_short, handle_center, part_out, part_long, part_short, part_center), dim = 1)


        actions[:, :3] = quaternion_apply(self.quaternion_rot_invert, actions[:, :3])
        actions[:, 3:7] = quaternion_multiply(self.quaternion_rot_invert, actions[:, 3:7])

        # if self.use_pc:
        #     current_obs.state = state
        # else:
        #     current_obs = state
        current_obs.state = state
        return actions, current_obs

    # def residual_actions(self, actions, i):

    #     handle_out = self.handle_bbox_tensor[:, 0] - self.handle_bbox_tensor[:, 4]
    #     handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
    #     handle_long = self.handle_bbox_tensor[:, 1] - self.handle_bbox_tensor[:, 0]
    #     handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
    #     handle_short = self.handle_bbox_tensor[:, 3] - self.handle_bbox_tensor[:, 0]
    #     handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)

    #     rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
        
    #     # if self.vec_env.task.target_part == "drawer":
            
    #     #     #print(current_obs[:, -3:])
    #     #     if i < self.max_episode_length / 3:
    #     #         current_obs.state[:, -3:] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.1, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
    #     #     else:
    #     #         current_obs.state[:, -3:] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.5, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]

    #     # else:
    #     #     part_axis_xyz_tensor = self.vec_env.task.part_axis_xyz_tensor
    #     #     part_axis_dir_tensor = self.vec_env.task.part_axis_dir_tensor
    #     #     if i < self.max_episode_length / 3:
    #     #         current_obs.state[:, -3:] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.1, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
    #     #     elif i < self.max_episode_length * 2 / 3:
    #     #         initial_target = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)
    #     #         relative_target = initial_target - part_axis_xyz_tensor
    #     #         angle = 3.1415926 / 12
    #     #         rotation_ = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
    #     #         rotated_relative_target = torch.bmm(rotation_, relative_target.reshape(-1,3,1), ).reshape(-1,3)
    #     #         rotated_target = rotated_relative_target + part_axis_xyz_tensor
    #     #         current_obs.state[:, -3:] = rotated_target
    #     #     else:
    #     #         initial_target = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)
    #     #         relative_target = initial_target - part_axis_xyz_tensor
    #     #         angle = 3.1415926 / 6
    #     #         rotation_ = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
    #     #         rotated_relative_target = torch.bmm(rotation_, relative_target.reshape(-1,3,1), ).reshape(-1,3)
    #     #         rotated_target = rotated_relative_target + part_axis_xyz_tensor
    #     #         current_obs.state[:, -3:] = rotated_target
    #     heuristic_actions = torch.zeros_like(actions)
    #     if self.vec_env.task.target_part == "drawer":
    #         if i < self.max_episode_length / 3:
    #             heuristic_actions[:,:3] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.1, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
    #             heuristic_actions[:, -2:] = (torch.norm(self.handle_bbox_tensor[:,0,:] - self.handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.04)/2
    #         else:
    #             heuristic_actions[:,:3] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.5, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
    #             heuristic_actions[:, -2:] = 0
    #         heuristic_actions[:,3:7] = rotations

    #     else:
    #         part_axis_xyz_tensor = self.vec_env.task.part_axis_xyz_tensor
    #         part_axis_dir_tensor = self.vec_env.task.part_axis_dir_tensor
    #         if i < self.max_episode_length / 3:
    #             heuristic_actions[:,:3] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.1, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
    #             heuristic_actions[:, -2:] = (torch.norm(self.handle_bbox_tensor[:,0,:] - self.handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.04)/2
    #         else:
    #             initial_target = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)
    #             relative_target = initial_target - part_axis_xyz_tensor
    #             angle = (3.14159 / (6 * (self.max_episode_length * 2 / 3))) * (i - self.max_episode_length / 3)
    #             # pdb.set_trace()
    #             rotation_ = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
    #             # rotation_angle = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
    #             quat_ = matrix_to_quaternion(rotation_)
    #             rotated_relative_target = torch.bmm(rotation_, relative_target.reshape(-1,3,1), ).reshape(-1,3)
    #             rotated_target = rotated_relative_target + part_axis_xyz_tensor
    #             heuristic_actions[:,:3] = rotated_target
    #             heuristic_actions[:,3:7] = quaternion_multiply(quat_, rotations)
    #             heuristic_actions[:, -2:] = 0

    #     actions += heuristic_actions

    #     return actions
    
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
                    #self.pre_grasp()
                    for i in range(self.max_episode_length) :
                        if self.apply_reset:
                            current_obs = self.vec_env.reset()
                        if self.use_state_norm and not self.use_pc : ### trick 2
                            current_obs = self.state_norm(current_obs)
                
                        # if self.use_first_bbox is not None:
                        #     if not self.use_pc:
                        #         current_obs[:, self.qpose_num+16:] = self.init_obs[:, self.qpose_num+16:]
                        #     else:
                        #         current_obs.state[:, self.qpose_num+16:] = self.init_obs.state[:, self.qpose_num+16:]
                        #         # current_obs.points[:, self.qpose_num+16:] = self.init_obs.points[:, self.qpose_num+16:]
                            # current_obs[:, self.qpose_num+16:] = self.init_obs[:, self.qpose_num+16:]

                        # if self.use_residual_traj:
                        #     current_obs = self.residual_actions(current_obs, i)

                        if self.canonical:
                            current_obs = self.canonicalize(current_obs)

                        actions, _ = self.actor_critic.act_inference(current_obs)

                        if self.canonical:
                            actions, current_obs = self.uncanonicalize(actions, current_obs)

                        # if self.use_residual_traj:
                        #     actions = self.residual_actions(actions, i)


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
    
        return (train_reward, valIntra_reward, valInter_reward,  train_success, valIntra_success, valInter_success)

    def run(self, num_learning_iterations, log_interval=1):
        ####一开始先判断那些已经collect完事了
        total_done = torch.zeros((self.train_env_num), device=self.device)
        for i in range(self.train_env_num):
            # import pdb
            # pdb.set_trace()
            path = str(self.vec_env.task.selected_asset_path_list[i]).split("/")[-1]
            if self.use_part_pregrasp: 
                path = self.save_path + "/pc_part_pregrasp/{}/{}.tar".format(self.vec_env.task.cfg["task"]["target_part_category"], path)
            else:
                path = self.save_path + "/pc/{}/{}.tar".format(self.vec_env.task.cfg["task"]["target_part_category"], path)
            if os.path.exists(path):
                total_done[i] = 1
                continue
        if total_done.min() > 0:
            return

        if self.canonical:
            self.canonicalize_init()

        # if self.vec_env.task.cfg["env"]["usePreGraspPose"]:
        #     self.pre_grasp()
        current_obs = self.vec_env.reset()

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
        

        current_states = self.vec_env.get_state()   # size (16, 0) ?
        
        if self.is_testing:

            self.vec_env.task.eval()

            _ = self.eval(self.current_learning_iteration)

        else:

            self.vec_env.task.eval()

            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            success_rate = []
            reward_sum = []
            episode_length = []

            #####collect_demo####
            demo_obs = torch.zeros((self.demo_num, self.vec_env.num_envs, self.num_transitions_per_env, *self.observation_space.shape), device = self.device)
            # if not self.use_pc:
            #     demo_pcs = torch.zeros((self.demo_num, self.vec_env.num_envs, self.num_transitions_per_env, *self.pc_feature_shape), device = self.device)
            # else:
            demo_pcs = torch.zeros((self.demo_num, self.vec_env.num_envs, self.num_transitions_per_env, *self.pc_space.shape), device = self.device)
            demo_actions = torch.zeros((self.demo_num, self.vec_env.num_envs, self.num_transitions_per_env, *self.action_space.shape), device = self.device)
            demo_step = torch.zeros((self.vec_env.num_envs), device=self.device, dtype=torch.long)
            demo_reward = torch.zeros((self.demo_num, self.vec_env.num_envs, self.num_transitions_per_env), device=self.device, dtype=torch.float64)
            demo_last_value = torch.zeros((self.demo_num, self.vec_env.num_envs), device=self.device, dtype=torch.float64)
            ####################

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                task_info = {}  
               
                # if it % self.eval_freq == 0:
                #     train_reward, valIntra_reward, valInter_reward,  train_success,  valIntra_success, valInter_success = self.eval(it)
                #     if self.wandb_writer is not None:
                #         self.wandb_writer.log({
                #             "Val/train_reward": train_reward,
                #             "Val/valIntra_reward": valIntra_reward,
                #             "Val/valInter_reward": valInter_reward,
                #             "Val/train_success": train_success,
                #             "Val/valIntra_success": valIntra_success,
                #             "Val/valInter_success": valInter_success,
                #         })
                    
                total_reward = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num), device=self.device)
                total_success = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num), device=self.device)
                self.storage.clear()

                # self.vec_env.task.progress_buf = torch.zeros_like(self.vec_env.task.progress_buf, device=self.device)
                for i in range(self.num_transitions_per_env): 
                    # print(self.vec_env.task.progress_buf.float().mean(), i, self.storage.step)
                    # import pdb
                    # pdb.set_trace()
                    if self.use_state_norm : ### trick 2
                        current_obs = self.state_norm(current_obs)
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        #current_states = self.vec_env.get_state()
                    #print(self.vec_env.task.progress_buf)
                    ####canonicalize
                    # if self.use_first_bbox is not None :
                    #     if not self.use_pc:
                    #         current_obs[:, self.qpose_num+16:] = self.init_obs[:, self.qpose_num+16:]
                    #     else:
                    #         current_obs.state[:, self.qpose_num+16:] = self.init_obs.state[:, self.qpose_num+16:]
                    
                    # if self.use_residual_traj:
                    #     current_obs = self.residual_actions(current_obs, i)
                        
                    if self.canonical: 
                        current_obs = self.canonicalize(current_obs)

                        
                    if self.valIntra_env_num or self.valInter_env_num :
                        raise NotImplementedError
                        current_train_obs = {"state": current_obs.state[:self.train_env_num], "pc": current_obs.points[:self.train_env_num]}
                        current_valIntra_obs = {"state": current_obs.state[self.train_env_num:self.valIntra_env_num + self.train_env_num], "pc": current_obs.points[self.train_env_num:self.valIntra_env_num + self.train_env_num]}
                        current_valInter_obs = {"state": current_obs.state[self.valIntra_env_num + self.train_env_num:], "pc": current_obs.points[self.valIntra_env_num + self.train_env_num:]}
                        current_train_states = current_states[:self.train_env_num]
                        current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                        current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                        # Compute the action
                        train_actions, train_actions_log_prob, train_values, train_mu, train_sigma, _ = self.actor_critic.act(current_train_obs['state'], current_train_states)
                        train_actions = self.actor_critic.act_inference(current_train_obs, current_train_states)
                        valIntra_actions = self.actor_critic.act_inference(current_valIntra_obs)
                        valInter_actions = self.actor_critic.act_inference(current_valInter_obs)
                        actions = torch.cat((train_actions, valIntra_actions, valInter_actions))
                        ######actions un-canonlicalize#####
                        if self.canonical:
                            actions, current_obs = self.uncanonicalize(actions, current_obs)
                        if self.use_residual_traj:
                            actions = self.residual_actions(actions, i)
                        ######actions un-canonlicalize end##### 
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
                        if self.use_pc:
                            features = current_train_obs.points

                        self.storage.add_transitions(
                            current_train_obs.state,
                            current_train_obs.points,
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
                        # print(current_obs.shape, current_states.shape)
                        actions, actions_log_prob, values, mu, sigma, _ = self.actor_critic.act(current_obs)
                        actions, _ = self.actor_critic.act_inference(current_obs)
                        ######actions un-canonlicalize#####
                        if self.canonical:
                            actions, current_obs = self.uncanonicalize(actions, current_obs)

                        if self.use_residual_traj:
                            actions = self.residual_actions(actions, i)
                        ######actions un-canonlicalize end##### 
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
                        self.storage.add_transitions(
                            current_obs.state,
                            current_obs.points[...,:7],
                            current_states,
                            actions,
                            rews, 
                            dones,
                            values, 
                            actions_log_prob,
                            mu,
                            sigma
                        )

                    current_obs = Observations(state=next_obs.state, points=next_obs.points)
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
                
                    total_reward += rews.to(self.device)
                    total_success = torch.logical_or(infos["successes"].to(self.device), total_success)

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _, _= self.actor_critic.act(current_obs)
                # self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)

                ####save_data####
                total_success &= demo_step < self.demo_num
            
                success_num = []
                for i in range(self.train_env_num):
                    if total_success[i]:
                        demo_obs[demo_step[i], i] = self.storage.observations[:, i]
                        demo_actions[demo_step[i], i] = self.storage.actions[:, i]
                        demo_reward[demo_step[i], i] = self.storage.rewards[:, i]
                        demo_last_value[demo_step[i], i] = last_values[:, i]
                        demo_pcs[demo_step[i], i] = self.storage.pointcloud[:, i]
                        success_num.append(i)
                print("##################round{}####################".format(it))
                self.storage.clear()
                demo_step[total_success] += 1
                print("average reward:     ", total_reward / (self.max_episode_length * self.train_env_num))
                print("average success:    ", total_success.float().mean())
                print("success_number:     ",  success_num)
                print("demo_step", demo_step)

                # if demo_step.min() >= self.demo_num or it > 50:
                #     break
                ####save_data####
                if self.use_part_pregrasp:
                    if not os.path.exists(self.save_path + "/pc_part_pregrasp/{}".format(self.vec_env.task.cfg["task"]["target_part_category"])):
                        os.mkdir(self.save_path + "/pc_part_pregrasp/{}".format(self.vec_env.task.cfg["task"]["target_part_category"]))
                else:
                    if not os.path.exists(self.save_path + "/pc/{}".format(self.vec_env.task.cfg["task"]["target_part_category"])):
                        os.mkdir(self.save_path + "/pc/{}".format(self.vec_env.task.cfg["task"]["target_part_category"]))
                for i in range(self.train_env_num):
                    # import pdb
                    # pdb.set_trace()
                    path = str(self.vec_env.task.selected_asset_path_list[i]).split("/")[-1]
                    if self.use_part_pregrasp: 
                        p = self.save_path + "/pc_part_pregrasp_new/{}/{}".format(self.vec_env.task.cfg["task"]["target_part_category"], path)
                    else:
                        p = self.save_path + "/pc/{}/{}".format(self.vec_env.task.cfg["task"]["target_part_category"], path)
                    if os.path.exists(p + ".npy"):
                        total_done[i] = 1
                        continue
                    if demo_step[i] < self.demo_num:
                        continue

                    # import pdb
                    # pdb.set_trace()
                    # torch.save({
                    #         "observations": demo_obs[:, i],
                    #         "actions": demo_actions[:, i],
                    #         "pcs": demo_pcs[:, i],
                    #         "traj_num": self.demo_num,
                    #         }, path)
                    np.save(p, {
                            "observations": demo_obs[:, i].cpu().numpy(),
                            "actions": demo_actions[:, i].cpu().numpy(),
                            "pcs": demo_pcs[:, i].cpu().numpy(),
                            "rewards" : demo_reward[:, i].cpu().numpy(),
                            "last_value" : demo_last_value[:, i].cpu().numpy(),
                            "traj_num": self.demo_num,
                            })
                    print("save_to_path_{}".format(p))
                
                if total_done.min() > 0:
                    break

                if demo_step.min() >= self.demo_num or it > self.demo_num * 3:
                    break
                # stop = time.time()
                # collection_time = stop - start

            #     mean_trajectory_length, mean_reward = self.storage.get_statistics()

            #     # Learning step
            #     start = stop
                
            #     #(self.storage.observations[:,:,14])
            #     mean_value_loss, mean_surrogate_loss = self.update(it)
            #     if self.vec_env.task.task_meta["need_update"] :
            #         task_info = self.vec_env.task.update(it)

            #     self.storage.clear()
            #     stop = time.time()
            #     learn_time = stop - start
            
            #     if self.print_log:
            #         #self.log(locals())
            #         self.imit_log(locals())
            #     if it % log_interval == 0:
            #         self.save(os.path.join(self.log_dir, 'model_{}.tar'.format