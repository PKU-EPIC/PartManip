
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

from algorithms.pregrasp_ppo import RolloutStorage
from algorithms.pregrasp_ppo import RolloutStoragePC
from .module import Normalization, RewardScaling
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

class PREGRASP_PPO:
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
        self.qpose_num = (self.vec_env.task.franka_num_dofs - 4)*2+2 
        
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, asymmetric=asymmetric, qpose_num = self.qpose_num* (1- self.use_qpose_state), device = self.device)

        
        # print(self.actor_critic)
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
        # if self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed' or self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed_newtips' :
        #     self.order_part_pos = 10
        # elif self.vec_env.task.cfg["env"]["robotName"] == 'FrankaSliderXYZ' or self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed_SliderXYZ':
        #     self.order_part_pos = 16
        # else:
        #     self.order_part_pos = 14
        self.order_part_pos = self.qpose_num - 2
            
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
        load(self, path)

    def save(self, path, it):    
        save(self, path, it)
    
    def lr_decay(self, total_steps):
        lr_decay(self, total_steps)

    def pre_grasp(self):
        # import pynvml
        # pynvml.nvmlInit()
        # # 这里的0是GPU id
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(meminfo.used)

        print("start pregrasp")
        actions = torch.zeros((self.train_env_num+self.valIntra_env_num+self.valInter_env_num, self.action_space.shape[0]), device = self.device)
        #print(actions.shape)

        #先向后移动, 防止碰撞
        tar0 = torch.tensor([0.8, 0, 0], device=self.device)+ self.vec_env.task.hand_rigid_body_tensor[:, :3]
        for i in range(200):
            actions[:, :3] = tar0 - self.vec_env.task.hand_rigid_body_tensor[:, :3]
            self.vec_env.control_step(actions)


        if self.use_part_pregrasp:
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
        # print(self.vec_env.task.initial_dof_states.shape)
        # import pdb
        # pdb.set_trace()
        
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

    def change_obs(self, part_bbox_tensor, handle_bbox_tensor, current_obs):
        
        if self.vec_env.task.cfg["task"]["use_bbox"]:
            if not self.use_pc:
                current_obs.state[:, self.qpose_num+16:self.qpose_num+40]\
                    .copy_(handle_bbox_tensor.view(-1, 24))
                current_obs.state[:, self.qpose_num+40:self.qpose_num+16+64]\
                    .copy_(part_bbox_tensor.view(-1, 24))
            else:
                current_obs.state[:, self.qpose_num+16:self.qpose_num+40]\
                    .copy_(handle_bbox_tensor.reshape(-1, 24))
                current_obs.state[:, self.qpose_num+40:self.qpose_num+64]\
                    .copy_(part_bbox_tensor.reshape(-1, 24))

        else:

            handle_out_canon = handle_bbox_tensor[:, 0] -  handle_bbox_tensor[:, 4]
            handle_long_canon =  handle_bbox_tensor[:, 1] -  handle_bbox_tensor[:, 0]
            handle_short_canon =  handle_bbox_tensor[:, 3] -  handle_bbox_tensor[:, 0]
            handle_center_canon = (handle_bbox_tensor[:, 0] + handle_bbox_tensor[:, 6]) / 2
            part_out_canon = part_bbox_tensor[:, 0] - part_bbox_tensor[:, 4]
            part_long_canon =  part_bbox_tensor[:, 1] - part_bbox_tensor[:, 0]
            part_short_canon =  part_bbox_tensor[:, 3] - part_bbox_tensor[:, 0]
            part_center_canon = (part_bbox_tensor[:, 0] + part_bbox_tensor[:, 6]) / 2
            #34-46: part bounding box
            if not self.use_pc:
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

            else:
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

        return current_obs

    def canonicalize(self, current_obs):
        # if not self.use_pc:
        #     current_obs = current_obs_old.clone()
        # else:
        #     current_obs = {}
        #     current_obs.state = current_obs_old["state"].clone()
        #     current_obs.points = current_obs_old["pc"].clone()

        # if self.use_first_bbox is not None:
        #     part_bbox_tensor = self.part_bbox_tensor
        #     handle_bbox_tensor = self.handle_bbox_tensor
        # else:
        #     if self.vec_env.task.target_part == "door":
        #         part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
        #     elif self.vec_env.task.target_part == "drawer":
        #         part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 1)
        
        
        #part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
        #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
        # _draw_line(self.vec_env.task, handle_bbox_tensor[0, 0].cpu(), (handle_bbox_tensor[0, 0]).cpu() + torch.tensor([1, 0, 0]), clear=False)
        # _draw_line(self.vec_env.task, handle_bbox_tensor[0, 0].cpu(), (handle_bbox_tensor[0, 0]).cpu() + torch.tensor([0, 1, 0]), clear=False)
        # _draw_line(self.vec_env.task, handle_bbox_tensor[0, 0].cpu(), (handle_bbox_tensor[0, 0]).cpu() + torch.tensor([0, 0, 1]), clear=False)
    
        # ########hand_pose
        # if not self.use_pc:
        #     hand_pose = current_obs.state[:,self.qpose_num:self.qpose_num+7]   #N * 7
        # else:
        #     hand_pose = current_obs.state[:,self.qpose_num:self.qpose_num+7]   #N * 7

        # hand_pose[:, :3] += self.vec_env.task.franka_root_tensor[:, :3]  
        # hand_pose[:, :3] -= self.center
        # #print(hand_pose.dtype, rot_matrix.float().dtype)
        # #print(hand_pose[:, :3].shape, rot_matrix.shape)
        # #hand_pose[:, :3] = (hand_pose[:, :3].view(-1, 1, 3) @ self.rot_matrix.float()).view(-1, 3)
        # hand_pose[:, :3] = quaternion_apply(self.quaternion_rot, hand_pose[:, :3])
        # #root_new = (self.vec_env.task.franka_root_tensor[:, :3] .view(-1, 1, 3) @ rot_matrix.float()).view(-1, 3)
        # #hand_pose[:, :3] -= root_new
        # #_draw_line(self.vec_env.task, hand_pose[0, :3].cpu(), (hand_grip_dir[0]@rot_matrix[0].float() + hand_pose[0, :3]).cpu(), clear=False)               
        
        # hand_pose[:, 3:7] = quaternion_multiply(self.quaternion_rot, hand_pose[:, 3:7])
        # #######bbox
        # handle_bbox_tensor_canon = handle_bbox_tensor - self.center.view(-1, 1, 3)
        # part_bbox_tensor_canon = part_bbox_tensor - self.center.view(-1, 1, 3)
        # handle_bbox_tensor_canon = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), handle_bbox_tensor_canon) #@self.rot_matrix
        # part_bbox_tensor_canon = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), part_bbox_tensor_canon) #@self.rot_matrix
        # # hand_grip_dir = quat_axis(hand_pose[:, 3:7], 2)
        # # hand_sep_dir = quat_axis(hand_pose[:, 3:7], 1)
        # # hand_down_dir = quat_axis(hand_pose[:, 3:7], 0)
        # #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
        # #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
        # #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor_canon)
        # #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor_canon)
        # #_draw_line(self.vec_env.task, hand_pose[0, :3].cpu(), torch.tensor([0, 0, 0]), clear=False)
        # #_draw_line(self.vec_env.task, hand_pose[0, :3].cpu(), (hand_grip_dir[0] + hand_pose[0, :3]).cpu(), clear=False)
        # #time.sleep(1)
        if self.use_pc:
            # current_obs.state[:,self.qpose_num:self.qpose_num+7] = hand_pose
            # current_obs.state[:,self.qpose_num+13:self.qpose_num+16] =quaternion_apply(self.quaternion_rot.view(-1, 1, 4), (self.vec_env.task.franka_root_tensor[:, :3] - self.center).reshape(-1,1,3)).view(-1, 3)
            current_obs.points[:,:, :3] = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), current_obs.points[:,:, :3]-self.center.view(-1,1,3))
        
        # else:
        #     current_obs.state[:,self.qpose_num:self.qpose_num+7] = hand_pose
        #     #print(current_obs.shape)
        #     #print(self.vec_env.task.franka_root_tensor.shape)
        #     #print(self.quaternion_rot.shape)
        #     #print(self.center.shape)
        #     #print(quaternion_apply(self.quaternion_rot.view(-1, 4), self.vec_env.task.franka_root_tensor[:, :3] - self.center).shape)
        #     current_obs.state[:,self.qpose_num+13:self.qpose_num+16] = quaternion_apply(self.quaternion_rot.view(-1, 4), self.vec_env.task.franka_root_tensor[:, :3] - self.center)
        #     #exit()

        return current_obs #self.change_obs(part_bbox_tensor_canon, handle_bbox_tensor_canon, current_obs)

    def uncanonicalize(self, actions):

        #actions[:, :3] = (actions[:, :3].view(-1, 1, 3) @ self.rot_matrix.float().permute(0, 2, 1)).view(-1, 3)
        actions[:, :3] = quaternion_apply(self.quaternion_rot_invert, actions[:, :3])
        actions[:, 3:7] = quaternion_multiply(self.quaternion_rot_invert, actions[:, 3:7])
        return actions

    def residual_actions(self, current_obs, i):

        handle_out = self.handle_bbox_tensor[:, 0] - self.handle_bbox_tensor[:, 4]
        handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
        handle_long = self.handle_bbox_tensor[:, 1] - self.handle_bbox_tensor[:, 0]
        handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
        handle_short = self.handle_bbox_tensor[:, 3] - self.handle_bbox_tensor[:, 0]
        handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)

        rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
        
        if self.vec_env.task.target_part == "drawer":
            
            #print(urrent_obs.state[:, -3:])
            if i < self.max_episode_length / 3:
                current_obs.state[:, -3:] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.1, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
            else:
                current_obs.state[:, -3:] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.5, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]

        else:
            part_axis_xyz_tensor = self.vec_env.task.part_axis_xyz_tensor
            part_axis_dir_tensor = self.vec_env.task.part_axis_dir_tensor
            if i < self.max_episode_length / 3:
                current_obs.state[:, -3:] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.1, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
            elif i < self.max_episode_length * 2 / 3:
                initial_target = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)
                relative_target = initial_target - part_axis_xyz_tensor
                angle = 3.1415926 / 12
                rotation_ = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
                rotated_relative_target = torch.bmm(rotation_, relative_target.reshape(-1,3,1), ).reshape(-1,3)
                rotated_target = rotated_relative_target + part_axis_xyz_tensor
                current_obs.state[:, -3:] = rotated_target
            else:
                initial_target = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)
                relative_target = initial_target - part_axis_xyz_tensor
                angle = 3.1415926 / 6
                rotation_ = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
                rotated_relative_target = torch.bmm(rotation_, relative_target.reshape(-1,3,1), ).reshape(-1,3)
                rotated_target = rotated_relative_target + part_axis_xyz_tensor
                current_obs.state[:, -3:] = rotated_target

        # if self.vec_env.task.target_part == "drawer":
        #     if i < self.max_episode_length / 3:
        #         heuristic_actions[:,:3] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.1, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
        #         heuristic_actions[:, -2:] = (torch.norm(self.handle_bbox_tensor[:,0,:] - self.handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.04)/2
        #     else:
        #         heuristic_actions[:,:3] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.5, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
        #         heuristic_actions[:, -2:] = 0
        #     heuristic_actions[:,3:7] = rotations

        # else:
        #     part_axis_xyz_tensor = self.vec_env.task.part_axis_xyz_tensor
        #     part_axis_dir_tensor = self.vec_env.task.part_axis_dir_tensor
        #     if i < self.max_episode_length / 3:
        #         heuristic_actions[:,:3] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.1, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
        #         heuristic_actions[:, -2:] = (torch.norm(self.handle_bbox_tensor[:,0,:] - self.handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.04)/2
        #     else:
        #         initial_target = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)
        #         relative_target = initial_target - part_axis_xyz_tensor
        #         angle = (3.14159 / (6 * (self.max_episode_length * 2 / 3))) * (i - self.max_episode_length / 3)
        #         # pdb.set_trace()
        #         rotation_ = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
        #         # rotation_angle = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
        #         quat_ = matrix_to_quaternion(rotation_)
        #         rotated_relative_target = torch.bmm(rotation_, relative_target.reshape(-1,3,1), ).reshape(-1,3)
        #         rotated_target = rotated_relative_target + part_axis_xyz_tensor
        #         heuristic_actions[:,:3] = rotated_target
        #         heuristic_actions[:,3:7] = quaternion_multiply(quat_, rotations)
        #         heuristic_actions[:, -2:] = 0

        # actions += heuristic_actions

        return current_obs
    
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
                        #         current_obs.state[:, self.qpose_num+16:] = self.init_obs[:, self.qpose_num+16:]
                        #     else:
                        #         current_obs.state[:, self.qpose_num+16:] = self.init_obs["state"][:, self.qpose_num+16:]
                        #         # current_obs.points[:, self.qpose_num+16:] = self.init_obs.points[:, self.qpose_num+16:]
                            # current_obs.state[:, self.qpose_num+16:] = self.init_obs[:, self.qpose_num+16:]

                        if self.use_residual_traj:
                            current_obs = self.residual_actions(current_obs, i)

                        if self.canonical:
                            current_obs = self.canonicalize(current_obs)

                        # import time
                        # s = time.time()
                        actions = self.actor_critic.act_inference(current_obs)
                        # t = time.time()
                        # print("total-------------------", t-s)

                        if self.canonical:
                            actions = self.uncanonicalize(actions)

                        # if self.use_residual_traj:
                        #     actions = self.residual_actions(actions, i)


                        next_obs, rews, dones, infos = self.vec_env.step(actions)
                        # next_obs_clouds, next_obs_states, rews, dones, infos = self.vec_env.step(actions)
                        current_obs = Observations(state=next_obs.state, points=next_obs.points)
                      
                        total_reward += rews.to(self.device)
                        total_success[:, r] = torch.logical_or(infos["successes"].to(self.device), total_success[:, r])
                        # if infos["successes"].item() != 0 :

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

        if self.canonical:
            self.canonicalize_init()

        if self.vec_env.task.cfg["env"]["usePreGraspPose"]:
            self.pre_grasp()
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
        

        # import pynvml
        # pynvml.nvmlInit()
        # # 这里的0是GPU id
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(meminfo.used)
        # import pdb
        # pdb.set_trace()

        if self.is_testing:

            self.vec_env.task.eval()

            _ = self.eval(self.current_learning_iteration)

        else:

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
                # print("iter" , it)
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
                    
                # pre-grasp
                #self.pre_grasp()

                # Rollout
            # with torch.no_grad():
            # with 1:
                self.actor_critic.eval() # new
                
                for i in range(self.max_episode_length): 
                    # print("step: ", i)
                    if self.use_state_norm : ### trick 2
                        current_obs = self.state_norm(current_obs)
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        #current_states = self.vec_env.get_state()
                    #print(self.vec_env.task.progress_buf)
                    ####canonicalize
                    # if self.use_first_bbox is not None :
                    #     if not self.use_pc:
                    #         current_obs.state[:, self.qpose_num+16:] = self.init_obs[:, self.qpose_num+16:]
                    #     else:
                    #         current_obs.state[:, self.qpose_num+16:] = self.init_obs["state"][:, self.qpose_num+16:]
                    
                    if self.use_residual_traj:
                        current_obs = self.residual_actions(current_obs, i)

                    if self.canonical: 
                        current_obs = self.canonicalize(current_obs)

                        
                    if self.valIntra_env_num or self.valInter_env_num :
                        if not self.use_pc:
                            current_train_obs = Observations(state=current_obs.state[:self.train_env_num])
                            current_valIntra_obs = Observations(state=current_obs.state[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                            current_valInter_obs = Observations(state=current_obs.state[self.valIntra_env_num + self.train_env_num:])
                            current_train_states = current_states[:self.train_env_num]
                            current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                            current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                        else:
                            current_train_obs = Observations(state=current_obs.state[:self.train_env_num], points= current_obs.points[:self.train_env_num])
                            current_valIntra_obs = Observations(state=current_obs.state[self.train_env_num:self.valIntra_env_num + self.train_env_num], points= current_obs.points[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                            current_valInter_obs = Observations(state=current_obs.state[self.valIntra_env_num + self.train_env_num:], points=current_obs.points[self.valIntra_env_num + self.train_env_num:])
                            current_train_states = current_states[:self.train_env_num]
                            current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                            current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                        # Compute the action
                        # import pdb
                        # pdb.set_trace()
                        train_actions, train_actions_log_prob, train_values, train_mu, train_sigma = self.actor_critic.act(current_train_obs, current_train_states, require_grad = False)
                        valIntra_actions = self.actor_critic.act_inference(current_valIntra_obs)
                        valInter_actions = self.actor_critic.act_inference(current_valInter_obs)
                        actions = torch.cat((train_actions, valIntra_actions, valInter_actions))
                        ######actions un-canonlicalize#####
                        if self.canonical:
                            actions = self.uncanonicalize(actions)

                        # if self.use_residual_traj:
                        #     actions = self.residual_actions(actions, i)
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
                        if not self.use_pc:
                            self.storage.add_transitions(
                                current_train_obs.state,
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
                        actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states, require_grad = False)
                        ######actions un-canonlicalize#####
                        if self.canonical:
                            actions = self.uncanonicalize(actions)
                        # if self.use_residual_traj:
                        #     actions = self.residual_actions(actions, i)
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
                                current_obs.state,
                                current_obs.points,
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
                    torch.cuda.synchronize()

                    if self.print_log:
                        
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    
                    
                    self.actor_critic.train() # new
                    if (i + 1) % self.num_transitions_per_env == 0  and  (i + 1) != self.max_episode_length:
                        _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                        stop = time.time()
                        collection_time = stop - start

                        mean_trajectory_length, mean_reward = self.storage.get_statistics()

                        # Learning step
                        start = stop
                        self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)
                        #(self.storage.observations[:,:,14])
                        mean_value_loss, mean_surrogate_loss = self.update(it)
                        self.storage.clear()
                if self.print_log:
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                self.actor_critic.train() # new
                _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                torch.cuda.synchronize()
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
                torch.cuda.synchronize()
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

    def log_test(self, locs, width=80, pad=35) :
        return log_test(self, locs, width, pad)

    def log(self, locs, width=80, pad=35):
        return log(self, locs, width, pad)

    def update(self, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                
                if self.use_pc:
                    observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                    pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
                    obs_batch = Observations(points = pc_batch, state = observations_batch)
                    # import pdb
                    # pdb.set_trace()
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

