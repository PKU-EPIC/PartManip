import os
import time
from webbrowser import get
import random
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

from algorithms.imitation_learning import RolloutStorage, RolloutStoragePC, RolloutStoragePC_feature, RolloutStorage_expert
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

class behavior_cloning:
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
        self.backbone_freeze = model_cfg["Spconv"]["freeze"]
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
        self.pc_feature_shape = model_cfg["feature_dim"]


        assert(self.max_episode_length % self.num_transitions_per_env == 0)
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, asymmetric=asymmetric, qpose_num = self.qpose_num, device = self.device)
        
        self.actor_critic_expert = actor_critic_expert_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, expert_cfg, asymmetric=asymmetric, qpose_num = self.qpose_num, device = self.device)
       
        self.discriminator = Discriminator(discriminator_cfg, self.action_space.shape, feature_dim=self.pc_feature_shape ,device = self.device)
        
        
        print(self.actor_critic)
        print(self.actor_critic_expert)
        print(self.discriminator)

        self.actor_critic.to(self.device)
        self.actor_critic_expert.to(self.device)
        self.discriminator.to(self.device)
        self.is_testing = is_testing
        self.use_expert = True # use expert policy network or use demostration
        self.model_cfg = model_cfg
        self.expert_cfg = expert_cfg
        self.discriminator_cfg = discriminator_cfg
        self.rew_clip_magnitude =  discriminator_cfg["rew_clip_magnitude"]
        self.mean_rew_policy = 0
        self.mean_rew_expert = 0
        self.use_self_imitation = model_cfg["use_self_imitation"]


        if not self.is_testing: 
            if self.backbone_freeze:
                self.storage = RolloutStoragePC_feature(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                                        self.state_space.shape, self.action_space.shape, self.pc_feature_shape, self.device, sampler)
            else:
                self.storage = RolloutStoragePC(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                                        self.state_space.shape, self.action_space.shape, self.pc_space.shape, self.device, sampler)
            
            if self.use_expert:
                self.expert_storage = RolloutStorage_expert(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
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
        self.discriminator.load_state_dict(checkpoint_dict["discriminator_dict"])
        self.disc_optimizer.load_state_dict(checkpoint_dict["discriminator_dict"])
        self.current_learning_iteration = checkpoint_dict["iteration"]
        self.tot_timesteps = checkpoint_dict["total_steps"]

    def load_expert(self, path):

        #root_path, model_name = os.path.split(path)
        #self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        #self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        #self.actor_critic.train()
        #if self.vec_env.task.task_meta["need_update"] :
        #    self.vec_env.task.load(root_path, self.current_learning_iteration)
        #    self.vec_env.task.train()
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
                
            ############TODO############

            #load expert demostration and put it into deque

    def save(self, path, it):    
        torch.save({
                    'iteration': it + 1,
                    'model_state_dict': self.actor_critic.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'total_steps' : self.tot_timesteps,
                    'discriminator_dict': self.discriminator.state_dict(),
                    "discriminator_optimizer": self.disc_optimizer.state_dict(),
                    }, path)
    
    def lr_decay(self, total_steps):
        lr_now = self.step_size * (1 - total_steps / self.max_iterations)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def pre_grasp(self):
        print("start pregrasp")
        actions = torch.zeros((self.train_env_num+self.valIntra_env_num+self.valInter_env_num, self.action_space.shape[0]), device = self.device)
        #print(actions.shape)
        part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task)  #####use pt box to pregrasp
        pos_pre = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
        pos_tar = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.35, 0, 0], device = self.vec_env.task.device)
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
            print("pregrasp_step_{}".format(step))
            for i in range(self.pregrasp_steps):
                #print(i)    
                # if i == 1:
                #     print(wrap_tensor(self.vec_env.task.gym.acquire_dof_state_tensor(self.vec_env.task.sim)))
                
                #part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
                #part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
                #hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
                #actions[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
                actions[:, :3] = (step/(total_step-1))*(pos_tar - pos_pre)+pos_pre -  self.vec_env.task.hand_rigid_body_tensor[:, :3]
                actions[:, :3] *= math.exp(-0.002 * i)
            
                actions[:, :3] = torch.clamp(actions[:, :3], max=1, min=-1)
                self.vec_env.control_step(actions)
        print("pregrasp_outcome")
        print("part open pose", self.vec_env.task.cabinet_dof_tensor[:,0])
        print("distance to target", torch.norm(pos_tar - self.vec_env.task.hand_rigid_body_tensor[:, :3], dim = 1))
        print("end pregrasp")
        
        self.vec_env.task.initial_dof_states = wrap_tensor(self.vec_env.task.gym.acquire_dof_state_tensor(self.vec_env.task.sim)).clone()
        #print(self.vec_env.task.initial_dof_states.shape)
        
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
                current_obs[:, self.qpose_num+16:self.qpose_num+40]\
                    .copy_(handle_bbox_tensor.view(-1, 24))
                current_obs[:, self.qpose_num+40:self.qpose_num+16+64]\
                    .copy_(part_bbox_tensor.view(-1, 24))
            else:
                current_obs["state"][:, self.qpose_num+16:self.qpose_num+40]\
                    .copy_(handle_bbox_tensor.reshape(-1, 24))
                current_obs["state"][:, self.qpose_num+40:self.qpose_num+64]\
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
                current_obs[:, self.qpose_num+16:self.qpose_num+19]\
                    .copy_(handle_out_canon)
                current_obs[:, self.qpose_num+19:self.qpose_num+22]\
                    .copy_(handle_long_canon)
                current_obs[:, self.qpose_num+22:self.qpose_num+25]\
                    .copy_(handle_short_canon)
                current_obs[:, self.qpose_num+25:self.qpose_num+28]\
                    .copy_(handle_center_canon)
                #46-57: handle bounding box
                current_obs[:, self.qpose_num+28:self.qpose_num+31]\
                    .copy_(part_out_canon)
                current_obs[:, self.qpose_num+31:self.qpose_num+34]\
                    .copy_(part_long_canon)
                current_obs[:, self.qpose_num+34:self.qpose_num+37]\
                    .copy_(part_short_canon)
                current_obs[:, self.qpose_num+37:self.qpose_num+40]\
                    .copy_(part_center_canon)

            else:
                current_obs["state"][:, self.qpose_num+16:self.qpose_num+19]\
                    .copy_(handle_out_canon)
                current_obs["state"][:, self.qpose_num+19:self.qpose_num+22]\
                    .copy_(handle_long_canon)
                current_obs["state"][:, self.qpose_num+22:self.qpose_num+25]\
                    .copy_(handle_short_canon)
                current_obs["state"][:, self.qpose_num+25:self.qpose_num+28]\
                    .copy_(handle_center_canon)
                #46-57: handle bounding box
                current_obs["state"][:, self.qpose_num+28:self.qpose_num+31]\
                    .copy_(part_out_canon)
                current_obs["state"][:, self.qpose_num+31:self.qpose_num+34]\
                    .copy_(part_long_canon)
                current_obs["state"][:, self.qpose_num+34:self.qpose_num+37]\
                    .copy_(part_short_canon)
                current_obs["state"][:, self.qpose_num+37:self.qpose_num+40]\
                    .copy_(part_center_canon)

        return current_obs

    def canonicalize(self, current_obs_old):
        if not self.use_pc:
            current_obs = current_obs_old.clone()
        else:
            current_obs = {}
            current_obs["state"] = current_obs_old["state"].clone()
            current_obs["pc"] = current_obs_old["pc"].clone()

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
        if not self.use_pc:
            hand_pose = current_obs[:,self.qpose_num:self.qpose_num+7]   #N * 7
        else:
            hand_pose = current_obs["state"][:,self.qpose_num:self.qpose_num+7]   #N * 7

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
            current_obs["pc"][:,:, :3] = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), current_obs["pc"][:,:, :3]-self.center.view(-1,1,3))
        
        else:
            current_obs[:,self.qpose_num:self.qpose_num+7] = hand_pose
            #print(current_obs.shape)
            #print(self.vec_env.task.franka_root_tensor.shape)
            #print(self.quaternion_rot.shape)
            #print(self.center.shape)
            #print(quaternion_apply(self.quaternion_rot.view(-1, 4), self.vec_env.task.franka_root_tensor[:, :3] - self.center).shape)
            current_obs[:,self.qpose_num+13:self.qpose_num+16] = quaternion_apply(self.quaternion_rot.view(-1, 4), self.vec_env.task.franka_root_tensor[:, :3] - self.center)
            #exit()

        return self.change_obs(part_bbox_tensor_canon, handle_bbox_tensor_canon, current_obs)

    def uncanonicalize(self, actions):

        #actions[:, :3] = (actions[:, :3].view(-1, 1, 3) @ self.rot_matrix.float().permute(0, 2, 1)).view(-1, 3)
        actions[:, :3] = quaternion_apply(self.quaternion_rot_invert[:self.train_env_num], actions[:, :3])
        actions[:, 3:7] = quaternion_multiply(self.quaternion_rot_invert[:self.train_env_num], actions[:, 3:7])
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
            
            #print(current_obs[:, -3:])
            if i < self.max_episode_length / 3:
                current_obs[:, -3:] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.1, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
            else:
                current_obs[:, -3:] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.5, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]

        else:
            part_axis_xyz_tensor = self.vec_env.task.part_axis_xyz_tensor
            part_axis_dir_tensor = self.vec_env.task.part_axis_dir_tensor
            if i < self.max_episode_length / 3:
                current_obs[:, -3:] = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.1, 0, 0], device = self.vec_env.task.device) - self.vec_env.task.hand_rigid_body_tensor[:, :3]
            elif i < self.max_episode_length * 2 / 3:
                initial_target = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)
                relative_target = initial_target - part_axis_xyz_tensor
                angle = 3.1415926 / 12
                rotation_ = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
                rotated_relative_target = torch.bmm(rotation_, relative_target.reshape(-1,3,1), ).reshape(-1,3)
                rotated_target = rotated_relative_target + part_axis_xyz_tensor
                current_obs[:, -3:] = rotated_target
            else:
                initial_target = (self.handle_bbox_tensor[:,0,:] + self.handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)
                relative_target = initial_target - part_axis_xyz_tensor
                angle = 3.1415926 / 6
                rotation_ = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
                rotated_relative_target = torch.bmm(rotation_, relative_target.reshape(-1,3,1), ).reshape(-1,3)
                rotated_target = rotated_relative_target + part_axis_xyz_tensor
                current_obs[:, -3:] = rotated_target

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
                        #         current_obs[:, self.qpose_num+16:] = self.init_obs[:, self.qpose_num+16:]
                        #     else:
                        #         current_obs["state"][:, self.qpose_num+16:] = self.init_obs["state"][:, self.qpose_num+16:]
                        #         # current_obs["pc"][:, self.qpose_num+16:] = self.init_obs["pc"][:, self.qpose_num+16:]
                            # current_obs[:, self.qpose_num+16:] = self.init_obs[:, self.qpose_num+16:]

                        if self.use_residual_traj:
                            current_obs = self.residual_actions(current_obs, i)

                        # if self.canonical:
                        #     current_obs = self.canonicalize(current_obs)
                        current_obs_new = {"state": current_obs["state"][:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
                        # print(current_obs["pc"].shape)
                        actions, _ = self.actor_critic.act_inference(current_obs_new)

                        # if self.canonical:
                        #     actions = self.uncanonicalize(actions)

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
        
        self.vec_env.task.train()
        self.actor_critic.train()
        return (train_reward, valIntra_reward, valInter_reward,  train_success, valIntra_success, valInter_success)

    def run(self, num_learning_iterations, log_interval=1):

        if self.canonical or self.expert_cfg["canonicalize"]:
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
        
        if self.is_testing:

            self.vec_env.task.eval()

            _ = self.eval(self.current_learning_iteration)

        else:

            self.vec_env.task.train()

            rewbuffer = deque(maxlen=200)
            gt_rewbuffer = deque(maxlen=200)
            disc_loss_all = deque(maxlen=10)
            lenbuffer = deque(maxlen=200)

            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_gt_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            
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
                    #         current_obs["state"][:, self.qpose_num+16:] = self.init_obs["state"][:, self.qpose_num+16:]
                    
                    if self.use_residual_traj:
                        current_obs = self.residual_actions(current_obs, i)
                        
                    # if self.canonical: 
                    #     current_obs = self.canonicalize(current_obs)

                        
                    if self.valIntra_env_num or self.valInter_env_num :
                        # if not self.use_pc:
                        #     current_train_obs = current_obs[:self.train_env_num]
                        #     current_valIntra_obs = current_obs[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                        #     current_valInter_obs = current_obs[self.valIntra_env_num + self.train_env_num:]
                        #     current_train_states = current_states[:self.train_env_num]
                        #     current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                        #     current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                        #     current_train_obs_new = current_train_obs[:, self.qpose_num:self.qpose_num+16]
                        #     current_valIntra_obs_new =  current_valIntra_obs[:, self.qpose_num:self.qpose_num+16]
                        #     current_valInter_obs_new =  current_valInter_obs[:, self.qpose_num:self.qpose_num+16]
                        # else:
                        current_train_obs = {"state": current_obs["state"][:self.train_env_num], "pc": current_obs["pc"][:self.train_env_num]}
                        current_valIntra_obs = {"state": current_obs["state"][self.train_env_num:self.valIntra_env_num + self.train_env_num], "pc": current_obs["pc"][self.train_env_num:self.valIntra_env_num + self.train_env_num]}
                        current_valInter_obs = {"state": current_obs["state"][self.valIntra_env_num + self.train_env_num:], "pc": current_obs["pc"][self.valIntra_env_num + self.train_env_num:]}
                        current_train_states = current_states[:self.train_env_num]
                        current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                        current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                        current_train_obs_new = {"state": current_train_obs["state"][:, self.qpose_num:self.qpose_num+16], "pc": current_train_obs["pc"]}
                        current_valIntra_obs_new =  {"state": current_valIntra_obs["state"][:, self.qpose_num:self.qpose_num+16], "pc": current_valIntra_obs["pc"]}
                        current_valInter_obs_new =  {"state": current_valInter_obs["state"][:, self.qpose_num:self.qpose_num+16], "pc": current_valInter_obs["pc"]}
                        
                        #####use noqpose and no boundingbox observations for actor

                        # Compute the action
                        train_actions, train_actions_log_prob, train_values, train_mu, train_sigma , train_features = self.actor_critic.act(current_train_obs_new, current_train_states)
                        valInter_actions, _ = self.actor_critic.act_inference(current_valInter_obs_new)
                        valIntra_actions, _ = self.actor_critic.act_inference(current_valIntra_obs_new)
                        actions = torch.cat((train_actions, valIntra_actions, valInter_actions))
                        ######actions un-canonlicalize#####
                        # if self.canonical:
                        #     actions = self.uncanonicalize(actions)


                        #####compute expert action@######
                        if self.use_expert:
                            if self.expert_cfg["canonicalize"]:
                                current_obs_expert = self.canonicalize(current_obs)
                            else:
                                current_obs_expert = current_obs

                            expert_actions = self.actor_critic_expert.act_inference(current_obs_expert["state"][:self.train_env_num])
                            
                            if self.expert_cfg["canonicalize"]:
                                expert_actions = self.uncanonicalize(expert_actions)

                        # if self.use_residual_traj:
                        #     actions = self.residual_actions(actions, i)
                        ######actions un-canonlicalize end##### 
                        # Step the vec_environment
                        next_obs, gt_rews, dones, infos = self.vec_env.step(actions)
                        total_success = torch.logical_or(infos["successes"].to(self.device), total_success)

                        # with torch.no_grad():
                        #     disc_logits = self.discriminator.act(current_obs["state"], actions, features).view(-1)
                        #     rews = F.softplus(disc_logits, beta=1)
                        #     if self.rew_clip_magnitude is not None:
                        #         rews = torch.clamp(
                        #             rews, max=self.rew_clip_magnitude, min= -1 * self.rew_clip_magnitude
                        #         )

                        if self.use_expert:

                            self.expert_storage.add_transitions(
                                current_train_obs["state"],
                                train_actions,
                                expert_actions,
                            )

                        train_rews = rews[:self.train_env_num] #* self.model_cfg["disc_reward_perportion"] + gt_rews[:self.train_env_num] 

                        #####ppo trick 3, 4#########
                        if self.use_reward_norm:    
                            rews = self.reward_norm(rews)
                        elif self.use_reward_scaling:
                            rews = self.reward_scaling(rews)

                        next_states = self.vec_env.get_state()
                        dones = dones[:self.train_env_num]

                        # Record the transition
                        if self.backbone_freeze:
                            self.storage.add_transitions(
                                current_train_obs["state"],
                                train_features,
                                current_train_states,
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
                                current_train_obs["state"],
                                current_train_obs["pc"],
                                current_train_states,
                                train_actions,
                                train_rews.clone(), 
                                dones,
                                train_values, 
                                train_actions_log_prob,
                                train_mu,
                                train_sigma
                            )
                    else :
                        if self.use_expert:
                            if self.expert_cfg["canonicalize"]:
                                current_obs_expert = self.canonicalize(current_obs)
                            else:
                                current_obs_expert =current_obs 

                            expert_actions = self.actor_critic_expert.act_inference(current_obs_expert["state"])
                            
                            if self.expert_cfg["canonicalize"]:
                                expert_actions = self.uncanonicalize(expert_actions)

                            
                        # print(current_obs.shape, current_states.shape)
                       
                        current_obs_new = {"state": current_obs["state"][:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
                        actions, actions_log_prob, values, mu, sigma, features = self.actor_critic.act(current_obs_new, current_states)
                        ######actions un-canonlicalize#####
                        if self.canonical:
                            actions = self.uncanonicalize(actions)
                        # if self.use_residual_traj:
                        #     actions = self.residual_actions(actions, i)
                        ######actions un-canonlicalize end##### 
                        next_obs, gt_rews, dones, infos = self.vec_env.step(actions)

                        total_success = torch.logical_or(infos["successes"].to(self.device), total_success)
       
                        # with torch.no_grad():
                        #     disc_logits = self.discriminator.act(current_obs["state"], actions, features).view(-1)
                        #     rews = F.softplus(disc_logits, beta=1)
                        #     if self.rew_clip_magnitude is not None:
                        #         rews = torch.clamp(
                        #             rews, max=self.rew_clip_magnitude, min= -1 * self.rew_clip_magnitude
                        #         )
                        
                        if self.use_expert:

                            self.expert_storage.add_transitions(
                                current_obs["state"],
                                actions,
                                expert_actions,
                            )
                        
                        rews = gt_rews #+ rews * self.model_cfg["disc_reward_perportion"]

                        # trick 3, 4
                        if self.use_reward_norm:    
                            rews = self.reward_norm(rews)
                        elif self.use_reward_scaling:
                            rews = self.reward_scaling(rews)
                            
                        #print(rews)
                        #next_states = self.vec_env.get_state()
                        if self.backbone_freeze:
                            self.storage.add_transitions(
                                current_obs["state"],
                                features,
                                current_states,
                                actions,
                                rews.clone(), 
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
                                rews.clone(), 
                                dones,
                                values, 
                                actions_log_prob,
                                mu,
                                sigma
                            )

                    if self.use_pc:
                        current_obs= next_obs.copy()
                    else:
                        current_obs= next_obs.clone()
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
                    
                    ####update_discriminator
                    # if (i+1) % self.discriminator_cfg["update_freq"] == 0:
                    #     disc_loss = self.discriminator_update()
                    #     if self.use_expert:
                    #         self.expert_storage.clear() 
                    #     disc_loss_all.extend([disc_loss.item()])

                     # Learning step
                    if (i + 1) % self.num_transitions_per_env == 0 and (i + 1) != self.max_episode_length:

                        current_obs_new = {"state": current_obs["state"][:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
                        _, _, last_values, _, _ , _= self.actor_critic.act(current_obs_new, current_states)
                        stop = time.time()

                        mean_trajectory_length, mean_reward = self.storage.get_statistics()

                        # Learning step
                        
                        self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)

                        mean_value_loss, mean_surrogate_loss = self.update(it)

                        if self.vec_env.task.task_meta["need_update"] :
                            task_info = self.vec_env.task.update(it)

                        self.storage.clear()
                        self.expert_storage.clear()
                    
                    if self.print_log:
                        # reward_sum = [x[0] for x in reward_sum]
                        # episode_length = [x[0] for x in episode_length]
                        rewbuffer.extend(reward_sum)
                        gt_rewbuffer.extend(gt_reward_sum)
                        lenbuffer.extend(episode_length)

                current_obs_new = {"state": current_obs["state"][:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
                _, _, last_values, _, _ , _= self.actor_critic.act(current_obs_new, current_states)
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

                #################self imitate, update the demostration buffer##################
                ###############borrowed from Hao Shen, Weikang Wan###################
                # if not self.use_expert and self.use_self_imitation:
                #     for i in range(self.train_env_num):
                #         if total_success[i]:
                #             self.demo_obs[self.demo_step[i], i] = self.storage.observations[:, i]
                #             self.demo_actions[self.demo_step[i], i] = self.storage.actions[:, i]
                #             if self.backbone_freeze:
                #                 self.demo_pcs[self.demo_step[i], i] = self.storage.pointcloud_features[:, i]
                #             else:
                #                 self.demo_pcs[self.demo_step[i], i] = self.storage.pointcloud[:, i]
                    
                #         self.demo_step[i] += 1
                #     self.demo_step = self.demo_step % self.demo_num

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

    def update(self, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for indices in batch:
            if not self.backbone_freeze:
                observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
                observations_batch = observations_batch[:, self.qpose_num:self.qpose_num+16]
                obs_batch = {"pc": pc_batch, "state": observations_batch}
            else:
                observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                pc_batch = self.storage.pointcloud_features.view(-1, *self.storage.pointcloud_features.size()[2:])[indices]
                observations_batch = observations_batch[:, self.qpose_num:self.qpose_num+16]
                obs_batch = {"pc": pc_batch, "state": observations_batch}
            
            if self.asymmetric:
                states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
            else:
                states_batch = None

            expert_actions_batch = self.expert_storage.expert_actions.reshape(-1, self.expert_storage.actions.size(-1))[indices]
            actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
            
            surrogate_loss = self.l2(actions_batch, expert_actions_batch)



            target_values_batch = self.storage.values.view(-1, 1)[indices]
            returns_batch = self.storage.returns.view(-1, 1)[indices]
            old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
            advantages_batch = self.storage.advantages.view(-1, 1)[indices]
            old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
            old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]
            actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch,
                                                                                                                       states_batch,
                                                                                                                       actions_batch)

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


        # for epoch in range(self.num_learning_epochs):
        #     # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
        #     #        in self.storage.mini_batch_generator(self.num_mini_batches):

        #     for indices in batch:
        #         if not self.backbone_freeze:
        #             observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
        #             pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
        #             observations_batch = observations_batch[:, self.qpose_num:self.qpose_num+16]
        #             obs_batch = {"pc": pc_batch, "state": observations_batch}
        #         else:
        #             observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
        #             pc_batch = self.storage.pointcloud_features.view(-1, *self.storage.pointcloud_features.size()[2:])[indices]
        #             observations_batch = observations_batch[:, self.qpose_num:self.qpose_num+16]
        #             obs_batch = {"pc": pc_batch, "state": observations_batch}
                
        #         if self.asymmetric:
        #             states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
        #         else:
        #             states_batch = None
                
        #         actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
        #         target_values_batch = self.storage.values.view(-1, 1)[indices]
        #         returns_batch = self.storage.returns.view(-1, 1)[indices]
        #         old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
        #         advantages_batch = self.storage.advantages.view(-1, 1)[indices]
        #         old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
        #         old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]
        #         actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch,
        #                                                                                                                states_batch,
        #                                                                                                                actions_batch)
        #         ###Trick 1 advantage normalization
        #         if self.use_adv_norm:
        #             advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + self.adv_norm_epsilon)

        #         # KL learning rate schedule
        #         if self.desired_kl > 0 and self.schedule == 'adaptive':

        #             kl = torch.sum(
        #                 sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
        #             kl_mean = torch.mean(kl)

        #             if kl_mean > self.desired_kl * 2.0:
        #                 self.step_size = max(self.lr_lower, self.step_size / 1.5)
        #             elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
        #                 self.step_size = min(self.lr_upper, self.step_size * 1.5)
                    
        #             # if it > 2000 :
        #             #     self.step_size = max(min(self.step_size, 3e-4 - (it-2000)/1000*3e-4), 0.0)

        #             for param_group in self.optimizer.param_groups:
        #                 param_group['lr'] = self.step_size
                
        #         # Surrogate loss
        #         ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        #         surrogate = -torch.squeeze(advantages_batch) * ratio
                
                
        #         surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
        #                                                                            1.0 + self.clip_param)
        #         surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        #         # #!
        #         # surrogate_loss = surrogate.mean()


        #         # Value function loss
        #         if self.use_clipped_value_loss:
        #             value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
        #                                                                                             self.clip_param)
        #             value_losses = (value_batch - returns_batch).pow(2)
        #             value_losses_clipped = (value_clipped - returns_batch).pow(2)
        #             value_loss = torch.max(value_losses, value_losses_clipped).mean()
        #         else:
        #             value_loss = (returns_batch - value_batch).pow(2).mean()
        #         # ipdb.set_trace()
        #         loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

        #         # Gradient step
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         if self.use_grad_clip:   ###trick 7
        #             nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        #         self.optimizer.step()
        #         # self.network_lr_scheduler.step()

        #         mean_value_loss += value_loss.item()
        #         mean_surrogate_loss += surrogate_loss.item()

        # num_updates = self.num_learning_epochs * self.num_mini_batches
        # mean_value_loss /= num_updates
        # mean_surrogate_loss /= num_updates
        
        # if self.learning_rate_decay: 
        #     self.lr_decay(it)

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

