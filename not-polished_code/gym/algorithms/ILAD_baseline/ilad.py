
from email.policy import strict
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
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler
from algorithms.ILAD_baseline import RolloutStorage, RolloutStoragePC
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
from ..ppo_utils.io_util import load, save, load_backbone_only
from ..ppo_utils.misc_util import lr_decay
from ..ppo_utils.log_util import log, log_test
from data_structure.observation import Observations

import argparse
import pickle
import sys
import gym

class Spec():
    def __init__(self, env=None, env_name=None):
        self.observation_dim = env.observation_space
        self.action_dim = env.action_space
        self.env_id = env_name

        print(f'observation dim: {self.observation_dim}, action dim: {self.action_dim}')

class ILAD:
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

                 max_iterations = 50000,
                 wandb_writer = None,
                 checkpoint_path = ' ',

                 canonical = False,
                 canon_space = 'part',
                 use_residual_traj = False,

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
        self.env_per_asset = vec_env.task.env_per_asset
        self.eval_round = eval_round
        self.eval_freq = eval_freq
        self.numPCObservations = vec_env.task.cfg["obs"]["pc"]["numPCObservations"]
        self.vec_env = vec_env
        self.device = device
        self.desired_kl = desired_kl
        self.lr_upper = float(max_lr)
        self.lr_lower = float(min_lr)
        self.schedule = schedule
        self.step_size = learning_rate
        model_cfg["task_meta"].update(vec_env.task.task_meta)
        self.train_asset_num = self.train_env_num // self.vec_env.task.env_per_asset

        self.init_obs = None
        self.use_first_bbox = vec_env.task.cfg["task"]["use_first_bbox"]

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
        


        assert(self.max_episode_length % self.num_transitions_per_env == 0)
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, device = self.device)

        if model_cfg["ckpt"] != "None" and model_cfg["ckpt"] != None:
            path = model_cfg["ckpt"]
            checkpoint_dict= torch.load(path, map_location=self.device)
            try:
                self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"], strict = False)
            except:
                self.actor_critic.backbone.load_state_dict(checkpoint_dict["model_state_dict"], strict = False)
            print(f"Loading from ckpt {path}")

        self.demo_num = model_cfg["demo_num"]
        self.demo_total_num = torch.zeros(self.train_asset_num, device = self.device)
        self.demo_step =  torch.zeros((self.train_asset_num), device=self.device, dtype=torch.int)
    
        self.optimizer_bc = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.loss_function_bc = torch.nn.MSELoss()

        print(self.actor_critic)
        self.actor_critic.to(self.device)
        self.is_testing = is_testing
        if self.is_testing and model_cfg["save_obs_path"] is not None:
            self.test_save = True
            self.test_save_path = model_cfg["save_obs_path"]
            self.test_obs = []
        else:
            self.test_save = False
        if not self.use_pc:
            self.storage =  RolloutStorage(self.train_env_num, self.max_episode_length, self.observation_space.shape,
                                    self.state_space.shape, self.action_space.shape, self.device, sampler)
        else:
            self.storage = RolloutStoragePC(self.train_env_num, self.max_episode_length, self.observation_space.shape,
                                    self.state_space.shape, self.action_space.shape, self.pc_space.shape, self.device, sampler)
            self.storage_buffer = RolloutStoragePC(self.train_env_num, model_cfg["buffer_length"] * self.max_episode_length, self.observation_space.shape,
                                    self.state_space.shape, self.action_space.shape, self.pc_space.shape, self.device, sampler)
            self.storage_step = 0
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)   # , weight_decay=float(self.weight_decay), trick 9

        
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
        self.order_part_pos = 22
        
        self.canonical = canonical
        self.canon_space = canon_space
        self.episode_step_for_now = 0

        if self.use_first_bbox == "gt": 
            self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
        elif self.use_first_bbox == "pt":
            self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
        else:
            raise NotImplementedError
        
        self.model_cfg = model_cfg

    def load_expert(self, path):

        self.demo_states = torch.zeros((self.train_asset_num, self.demo_num, self.max_episode_length, *self.state_space.shape), device = self.device)
        self.demo_pcs = torch.zeros((self.train_asset_num, self.demo_num, *self.pc_space.shape), device = self.device)
        self.demo_actions = torch.zeros((self.train_asset_num, self.demo_num, self.max_episode_length, *self.action_space.shape), device = self.device)
        self.demo_rewards = torch.zeros((self.train_asset_num, self.demo_num, self.max_episode_length, 1), device = self.device)
        self.demo_last_value = torch.zeros((self.train_asset_num, self.demo_num, 1), device = self.device)
        print(path)
        assert os.path.exists(path)

        path = os.path.join(path, '{}'.format(self.vec_env.task.cfg["task"]["target_part_category"]))
        i = 0
        for _ in range(self.train_env_num // self.vec_env.task.env_per_asset):
            path_i_new = path + '/{}.npy'.format(str(self.vec_env.task.selected_asset_path_list[i]).split("/")[-1])
            if os.path.exists(path_i_new):
                print("load demo_{} from ".format(i), path_i_new)
                dict_load = np.load(path_i_new, allow_pickle=True).item()
                dict = {}
                for k in dict_load.keys():
                    if k != "traj_num":
                        dict[k] = torch.from_numpy(dict_load[k]).to(self.device)
                    else:
                        dict[k] = dict_load[k]
                assert(self.demo_num <= dict["traj_num"])
                self.demo_states[i, :] = dict["observations"][:self.demo_num]
                self.demo_actions[i, :] = dict["actions"][:self.demo_num]
                self.demo_rewards[i, :] = dict["rewards"][:self.demo_num]
                self.demo_last_value[i, :] = dict["last_value"][:self.demo_num]
                self.demo_pcs[i, :] = dict["pcs"][:self.demo_num, 0, :, :7] ##都存成第一步的points
                i += 1
            else:
                print(f"{path_i_new} is not found !")
        print("total_demo_num", i * self.demo_num)

        self.demo_states = self.demo_states[:i]
        self.demo_pcs = self.demo_pcs[:i]
        self.demo_actions = self.demo_actions[:i]
        self.demo_rewards = self.demo_rewards[:i]
        self.demo_last_value = self.demo_last_value[:i] 
        self.demo_pcs = self.demo_pcs.unsqueeze(2).repeat([1, 1, self.max_episode_length, 1, 1])
        self.demo_part_center = self.demo_states[:, :, 0, -3:].unsqueeze(2).repeat([1, 1, self.max_episode_length, 1])


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
        if self.test_save:
            points = (self.test_save[0].points)#.reshape((1, self.test_save[0].points.shape))
            for i in range(1, len(self.save)):
                points.cat((points, self.test_save[0].points))
            points = points.reshape(len(self.save), 200, -1, 6)

    def load(self, path):
        load(self, path)

    def save(self, path, it):    
        save(self, path, it)
    
    def lr_decay(self, total_steps):
        lr_decay(self, total_steps)

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
        
        if self.use_first_bbox == "pt":
            out_dir = torch.tensor([1.0, 0, 0], device=self.device)
            out_dir = out_dir.repeat(self.train_env_num + self.val_env_num, 1)
            long_dir = torch.tensor([0, 0, 1.0], device=self.device)
            long_dir = long_dir.repeat(self.train_env_num + self.val_env_num, 1)
            short_dir = torch.tensor([0, 1.0, 0], device=self.device)
            short_dir = short_dir.repeat(self.train_env_num + self.val_env_num, 1)
        else:
            out_dir = out / torch.norm(out, dim = 1, keepdim=True)
            long_dir = long / torch.norm(long, dim = 1, keepdim=True)
            short_dir = short /torch.norm(short, dim = 1, keepdim=True)

        self.rot_matrix = torch.cat([out_dir.view(-1, 1, 3), short_dir.view(-1, 1, 3), long_dir.view(-1, 1, 3)], dim = 1)
        self.quaternion_rot = matrix_to_quaternion(self.rot_matrix)
        self.quaternion_rot_invert = quaternion_invert(self.quaternion_rot)

    def canonicalize(self, current_obs):

        return current_obs #self.change_obs(part_bbox_tensor_canon, handle_bbox_tensor_canon, current_obs)

    def uncanonicalize(self, actions):
        
        if self.vec_env.task.cfg["env"]["driveMode"] in ["ik_rela", "ik_abs"]:
            actions[:, :3] = quaternion_apply(self.quaternion_rot_invert, actions[:, :3])
            actions[:, 3:7] = quaternion_multiply(self.quaternion_rot_invert, actions[:, 3:7])
        elif self.vec_env.task.cfg["env"]["driveMode"] in ["ik_rela_axis_angle", "ik_abs_axis_angle"]:
            actions[:, :3] = quaternion_apply(self.quaternion_rot_invert, actions[:, :3])
            actions[:, 3:6] = quaternion_apply(self.quaternion_rot_invert, actions[:, 3:6])
        
        return actions
    
    def eval(self, it) :
        
        self.vec_env.task.eval()
        self.actor_critic.eval()
        current_obs = self.vec_env.reset()
        self.episode_step_for_now = 0
        total_reward = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num), device=self.device)
        total_success = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num, self.eval_round), device=self.device)
        
        with tqdm(total=self.eval_round) as pbar:
            pbar.set_description('Validating:')
            with torch.no_grad() :
                for r in range(self.eval_round) :
                    for i in range(self.max_episode_length) :
                        # import pdb
                        # pdb.set_trace()
                        if self.apply_reset:
                            current_obs = self.vec_env.reset()
                            self.episode_step_for_now = 0

                        if self.canonical:
                            current_obs = self.canonicalize(current_obs)

                        # import time
                        # s = time.time()
                        actions, _ = self.actor_critic.act_inference(current_obs)
                        # t = time.time()
                        # print("total-------------------", t-s)

                        if self.canonical:
                            actions = self.uncanonicalize(actions)

                        # if self.use_residual_traj:
                        #     actions = self.residual_actions(actions, i)
                        # next_obs = current_obs
                        # current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)
                        if self.test_save:
                            self.test_obs.append(current_obs)

                        next_obs, rews, dones, infos = self.vec_env.step(actions)
                        # next_obs_clouds, next_obs_states, rews, dones, infos = self.vec_env.step(actions)
                        current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)
                      
                        total_reward += rews.to(self.device)
                        total_success[:, r] = torch.logical_or(infos["successes"].to(self.device), total_success[:, r])
                        # if infos["successes"].item() != 0 :
                    # import pdb
                    # pdb.set_trace()
                    pbar.update(1)
                    
        if self.test_save:
            
            points = self.test_obs[0].points[..., :self.numPCObservations] #.reshape((1, self.test_save[0].points.shape))
            total =len(self.test_obs)
            for i in range(1, total):
                points= torch.cat((points, self.test_obs[i].points[..., :self.numPCObservations]))
            points = points.reshape(len(self.test_obs)//200, 200, self.test_obs[0].points.shape[0], self.test_obs[0].points.shape[1], 6)
            import pdb
            pdb.set_trace()
            torch.save(points, self.test_save_path)
        
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
        print("per eval_round success")
        print(torch.mean(total_success[:self.train_env_num].float(), dim = 0))
        print(torch.mean(total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float(), dim = 0))
        print(torch.mean(total_success[self.valIntra_env_num + self.train_env_num:].float(), dim = 0))
        print("per asset success")
        print(torch.mean(total_success[:self.train_env_num].float().reshape(self.train_env_num//self.env_per_asset, self.env_per_asset * self.eval_round),dim = 1))
        print(torch.mean(total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float().reshape(self.valIntra_env_num//self.env_per_asset, self.env_per_asset * self.eval_round), dim = 1))
        print(torch.mean(total_success[self.valIntra_env_num + self.train_env_num:].float().reshape(self.valInter_env_num//self.env_per_asset, self.env_per_asset * self.eval_round), dim = 1))
        asset_train_mean = torch.mean(total_success[:self.train_env_num].float().reshape(self.train_env_num//self.env_per_asset, self.env_per_asset * self.eval_round),dim = 1)
        if self.is_testing:
            not_success = []
            for i in range(self.train_env_num // self.env_per_asset):
                if (asset_train_mean[i] <= 0.8):
                    not_success.append(i)
            print("not success train assert num")
            print(not_success)

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
        self.episode_step_for_now = 0 

        if self.is_testing:

            self.vec_env.task.eval()

            _ = self.eval(self.current_learning_iteration)

        else:

            if self.model_cfg["use_bc_pretrain"]:
                self.pretrain_bc()

            self.vec_env.task.train()

            rewbuffer = deque(maxlen=200)
            lenbuffer = deque(maxlen=200)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            success_rate = []
            reward_sum = []
            episode_length = []
            self.warm_up = True

            for it in range(self.current_learning_iteration, num_learning_iterations):
                current_obs = self.vec_env.reset()
                init_points = current_obs.points
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
                    self.save(os.path.join(self.log_dir, 'model_{}.tar'.format(it)), it)

                for i in range(self.max_episode_length):
                    self.episode_step_for_now = i
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        self.episode_step_for_now = 0
                        #current_states = self.vec_env.get_state()
                    #print(self.vec_env.task.progress_buf)
                    ####canonicalize
                    if self.canonical: 
                        current_obs = self.canonicalize(current_obs)

                        
                    # if self.valIntra_env_num or self.valInter_env_num :
                    if not self.use_pc:
                        current_train_obs = Observations(state=current_obs.state[:self.train_env_num], obs=current_obs.obs[:self.train_env_num])
                        if self.valIntra_env_num:
                            current_valIntra_obs = Observations(state=current_obs.state[self.train_env_num:self.valIntra_env_num + self.train_env_num], obs=current_obs.obs[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                        if self.valIntra_env_num:
                            current_valInter_obs = Observations(state=current_obs.state[self.valIntra_env_num + self.train_env_num:], obs=current_obs.obs[self.valIntra_env_num + self.train_env_num:])
                        # current_train_states = current_states[:self.train_env_num]
                        # current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                        # current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                    else:
                        current_train_obs = Observations(state=current_obs.state[:self.train_env_num], points= init_points[:self.train_env_num], obs=current_obs.obs[:self.train_env_num])
                        if self.valIntra_env_num:
                            current_valIntra_obs = Observations(state=current_obs.state[self.train_env_num:self.valIntra_env_num + self.train_env_num],obs=current_obs.obs[self.train_env_num:self.valIntra_env_num + self.train_env_num], points= init_points[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                        if self.valIntra_env_num:
                            current_valInter_obs = Observations(state=current_obs.state[self.valIntra_env_num + self.train_env_num:],obs=current_obs.obs[self.valIntra_env_num + self.train_env_num:], points=init_points[self.valIntra_env_num + self.train_env_num:])
                        # current_train_states = current_states[:self.train_env_num]
                        # current_valIntra_states = current_states[self.train_env_num:self.valIntra_env_num + self.train_env_num]
                        # current_valInter_states = current_states[self.valIntra_env_num + self.train_env_num:]
                    # Compute the action
                    # import 
                    # pdb.set_trace()
                    train_actions, train_actions_log_prob, train_values, train_mu, train_sigma, _ = self.actor_critic.act(current_train_obs, require_grad = False)
                    actions = train_actions
                    if self.valIntra_env_num:
                        valIntra_actions, _ = self.actor_critic.act_inference(current_valIntra_obs)
                        actions = torch.cat((actions, valIntra_actions))
                    if self.valIntra_env_num:
                        valInter_actions, _ = self.actor_critic.act_inference(current_valInter_obs)
                        actions = torch.cat((actions, valInter_actions))
                    
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

                    next_states = self.vec_env.get_state()
                    train_rews = rews[:self.train_env_num]
                    train_dones = dones[:self.train_env_num]
                    # Record the transition
                    if not self.use_pc:
                        # import pdb
                        
                        # pdb.set_trace()
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
                    
                        
                    current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=init_points)
                    
                    #current_states.copy_(next_states)
                    # Book keeping

                    ep_infos.append(infos)
                    torch.cuda.synchronize()

                    if self.print_log:
                        # import pdb
                        # pdb.set_trace()
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # if new_ids.sum() != 0:
                            # pdb.set_trace()

                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    

                    self.actor_critic.train() # new
                    if (i + 1) % self.num_transitions_per_env == 0  and  (i + 1) != self.max_episode_length and not self.warm_up:
                        _, _, last_values, _, _, _ = self.actor_critic.act(current_obs)
                        torch.cuda.synchronize()
                        stop = time.time()
                        collection_time = stop - start

                        # mean_trajectory_length, mean_reward = self.storage.get_statistics()
                        mean_reward = self.storage.rewards.mean()
                        # Learning step
                        start = stop
                        # self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)
                        #(self.storage.observations[:,:,14])
                        mean_value_loss, mean_surrogate_loss = self.update(it)
                        # self.storage.clear()
                        torch.cuda.synchronize()
                        stop = time.time()
                        learn_time = stop - start
                        start = stop
                        if self.print_log:
                            self.log(locals())

                if self.print_log:
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                self.actor_critic.train() # new
                _, _, last_values, _, _ , _= self.actor_critic.act(current_obs)
                torch.cuda.synchronize()
                stop = time.time()
                collection_time = stop - start

                # mean_trajectory_length, mean_reward = self.storage.get_statistics()
                mean_reward = self.storage.rewards.mean()
                # Learning step
                start = stop
                self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)
                ####ilad put storage to storage buffer#####

                self.storage_buffer.observations[self.storage_step * self.max_episode_length:(self.storage_step + 1) * self.max_episode_length] = \
                    self.storage.observations
                self.storage_buffer.actions[self.storage_step * self.max_episode_length:(self.storage_step + 1) * self.max_episode_length] = \
                    self.storage.actions
                self.storage_buffer.states[self.storage_step * self.max_episode_length:(self.storage_step + 1) * self.max_episode_length] = \
                    self.storage.states
                self.storage_buffer.rewards[self.storage_step * self.max_episode_length:(self.storage_step + 1) * self.max_episode_length] = \
                    self.storage.rewards
                self.storage_buffer.values[self.storage_step * self.max_episode_length:(self.storage_step + 1) * self.max_episode_length] = \
                    self.storage.values
                self.storage_buffer.returns[self.storage_step * self.max_episode_length:(self.storage_step + 1) * self.max_episode_length] = \
                    self.storage.returns
                self.storage_buffer.advantages[self.storage_step * self.max_episode_length:(self.storage_step + 1) * self.max_episode_length] = \
                    self.storage.advantages
                self.storage_buffer.mu[self.storage_step * self.max_episode_length:(self.storage_step + 1) * self.max_episode_length] = \
                    self.storage.mu
                self.storage_buffer.sigma[self.storage_step * self.max_episode_length:(self.storage_step + 1) * self.max_episode_length] = \
                    self.storage.sigma
                if self.use_pc:
                    self.storage_buffer.pointcloud[self.storage_step * self.max_episode_length:(self.storage_step + 1) * self.max_episode_length] = \
                        self.storage.pointcloud

                self.storage_step = self.storage_step + 1
                if self.storage_step >= self.model_cfg["buffer_length"]:
                    self.warm_up = False
                self.storage_step = self.storage_step % self.model_cfg["buffer_length"]

                ####
                if it % self.model_cfg["update_pn_interval"] == 0 and not self.warm_up:
                    self.update_pn(it)
                if not self.warm_up:
                    #(self.storage.observations[:,:,14])
                    mean_value_loss, mean_surrogate_loss = self.update(it)
                    if self.vec_env.task.task_meta["need_update"] :
                        task_info = self.vec_env.task.update(it)
                else:
                    print("warmup:", it)



                self.storage.clear()
                torch.cuda.synchronize()
                stop = time.time()
                learn_time = stop - start
                start = stop
            
                if self.print_log and not self.warm_up:
                    self.log(locals())
                if it % log_interval == 0 and self.train_env_num > 1:
                    if self.vec_env.task.task_meta["need_update"] :
                        self.vec_env.task.save(self.log_dir, it)
                ep_infos.clear()

            if self.vec_env.task.task_meta["need_update"] :
                self.vec_env.task.save(self.log_dir, num_learning_iterations)

    def pretrain_bc(self):
        ap = self.actor_critic.actor_mlp.state_dict()
        cp = self.actor_critic.critic_mlp.state_dict()
        self.actor_critic.backbone.unfreeze()
        demo_states = self.demo_states.reshape(-1, *self.state_space.shape)
        demo_pcs = self.demo_pcs.reshape(-1, *self.pc_space.shape)
        demo_actions = self.demo_actions.reshape(-1, *self.action_space.shape)
        demo_part_center = self.demo_part_center.reshape(-1, 3)
        num_samples = demo_states.shape[0]

        for ep in tqdm(range(self.model_cfg["bc_pretrain_epochs"])):
            l = int(num_samples / self.model_cfg["mb_size"])
            for mb in range(l):
                rand_idx = torch.from_numpy(np.random.choice(num_samples, size=self.model_cfg["mb_size"]))
                obs = Observations(obs=demo_states[rand_idx], state=demo_states[rand_idx], points=demo_pcs[rand_idx])
                obs_new = torch.cat((obs.obs[:, :22], obs.obs[:, 24:40], demo_part_center[rand_idx]), dim=1)
                obs.obs = obs_new
                act = demo_actions[rand_idx]
                act_pred, _, _ = self.actor_critic.act_dagger(obs)
                self.optimizer_bc.zero_grad()
                loss = self.loss_function_bc(torch.tanh(act), torch.tanh(act_pred))
                loss.backward()
                self.optimizer_bc.step()
                ###Log###
                print("loss:", loss)
                self.writer.add_scalar("BC/" + "loss", loss, ep * l + mb)
                if self.wandb_writer is not None:
                    self.wandb_writer.log({
                        "BC/" + "loss", loss
                    })
                ###Log end###
        self.actor_critic.backbone.freeze()

    def log_test(self, locs, width=80, pad=35) :
        return log_test(self, locs, width, pad)

    def log(self, locs, width=80, pad=35):
        return log(self, locs, width, pad)

    def update(self, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        ###compute demo_state####
        demo_states = self.demo_states.reshape(-1, self.max_episode_length, *self.state_space.shape)
        demo_pcs = self.demo_pcs.reshape(-1, self.max_episode_length, *self.pc_space.shape)
        demo_actions = self.demo_actions.reshape(-1, self.max_episode_length, *self.action_space.shape)
        demo_part_center = self.demo_part_center.reshape(-1, self.max_episode_length, 3)
        demo_last_value = self.demo_last_value.reshape(-1, 1)
        demo_rewards = self.demo_rewards.reshape(-1, self.max_episode_length, 1)
        demo_obs = Observations(obs=demo_states, state=demo_states, points=demo_pcs)
        # import pdb
        # pdb.set_trace()
        obs_new = torch.cat((demo_obs.obs[:, :, :22], demo_obs.obs[:, :, 24:40], demo_part_center), dim=2)
        demo_obs.obs = obs_new
        num_samples = self.train_asset_num * self.demo_num
        ###compute demo state end ####

        batch = self.storage_buffer.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                if self.use_pc:
                    state_batch = self.storage_buffer.states.view(-1, *self.storage_buffer.states.size()[2:])[indices]
                    pc_batch = self.storage_buffer.pointcloud.view(-1, *self.storage_buffer.pointcloud.size()[2:])[indices]
                    observations_batch = self.storage_buffer.observations.view(-1, *self.storage_buffer.observations.size()[2:])[indices]
                    obs_batch = Observations(points = pc_batch, state = state_batch, obs = observations_batch)
                else:
                    state_batch = self.storage_buffer.states.view(-1, *self.storage_buffer.states.size()[2:])[indices]
                    observations_batch = self.storage_buffer.observations.view(-1, *self.storage_buffer.observations.size()[2:])[indices]
                    obs_batch = Observations(state = state_batch, obs=observations_batch)
                actions_batch = self.storage_buffer.actions.view(-1, self.storage_buffer.actions.size(-1))[indices]
                target_values_batch = self.storage_buffer.values.view(-1, 1)[indices]
                returns_batch = self.storage_buffer.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage_buffer.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage_buffer.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage_buffer.mu.view(-1, self.storage_buffer.actions.size(-1))[indices]
                old_sigma_batch = self.storage_buffer.sigma.view(-1, self.storage_buffer.actions.size(-1))[indices]
                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch, _ = self.actor_critic.evaluate(obs_batch, actions_batch)

                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() +1e-8)

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

                ##surrogate loss for demo###

                # rand_idx = torch.from_numpy(np.random.choice(num_samples, size=self.model_cfg["traj_sample_num"]))

 
                rand_idx = torch.from_numpy(np.random.choice(num_samples, size=self.model_cfg["mb_size"] // self.max_episode_length))
                demo_obs_batch = Observations(obs=demo_obs.obs[rand_idx].reshape(-1, *self.state_space.shape), \
                                            state=demo_obs.obs[rand_idx].reshape(-1, *self.state_space.shape), \
                                            points=demo_pcs[rand_idx].reshape(-1, *self.pc_space.shape))
                demo_rewards_batch = demo_rewards[rand_idx]
                demo_last_value_batch = demo_last_value[rand_idx]
                # import pdb
                # pdb.set_trace()
                demo_actions_log_prob, _, demo_value, _  = self.actor_critic.evaluate(demo_obs_batch, demo_actions[rand_idx])

                demo_actions_log_prob_batch = demo_actions_log_prob.reshape(self.model_cfg["mb_size"] // self.max_episode_length, self.max_episode_length, 1)
                demo_value_batch = demo_value.reshape(self.model_cfg["mb_size"] // self.max_episode_length, self.max_episode_length, 1)
                #q_batch = q.reshape(self.model_cfg["mb_size"] // self.max_episode_length, self.max_episode_length, 1)
                ###lk
                traj_l = -1 * torch.mean(demo_actions_log_prob_batch, dim = 2)
                traj_l_max = torch.max(traj_l)
                traj_l_min = torch.min(traj_l)
                w = (traj_l - traj_l_min) / (traj_l_max - traj_l_min + 1e-9)
                ###adv
                advantage = 0
                returns = torch.zeros(self.model_cfg["mb_size"] //self.max_episode_length,  self.max_episode_length, 1, device=self.device)
                for step in reversed(range(self.num_transitions_per_env)):
                    if step == self.num_transitions_per_env - 1:
                        next_values = demo_last_value_batch
                        next_is_not_terminal = 0
                    else:
                        next_values = demo_value_batch[:, step + 1]
                        next_is_not_terminal = 1.0
                    delta = demo_rewards_batch[:, step] + next_is_not_terminal * self.gamma * next_values - demo_value_batch[:, step]
                    advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
                    returns[:, step] = advantage + demo_value_batch[:, step]
            
                a = returns - demo_value_batch
                a = (a - a.mean()) / (a.std() +1e-8) #normalize advantage

                value_loss = torch.mean(self.model_cfg["lambda0"] * math.pow(self.model_cfg["lambda1"], epoch) * w) + \
                                    torch.mean(self.model_cfg["lambda0_"] * (1 - math.pow(self.model_cfg["lambda1"], epoch)) * a)
                
                # Value function loss
                # if self.use_clipped_value_loss:
                #     value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                #                                                                                     self.clip_param)
                #     value_losses = (value_batch - returns_batch).pow(2)
                #     value_losses_clipped = (value_clipped - returns_batch).pow(2)
                #     value_loss = torch.max(value_losses, value_losses_clipped).mean()
                # else:
                #     value_loss = (returns_batch - value_batch).pow(2).mean()
                # ipdb.set_trace()
                # loss = surrogate_loss + self.value_loss_coef * value_loss + demo_surrogate_loss# - self.entropy_coef * entropy_batch.mean()
                loss = surrogate_loss + value_loss
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.network_lr_scheduler.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        
        # if self.learning_rate_decay: 
        #     self.lr_decay(it)

        return mean_value_loss, mean_surrogate_loss
    
    def update_pn(self, it):
        print("---------------------start update pn--------------------------")
        ap = self.actor_critic.actor_mlp.state_dict()
        cp = self.actor_critic.critic_mlp.state_dict()
        self.actor_critic.backbone.unfreeze()
        self.actor_critic.backbone.train()
        num_samples = self.storage_buffer.observations.view(-1, *self.storage_buffer.observations.size()[2:]).shape[0]
        for ep in range(self.model_cfg["bc_epochs"]):   
            l = int(num_samples / self.model_cfg["mb_size"])
            for mb in range(l):    
                indices = torch.from_numpy(np.random.choice(num_samples, size=self.model_cfg["mb_size"]))
                if self.use_pc:
                    state_batch = self.storage_buffer.states.view(-1, *self.storage_buffer.states.size()[2:])[indices]
                    pc_batch = self.storage_buffer.pointcloud.view(-1, *self.storage_buffer.pointcloud.size()[2:])[indices]
                    observations_batch = self.storage_buffer.observations.view(-1, *self.storage_buffer.observations.size()[2:])[indices]
                    obs_batch = Observations(points = pc_batch, state = state_batch, obs = observations_batch)
                else:
                    state_batch = self.storage_buffer.states.view(-1, *self.storage_buffer.states.size()[2:])[indices]
                    observations_batch = self.storage_buffer.observations.view(-1, *self.storage_buffer.observations.size()[2:])[indices]
                    obs_batch = Observations(state = state_batch, obs=observations_batch)
                actions_batch = self.storage_buffer.actions.view(-1, self.storage_buffer.actions.size(-1))[indices]
        
                act_pred, _, _ = self.actor_critic.act_dagger(obs_batch)
                self.optimizer_bc.zero_grad()
                loss = self.loss_function_bc(torch.tanh(actions_batch), torch.tanh(act_pred))
                loss.backward()
                self.optimizer_bc.step()
                self.actor_critic.actor_mlp.load_state_dict(ap)
                self.actor_critic.critic_mlp.load_state_dict(cp)
                ###Log###
                if l == 0:
                    print("loss:", loss)
                self.writer.add_scalar("BC/" + "loss", loss, ep * l + mb + it * l * self.model_cfg["bc_epochs"])
                if self.wandb_writer is not None:
                    self.wandb_writer.log({
                        "BC/" + "loss", loss
                    })
        self.actor_critic.backbone.freeze()
        self.actor_critic.backbone.eval()
        

def space_add(a, b):

    if len(a.shape) != 1 or len(b.shape) != 1 :
        
        raise TypeError("Shape of two spaces need to be 1d")
    
    elif not isinstance(a, Box) or not isinstance(b, Box) :

        raise TypeError("Type of two spaces need to be Box")
    
    else :

        low = np.concatenate((a.low, b.low))
        high = np.concatenate((a.high, b.high))
        return Box(low=low, high=high)

