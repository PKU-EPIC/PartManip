import os
import numpy as np
from collections import deque
import random
import collections
from copy import deepcopy
import math
import itertools 
import time
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torch import Tensor
from gym.spaces import Space
from torch.utils.tensorboard import SummaryWriter

from .storage import ReplayBuffer, ReplayBuffer_pc

from .module import MLPActorCritic, Discriminator

from envs.utils.misc import _draw_line
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
    quat_mul, tensor_clamp
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert, quaternion_multiply, quaternion_apply, euler_angles_to_matrix
from isaacgym.gymtorch import wrap_tensor
from envs.utils.get_running_bbox import get_bbox_isaac_tensor, _draw_bbox_tensor, get_bbox_pt, _draw_bbox_pt
from envs.utils.get_reward import quat_axis
from data_structure.observation import Observations

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class sac_il:

    #TODO： now，obs == state ？
    def __init__(self,
                 vec_env,
                 actor_critic = MLPActorCritic,
                 ac_kwargs=dict(),
                 num_transitions_per_env=8,
                 num_learning_epochs=5,
                 num_mini_batches=100,
                 replay_size=100000,
                 gamma=0.99,
                 polyak=0.99,
                 learning_rate=1e-3,
                 max_grad_norm =0.5,
                 entropy_coef=0.2,
                 use_clipped_value_loss=True,
                 reward_scale=1,
                 batch_size=32,
                 device='cpu',
                 sampler='random',
                 log_dir='run',
                 is_testing=False,
                 eval_round=1,
                 eval_freq = 50,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False, 
                
                 wandb_writer = None,
                 checkpoint_path = ' ',

                 canonical = False,
                 canon_space = 'part',

                 model_cfg=None,
                 discriminator_cfg=None,

                 ):

        
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")

        self.vec_env = vec_env
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.train_env_num = vec_env.task.task_meta["train_env_num"] 
        self.valIntra_env_num = vec_env.task.task_meta["valIntra_env_num"]
        self.valInter_env_num = vec_env.task.task_meta["valInter_env_num"]
        self.max_episode_length = vec_env.task.task_meta["max_episode_length"]
        self.use_pc = model_cfg["use_pc"]
        if self.use_pc:
            self.pc_space = vec_env.pc_space
        self.debug = model_cfg["debug"]
        self.backbone_freeze = model_cfg["Spconv"]["freeze"]
        self.device = device
        self.asymmetric = asymmetric
        self.learning_rate = learning_rate
        self.eval_round = eval_round
        self.eval_freq = eval_freq
        model_cfg["task_meta"].update(vec_env.task.task_meta)
        self.pregrasp_steps = vec_env.task.cfg["task"]["pregrasp_steps"]
        self.use_part_pregrasp= vec_env.task.cfg["task"]["use_part_pregrasp"]
        self.init_obs = None
        self.use_first_bbox = vec_env.task.cfg["task"]["use_first_bbox"]

        self.use_qpose_state = self.vec_env.task.cfg["task"]["use_qpose_state"]
        self.qpose_num = (self.vec_env.task.franka_num_dofs - 4)*2+2 
        self.pc_feature_shape = model_cfg["feature_dim"]


        #SAC parameters

        self.num_transitions_per_env = num_transitions_per_env
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.polyak = polyak
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.warm_up = True
        self.policy_update_freq = model_cfg.get("nsteps", self.num_transitions_per_env)

        # SAC components
        self.vec_env = vec_env
        self.actor_critic = actor_critic(vec_env.observation_space, vec_env.action_space, model_cfg=model_cfg, device = self.device,**ac_kwargs).to(self.device)
        
        print(self.actor_critic)
        self.actor_critic_targ = deepcopy(self.actor_critic)
        self.model_cfg = model_cfg
        self.replay_size = replay_size
        if self.debug:
            self.storage = ReplayBuffer(vec_env.num_envs, replay_size, batch_size, num_transitions_per_env, self.observation_space.shape,
                                     self.state_space.shape, self.action_space.shape, self.device, sampler)
        else:
            if self.backbone_freeze:
                self.storage = ReplayBuffer_pc(vec_env.num_envs, replay_size, batch_size, num_transitions_per_env, self.observation_space.shape,
                                     self.state_space.shape, self.action_space.shape, (self.pc_feature_shape,), self.device, sampler)
            else:
                self.storage = ReplayBuffer_pc(vec_env.num_envs, replay_size, batch_size, num_transitions_per_env, self.observation_space.shape,
                                     self.state_space.shape, self.action_space.shape, self.pc_space.shape, self.device, sampler)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters())

        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=self.learning_rate)
        self.q_optimizer = Adam(self.q_params, lr=self.learning_rate)
        
        ###############discriminator#############
        self.discriminator_cfg = discriminator_cfg
        self.rew_clip_magnitude =  discriminator_cfg["rew_clip_magnitude"]
        self.mean_rew_policy = 0
        self.mean_rew_expert = 0
        self.use_self_imitation = model_cfg["use_self_imitation"]
        self.train_asset_num = self.train_env_num // self.vec_env.task.env_per_asset
        self.disc_bc = self.discriminator_cfg.get("disc_bc", self.batch_size)
        # import pdb
        # pdb.set_trace()
        assert(self.disc_bc <= self.batch_size * self.train_env_num)
        self.disc_epoch = self.discriminator_cfg.get("disc_epoch", 1)
        self.disc_update_freq = self.discriminator_cfg.get("nsteps", self.num_transitions_per_env)
        
        self.discriminator = Discriminator(discriminator_cfg, self.action_space.shape, feature_dim=self.pc_feature_shape ,device = self.device)
        
        self.disc_optimizer = Adam(self.discriminator.parameters(), lr=discriminator_cfg["lr"], weight_decay=discriminator_cfg["weight_reg"])
        self.bce = nn.BCEWithLogitsLoss().to(self.device)
        self.disc_gt = torch.cat(
            [
                torch.zeros((self.disc_bc, 1), device=self.device),
                torch.ones((self.disc_bc, 1), device=self.device),
            ],
            dim=0,
        )
        print(self.discriminator)

        #############demo##############
        if not self.debug:
            self.demo_num = model_cfg["demo_num"]
            self.demo_obs = torch.zeros((self.demo_num, self.train_asset_num, self.num_transitions_per_env, *self.observation_space.shape), device = self.device)
        # if not self.debug:
            if self.backbone_freeze:
                self.demo_pcs = torch.zeros((self.demo_num, self.train_asset_num, self.num_transitions_per_env, self.pc_feature_shape), device = self.device)
            else:
                self.demo_pcs = torch.zeros((self.demo_num, self.train_asset_num, self.num_transitions_per_env, *self.pc_space.shape), device = self.device)
        ############only for debug##########
        # else:
        #     self.demo_pcs =None
            self.demo_actions = torch.zeros((self.demo_num, self.train_asset_num, self.num_transitions_per_env, *self.action_space.shape), device = self.device)
            self.demo_step =  torch.zeros((self.train_asset_num), device=self.device, dtype=torch.int)

        self.actor_critic.to(self.device)
        self.discriminator.to(self.device)

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0
        self.checkpoint_path = checkpoint_path
        if self.vec_env.task.cfg["env"]["robotName"] == 'Frankafixed':
            self.order_part_pos = 10
        elif self.vec_env.task.cfg["env"]["robotName"] == 'FrankaSliderXYZ':
            self.order_part_pos = 16
        else:
            self.order_part_pos = 14

        self.apply_reset = apply_reset
        self.wandb_writer = wandb_writer
        self.checkpoint_path = checkpoint_path

        #############canonicalize################
        self.canonical = canonical
        self.canon_space = canon_space
        #print(self.use_residual_traj)
        #exit()
        if self.use_first_bbox == "gt" or self.use_first_bbox is None: 
            self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
        elif self.use_first_bbox == "pt":
            self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_pt(self.vec_env.task)


    def test(self,path):
        # self.actor_critic.load_state_dict(torch.load(path))
        # self.actor_critic.eval()
        assert os.path.exists(self.checkpoint_path)
        checkpoint_dict= torch.load(self.checkpoint_path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
        self.current_learning_iteration = checkpoint_dict["iteration"]
        self.actor_critic.eval()


    def load(self,path):
        assert os.path.exists(self.checkpoint_path)
        checkpoint_dict= torch.load(self.checkpoint_path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
        self.pi_optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
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
        if self.use_part_pregrasp:
            path = os.path.join(path, 'pc_part_pregrasp', '{}'.format(self.vec_env.task.cfg["task"]["target_part_category"]))
        else:
            path = os.path.join(path, 'pc', '{}'.format(self.vec_env.task.cfg["task"]["target_part_category"]))
        for i in range(self.train_env_num // self.vec_env.task.env_per_asset):
            path_i_new = path + '/{}.tar'.format(str(self.vec_env.task.selected_asset_path_list[i]).split("/")[-1])
            print("load demo_{} from ".format(i), path_i_new)
            assert os.path.exists(path_i_new)
            
            
            dict = torch.load(path_i_new, map_location=self.device)
        
            assert(self.demo_num <= dict["traj_num"])
            self.demo_obs[:, i] = dict["observations"][:self.demo_num]
            self.demo_actions[:, i] = dict["actions"][:self.demo_num]
            
            if self.debug:
                continue            
            if self.backbone_freeze:
                self.demo_pcs[:, i] = self.actor_critic.get_features(dict["pcs"][:self.demo_num, :, :, :6].reshape(-1, dict["pcs"].shape[-2], dict["pcs"].shape[-1])).reshape(self.demo_num, self.num_transitions_per_env, -1)
            else:
                self.demo_pcs[:, i] = dict["pcs"][:self.demo_num, :, :, :6]

    def save(self,path, it):
        #torch.save(self.actor_critic.state_dict(),path)
        torch.save({
                    'iteration': it + 1,
                    'model_state_dict': self.actor_critic.state_dict(),
                    'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
                    'q_optimizer_state_dict': self.q_optimizer.state_dict(),
                    'total_steps' : self.tot_timesteps,
                    'discriminator_dict': self.discriminator.state_dict(),
                    "discriminator_optimizer": self.disc_optimizer.state_dict(),
                    }, path)
    
    def pre_grasp(self):
        print("start pregrasp")
        actions = torch.zeros((self.train_env_num+self.valIntra_env_num+self.valInter_env_num, self.action_space.shape[0]), device = self.device)
        #print(actions.shape)

        #先向后移动, 防止碰撞


        ###############new pregrasp#############
        if self.use_part_pregrasp:
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
                    for i in range(self.max_episode_length) :    ######?
                        if self.apply_reset:
                            current_obs = self.vec_env.reset()
                        # import pdb
                        # pdb.set_trace()
                        if self.debug:
                            current_obs_new = current_obs  
                            #  = {"state": current_obs.state[:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
                        else:
                            current_obs_new = Observations(state=current_obs.state[:, self.qpose_num:self.qpose_num+16], points=current_obs.points)
                    
                        # Compute the action
                        actions, _ = self.actor_critic.act(current_obs_new,deterministic =True)
                        # Step the vec_environment
                        next_obs, rews, dones, infos = self.vec_env.step(actions)
                        #next_obs, rews, dones, infos = self.vec_env.step(self.demo_obs[0, :, i])
                        current_obs= Observations(state=next_obs.state, points=next_obs.points)
                        total_reward += rews.to(self.device)
                        total_success[:, r] = torch.logical_or(infos["successes"].to(self.device), total_success[:, r])
                        
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

    def run(self,num_learning_iterations, log_interval = 1):
        """
        the main loop of training.
        :param num_learning_iterations: the maximum number of training steps
        :param log_interval: the frequency of saving model
        :return: None
        """
        if self.canonical:
            self.canonicalize_init()

        if self.vec_env.task.cfg["env"]["usePreGraspPose"]:
            self.pre_grasp()
        current_obs = self.vec_env.reset()
        
        current_states = self.vec_env.get_state()
        if self.is_testing:
            self.vec_env.task.eval()

            _ = self.eval(self.current_learning_iteration)

            # while True:
            #     with torch.no_grad():
            #         if self.apply_reset:
            #             current_obs = self.vec_env.reset()
            #         # Compute the action
            #         if self.debug:
            #             current_obs_new = current_obs  
            #            #  = {"state": current_obs.state[:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
            #         else:
            #             current_obs_new = {"state": current_obs.state[:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
                    
            #         actions = self.actor_critic.act(current_obs_new,deterministic =True)
            #         # Step the vec_environment
            #         next_obs, rews, dones, infos = self.vec_env.step(actions)
            #         current_obs= next_obs.copy()
        else:
            rewbuffer = deque(maxlen=self.num_transitions_per_env)
            lenbuffer = deque(maxlen=self.num_transitions_per_env)
            gt_rewbuffer = deque(maxlen=self.num_transitions_per_env)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_gt_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            
            reward_sum = []
            gt_reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                # import pdb
                # pdb.set_trace()

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

                # Rollout
                for i in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    if self.debug:
                        current_obs_new = current_obs  
                       #  = {"state": current_obs.state[:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
                    else:
                        current_obs_new = Observations(state=current_obs.state[:, self.qpose_num:self.qpose_num+16], points=current_obs.points)
                    actions, features = self.actor_critic.act(current_obs_new)

                    # Step the vec_environment
                    #next_obs, gt_rews, dones, infos = self.vec_env.step(self.demo_actions[0, :, i])
                    next_obs, gt_rews, dones, infos = self.vec_env.step(actions)

                    with torch.no_grad():
                        disc_logits = self.discriminator(current_obs.state, actions, features).view(-1)
                        rews = F.softplus(disc_logits, beta=1)
                        if self.rew_clip_magnitude is not None:
                            rews = torch.clamp(
                                rews, max=self.rew_clip_magnitude, min= -1 * self.rew_clip_magnitude
                            )

                    rews = gt_rews + rews * self.model_cfg["disc_reward_perportion"]

                    rews *= self.reward_scale
                    next_states = self.vec_env.get_state()
                    total_success = torch.logical_or(infos["successes"].to(self.device), total_success)
                    
                    # Record the transition
                    if self.debug:
                        cur_obs = current_obs
                        nex_obs = next_obs
                       #  = {"state": current_obs.state[:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
                    else:
                        if self.backbone_freeze:
                            cur_obs = Observations(state=current_obs.state,points=features)
                            nex_obs = Observations(state=next_obs.state, points=self.actor_critic.get_features(next_obs.points).detach())
                        else:
                            cur_obs = current_obs
                            nex_obs = next_obs
                    
                    self.storage.add_transitions(cur_obs, current_states, actions, rews, nex_obs, dones)
                    # print(self.storage.step, it)
                    current_obs = Observations(state=next_obs.state, points=next_obs.points)
                    current_states.copy_(next_states)
                    
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_gt_reward_sum[:] += gt_rews * self.reward_scale
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        gt_reward_sum.extend(cur_gt_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        cur_gt_reward_sum[new_ids] = 0

                    if self.storage.step >= self.replay_size:
                        self.warm_up = False

                    if self.warm_up == False and (i + 1) % self.policy_update_freq == 0 :
                        # print("update")
                        # time.sleep(10)
                        mean_value_loss, mean_surrogate_loss = self.update()

                    if self.warm_up == False and (i + 1) % self.disc_update_freq == 0 and not self.debug:
                        disc_loss = self.discriminator_update(it)
                    else:
                        disc_loss = 0

                ###############self imitate, update the demostration buffer##################
                ###############borrowed from Hao Shen, Weikang Wan###################
                if self.use_self_imitation :
                    for i in range(self.train_asset_num):
                        for j in range(self.vec_env.task.env_per_asset):
                            if total_success[i*self.vec_env.task.env_per_asset+j]:
                                self.demo_obs[self.demo_step[i], i] = self.storage.observations[self.storage.step-self.num_transitions_per_env:self.storage.step, i]
                                self.demo_actions[self.demo_step[i], i] = self.storage.actions[self.storage.step-self.num_transitions_per_env:self.storage.step, i]
                                if not self.debug:
                                    if self.backbone_freeze:
                                        self.demo_pcs[self.demo_step[i], i] = self.storage.pointcloud[self.storage.step-self.num_transitions_per_env:self.storage.step, i]
                                    else:
                                        self.demo_pcs[self.demo_step[i], i] = self.storage.pointcloud[self.storage.step-self.num_transitions_per_env:self.storage.step, i]
                    
                            self.demo_step[i] += 1
                            self.demo_step = self.demo_step % self.demo_num
                #####################END################

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    gt_rewbuffer.extend(gt_reward_sum)
                    lenbuffer.extend(episode_length)


                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                # TODO: need check the buffer size before update
                # add the update within the interaction loop
                # print(self.warm_up)
                if self.warm_up == False:  ###################?
                    # mean_value_loss, mean_surrogate_loss = self.update()
                    stop = time.time()
                    learn_time = stop - start
                    if self.print_log:
                        self.log(locals())
                    if it % log_interval == 0:
                        self.save(os.path.join(self.log_dir, 'model_{}.tar'.format(it)), it)
                    ep_infos.clear()

            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

        pass

    def log(self, locs, width=80, pad=35):
        """
        print training info
        :param locs:
        :param width:
        :param pad:
        :return:
        """
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        #if locs['ep_infos']:
        #    for key in locs['ep_infos'][0]:
        #        infotensor = torch.tensor([], device=self.device)
        #        for ep_info in locs['ep_infos']:
        #            infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        #        value = torch.mean(infotensor)
        #        self.writer.add_scalar('Episode/' + key, value, locs['it'])
        #        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
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
        
        #mean_std = self.actor_critic.log_std.exp().mean()           
        mean_part_position_train, _ = torch.max(self.storage.observations[:, :self.train_env_num, self.order_part_pos], dim=0)
        mean_part_position_valIntra, _ = torch.max(self.storage.observations[:, self.train_env_num:self.valIntra_env_num + self.train_env_num,self.order_part_pos], dim=0) if self.valIntra_env_num > 0 else (0, 0)
        mean_part_position_valInter, _ = torch.max(self.storage.observations[:, self.valIntra_env_num + self.train_env_num:, self.order_part_pos], dim=0) if self.valInter_env_num > 0 else (0, 0)
        mean_part_position_train = mean_part_position_train.mean()
        mean_part_position_valIntra = mean_part_position_valIntra.mean() if self.valIntra_env_num > 0 else 0
        mean_part_position_valInter = mean_part_position_valInter.mean() if self.valInter_env_num > 0 else 0
        ###wandb###
        if self.wandb_writer is not None :
            self.wandb_writer.log({
                    "total_steps": locs['it'], 
                    'Loss/value_function': locs['mean_value_loss'],
                    'Loss/surrogate': locs['mean_surrogate_loss'],
                    #'Policy/mean_noise_std': mean_std.item(),
                    #'Policy/lr': self.step_size,
                    'Train/mean_reward/step': locs['mean_reward'],
                    'Trian/Mean gt reward/step:': statistics.mean(locs['gt_rewbuffer']) / self.max_episode_length, 
                    #'Train_/mean_episode_length/episode': locs['mean_trajectory_length'],
                    'Train/part_pos_train' : mean_part_position_train,
                    'Train/part_pos_valIntra' : mean_part_position_valIntra,
                    'Train/part_pos_valInter' : mean_part_position_valInter,
                     })
        
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            if self.wandb_writer is not None : self.wandb_writer.log({
                    'Train/reward': torch.tensor(list(collections.deque(locs['rewbuffer']))),
                    'Train/episode_length/time': torch.tensor(list(collections.deque(locs['lenbuffer']))),
                    })
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            #self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        #self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          #f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.3f}\n"""
                          f"""{'Mean gt reward/step:':>{pad}} {(statistics.mean(locs['gt_rewbuffer']) / self.max_episode_length):.3f}\n"""
                          f"""{'Mean expert reward/step:':>{pad}} {self.mean_rew_expert:.3f}\n"""
                          f"""{'Mean policy reward/step:':>{pad}} {self.mean_rew_policy:.3f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""" 
                          f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                          f"""{'Mean_part_position_valIntra:':>{pad}} {mean_part_position_valIntra:.5f}\n"""
                          f"""{'Mean_part_position_valInter:':>{pad}} {mean_part_position_valInter:.5f}\n"""
                          f"""{'discriminator loss:':>{pad}} {locs['disc_loss']:.4f}\n"""
                          )
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.3f}\n"""
                          f"""{'Mean expert reward/step:':>{pad}} {self.mean_rew_expert:.3f}\n"""
                          f"""{'Mean policy reward/step:':>{pad}} {self.mean_rew_policy:.3f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                          f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                          f"""{'Mean_part_position_valIntra:':>{pad}} {mean_part_position_valIntra:.5f}\n"""
                          f"""{'Mean_part_position_valInter:':>{pad}} {mean_part_position_valInter:.5f}\n""" 
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

        # print(self.expert_storage.observations.permute(1, 0, 2).shape)
        # print(self.expert_storage.observations.size(-1))

        disc_ce_loss_mean = 0
        accuracy = 0
        self.mean_rew_policy = 0
        self.mean_rew_expert = 0

        for _ in range(self.disc_epoch):
            if self.storage.fullfill == True:
                subset = random.sample(range(self.storage.replay_size * self.train_env_num), self.disc_bc)
            else:
                subset = random.sample(range(self.storage.step * self.train_env_num), self.disc_bc)

            subset_exp = random.sample(range(self.demo_obs.shape[0]*self.demo_obs.shape[1]*self.demo_obs.shape[2]), self.disc_bc)

            obs_batch = self.storage.observations.reshape(-1, self.storage.observations.size(-1))[subset]
            actions_batch = self.storage.actions.reshape(-1, self.storage.actions.size(-1))[subset]
            pcs_batch = None
            expert_pcs_batch = None
            if not self.debug:
                if self.backbone_freeze:
                    pcs_batch = self.storage.pointcloud.reshape(-1, self.storage.pointcloud.size(-1))[subset]
                    expert_pcs_batch = self.demo_pcs.reshape(-1, self.demo_pcs.size(-1))[subset_exp]
                else:
                    #pcs_batch = self.storage.pointcloud_features.reshape(-1, self.storage.pointcloud.size(-1))
                    pcs_batch = self.storage.pointcloud.reshape(-1, self.storage.pointcloud.size(-2), self.storage.pointcloud.size(-1))[subset]
                    expert_pcs_batch = self.demo_pcs.reshape(-1, self.demo_pcs.size(-2), self.demo_pcs.size(-1))[subset_exp]
            
            expert_obs_batch = self.demo_obs.reshape(-1, self.storage.observations.size(-1))[subset_exp]
            expert_actions_batch = self.demo_actions.reshape(-1, self.storage.actions.size(-1))[subset_exp]
            # print(self.demo_pcs.shape)
            # print(self.storage.pointcloud.size(-2), self.storage.pointcloud_features.size(-1))
                    
            if not self.backbone_freeze and not self.debug:
                expert_pcs_batch_new = torch.zeros((self.disc_bc, self.pc_feature_shape), device = self.device)
                pcs_batch_new = torch.zeros((self.disc_bc, self.pc_feature_shape), device = self.device)
                with torch.no_grad():
                    expert_pcs_batch_new = self.actor_critic.get_features(expert_pcs_batch)####变成feature
                    pcs_batch_new = self.actor_critic.get_features(pcs_batch)
            else:
                expert_pcs_batch_new = expert_pcs_batch
                pcs_batch_new = pcs_batch

            ##############################
            # print(obs_batch.shape, actions_batch.shape, pcs_batch.shape)
            # print(expert_obs_batch.shape, expert_actions_batch.shape, expert_pcs_batch.shape)
            disc_logits_policy = self.discriminator(obs_batch, actions_batch, pcs_batch_new)
            disc_logits_expert = self.discriminator(expert_obs_batch, expert_actions_batch, expert_pcs_batch_new)
            disc_logits = torch.cat((disc_logits_policy, disc_logits_expert), dim = 0)
            disc_preds = (disc_logits > 0).type(disc_logits.data.type())

            self.disc_optimizer.zero_grad()

            if self.discriminator_cfg["use_grad_pen"]:
                
                expert_input = expert_actions_batch
                policy_input = actions_batch

                if self.discriminator_cfg["use_feature"]:
                    expert_input = torch.cat((expert_pcs_batch_new, expert_input), dim = 1)
                    policy_input = torch.cat((pcs_batch_new, policy_input), dim = 1)
                if self.discriminator_cfg["use_gt_obs"]:
                    expert_input = torch.cat((expert_obs_batch, expert_input), dim = 1)
                    policy_input = torch.cat((obs_batch, policy_input), dim = 1)

                eps = torch.rand(expert_input.size(0), 1, device=self.device)
                # print(eps.shape)
                # print(expert_input.shape)
                # import pdb
                interp_obs = eps * expert_input + (1 - eps) * policy_input
                interp_obs = interp_obs.detach()
                interp_obs.requires_grad_(True)
                # pdb.set_trace()
                # print(interp_obs.shape)
                gradients = torch.autograd.grad(
                    outputs=self.discriminator.disc(interp_obs).sum(),
                    inputs=[interp_obs],
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
            #########update discriminator#############
            if it % self.discriminator_cfg["update_freq"] == 0:
                # import pdb
                # pdb.set_trace()
                # torch.autograd.set_detect_anomaly(True)
                disc_ce_loss.backward(retain_graph=True) ####防止gradient graph中断
                self.disc_optimizer.step()

            ###########log##############
            disc_ce_loss_mean += disc_ce_loss.detach()
            accuracy += (disc_preds == self.disc_gt).type(torch.FloatTensor).mean()
        
            self.mean_rew_policy += F.softplus(disc_logits_policy, beta=1).detach().mean()
            self.mean_rew_expert += F.softplus(disc_logits_expert, beta=1).detach().mean()
            


        disc_ce_loss_mean /= self.disc_epoch
        accuracy /= self.disc_epoch
        self.mean_rew_policy /= self.disc_epoch 
        self.mean_rew_expert /= self.disc_epoch 

        if self.wandb_writer is not None:
            self.wandb_writer.log({
                "discriminator/loss": disc_ce_loss_mean, 
                "discriminator/accuracy": accuracy, 
                "discriminator/policy_rews": self.mean_rew_policy, 
                "discriminator/expert_rews": self.mean_rew_expert, 
            })
        
        self.writer.add_scalar('discriminator/loss', disc_ce_loss, it)
        self.writer.add_scalar('discriminator/accuracy', accuracy, it)
        self.writer.add_scalar('discriminator/policy_rews', self.mean_rew_policy, it)
        self.writer.add_scalar('discriminator/expert_rews', self.mean_rew_expert, it)    

        return disc_ce_loss.detach()

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
        #        in self.storage.mini_batch_generator(self.num_mini_batches):
        #TODO: sample a random indice of the batch
        # as now the training uses the whole dataset
        for epoch in range(self.num_learning_epochs):
            # learn_ep = 0
            for indices in batch:
                # learn_ep += 1
                
                # if learn_ep >= self.num_learning_epochs:
                #     break
                pc_batch = None
                nextpc_batch = None
                obs_batch = self.storage.observations.view(-1, self.storage.observations.size()[-1])[indices]
                nextobs_batch = self.storage.next_observations.view(-1, self.storage.observations.size()[-1])[indices]
                if not self.debug: ############Only for debug##############
                    #pc_batch = self.storage.pointcloud_features.view(-1, *self.storage.pointcloud_features.size()[2:])[indices]
                    pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
                    nextpc_batch = self.storage.next_pointcloud.view(-1, *self.storage.next_pointcloud.size()[2:])[indices]
                    
                    obs_batch = obs_batch[:, self.qpose_num:self.qpose_num+16]
                    nextobs_batch = nextobs_batch[:, self.qpose_num:self.qpose_num+16]

                obs_batch = Observations(points=pc_batch, state=obs_batch)
                nextobs_batch = Observations(points=nextpc_batch, state=nextobs_batch)

                if self.asymmetric:
                    states_batch = self.storage.states[indices]
                else:
                    states_batch = None

                actions_batch = self.storage.actions.view(-1, self.storage.actions.size()[-1])[indices]
                rewards_batch = self.storage.rewards.view(-1, 1)[indices]
                dones_batch = self.storage.dones.view(-1, 1)[indices]
                
                data = {'obs': obs_batch,
                        'act':actions_batch,
                        'r':rewards_batch,
                        'obs2':nextobs_batch,
                        'done':dones_batch}

                self.q_optimizer.zero_grad()
                loss_q = self.compute_loss_q(data)
                loss_q.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.q_optimizer.step()

                # Record things
                mean_value_loss += loss_q.item()

                # Freeze Q-networks so you don't waste computational effort
                # computing gradients for them during the policy learning step.
                for p in self.q_params:
                    p.requires_grad = False

                # Next run one gradient descent step for pi.
                self.pi_optimizer.zero_grad()
                loss_pi = self.compute_loss_pi(data)
                loss_pi.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.pi_optimizer.step()

                # Unfreeze Q-networks so you can optimize it at next DDPG step.
                for p in self.q_params:
                    p.requires_grad = True

                # Record things
                mean_surrogate_loss += loss_pi.item()

                # Finally, update target networks by polyak averaging.
                with torch.no_grad():
                    for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_targ.parameters()):
                        # NB: We use an in-place operations "mul_", "add_" to update target
                        # params, as opposed to "mul" and "add", which would make new tensors.
                        p_targ.data.mul_(self.polyak)
                        p_targ.data.add_((1 - self.polyak) * p.data)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # import pdb
        # pdb.set_trace()
        return mean_value_loss, mean_surrogate_loss

    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'],data['act'], data['r'], data['obs2'], data['done']
        if not self.debug:
            if self.backbone_freeze:
                o = torch.cat((o.state, o.points), dim=1)
                o2 = torch.cat((o2.state, o2.points), dim=1)
            else:
                o = torch.cat((o.state, self.actor_critic.get_features(o.points)), dim=1)
                o2 = torch.cat((o2.state, self.actor_critic.get_features(o2.points)), dim=1)
        else:
            o = o.state
            o2 = o2.state
        q1 = self.actor_critic.q1(o, a)
        q2 = self.actor_critic.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor_critic.pi(o2)

            # Target Q-values
            q1_pi_targ = self.actor_critic_targ.q1(o2, a2)
            q2_pi_targ = self.actor_critic_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = (r + self.gamma * (1 - d) * (q_pi_targ - self.entropy_coef * logp_a2))

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
        o = data['obs']
        if not self.debug:
            if self.backbone_freeze:
                o = torch.cat((o.state, o.points), dim=1)
            else:
                o = torch.cat((o.state, self.actor_critic.get_features(o.points)), dim=1)
        else:
            o = o.state
        pi, logp_pi = self.actor_critic.pi(o)

        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.entropy_coef * logp_pi - q_pi).mean()

        return loss_pi



