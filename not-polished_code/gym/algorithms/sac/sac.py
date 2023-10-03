import os
import math
import numpy as np
from collections import deque
import collections
from copy import deepcopy
import itertools 
import time
import statistics
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch import Tensor
from gym.spaces import Space
import math
from torch.utils.tensorboard import SummaryWriter

from algorithms.sac import ReplayBuffer, ReplayBuffer_pc

from algorithms.sac import MLPActorCritic

from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert, quaternion_multiply, quaternion_apply, euler_angles_to_matrix
from isaacgym.gymtorch import wrap_tensor
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
         quat_mul, tensor_clamp
from envs.utils.get_running_bbox import get_bbox_isaac_tensor, _draw_bbox_tensor, get_bbox_pt, _draw_bbox_pt
from envs.utils.get_reward import quat_axis
from ..ppo_utils.io_util import load, save
from ..ppo_utils.misc_util import lr_decay
from ..ppo_utils.log_util import log, log_test
from data_structure.observation import Observations

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class SAC:

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
                 model_cfg = None,

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
        self.train_env_num = vec_env.task.task_meta["train_env_num"] 
        self.valIntra_env_num = vec_env.task.task_meta["valIntra_env_num"]
        self.valInter_env_num = vec_env.task.task_meta["valInter_env_num"]
        self.max_episode_length = vec_env.task.task_meta["max_episode_length"]
        self.use_pc = model_cfg["use_pc"]
        self.device = device
        self.asymmetric = asymmetric
        self.learning_rate = learning_rate
        self.eval_round = eval_round
        self.eval_freq = eval_freq
        self.use_first_bbox = vec_env.task.cfg["task"]["use_first_bbox"]
        # SAC components
        self.vec_env = vec_env
        # ac_kwargs["model_cfg"] = dict()
        ac_kwargs["model_cfg"]["task_meta"] = vec_env.task.task_meta
        # contrastive learning
        self.actor_critic = actor_critic(vec_env.observation_space, vec_env.action_space,  **ac_kwargs).to(self.device)
        print(self.actor_critic)
        self.actor_critic_targ = deepcopy(self.actor_critic)
        self.replay_size = replay_size
        self.train_env_num = vec_env.task.task_meta["train_env_num"] 
        # import pdb
        # pdb.set_trace()

        self.storage = ReplayBuffer(self.train_env_num , replay_size, batch_size, num_transitions_per_env, self.observation_space.shape,
                                     self.state_space.shape, self.action_space.shape, self.device, sampler)
        # else:
        #     self.storage = ReplayBuffer_pc(vec_env.num_envs, replay_size, batch_size, num_transitions_per_env, self.observation_space.shape,
        #                              self.state_space.shape, self.action_space.shape, self.device, sampler)
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters())
        
        self.pi_optimizer = Adam(filter(lambda x: x.requires_grad, self.actor_critic.pi.parameters()), lr=self.learning_rate)
        self.q_optimizer = Adam(filter(lambda x: x.requires_grad, self.q_params), lr=self.learning_rate)


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

        self.canonical = canonical
        self.canon_space = canon_space
        self.pregrasp_steps = vec_env.task.cfg["task"]["pregrasp_steps"]
        self.use_part_pregrasp= vec_env.task.cfg["task"]["use_part_pregrasp"]
        

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0
        self.order_part_pos = 22

        self.apply_reset = apply_reset
        self.wandb_writer = wandb_writer
        self.checkpoint_path = checkpoint_path
        assert(self.replay_size >= self.batch_size / self.num_mini_batches)

        if self.use_first_bbox == "gt": 
            self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
        elif self.use_first_bbox == "pt":
            self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
        else:
            raise NotImplementedError

    def test(self,path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self,path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self,path, it):
        #torch.save(self.actor_critic.state_dict(),path)
        torch.save({
                    'iteration': it + 1,
                    'model_state_dict': self.actor_critic.state_dict(),
                    'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
                    'q_optimizer_state_dict': self.q_optimizer.state_dict(),
                    'total_steps' : self.tot_timesteps,
                    }, path)
    
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

        if self.use_part_pregrasp:
            # 先向后移动, 防止碰撞
            tar0 = torch.tensor([0.5, 0, 0], device=self.device) + self.vec_env.task.hand_rigid_body_tensor[:, :3]
            for i in range(200):
                actions[:, :3] = tar0 - self.vec_env.task.hand_rigid_body_tensor[:, :3]
                self.vec_env.control_step(actions)

            if self.vec_env.task.target_part == "door":
                pos_pre = (self.part_bbox_tensor[:,1,:] + self.part_bbox_tensor[:,3,:])/2 + torch.tensor([0.9, 0, 0.2], device = self.vec_env.task.device)
                pos_tar = (self.part_bbox_tensor[:,1,:] + self.part_bbox_tensor[:,3,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
            elif self.vec_env.task.target_part =="drawer":
                pos_pre = (self.part_bbox_tensor[:,2,:] + self.part_bbox_tensor[:,5,:])/2 + torch.tensor([1, 0, 0.2], device = self.vec_env.task.device)
                pos_tar = (self.part_bbox_tensor[:,2,:] + self.part_bbox_tensor[:,5,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
            else:
                raise NotImplementedError
            actions[:, -2:] = torch.tensor([1, 1], device=self.device)
            handle_out = torch.tensor([1.0, 0, 0], device=self.device)
            handle_out = handle_out.repeat(self.vec_env.num_envs, 1)
            handle_long = torch.tensor([0, 0, 1.0], device=self.device)
            handle_long = handle_long.repeat(self.vec_env.num_envs, 1)
            handle_short = torch.tensor([0, 1.0, 0], device=self.device)
            handle_short = handle_short.repeat(self.vec_env.num_envs, 1)
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

                actions[:,:3] = (step/(total_step-1))*(pos_tar - pos_pre)+pos_pre - self.vec_env.task.hand_rigid_body_tensor[:, :3]
                #actions[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
                actions[:, :3] *= math.exp(-0.002 * i)
            
                actions[:, :3] = torch.clamp(actions[:, :3], max=1, min=-1)
                self.vec_env.control_step(actions)
        # self.vec_env.task.cabinet_dof_tensor[:,0] *= 0
        print(self.vec_env.task.cabinet_dof_tensor[:,0].max())
        print(torch.norm(pos_tar - self.vec_env.task.hand_rigid_body_tensor[:, :3], dim = 1).max())
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
        # if current_obs.points is not None:
        #     current_obs.points[:,:, :3] = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), current_obs.points[:,:, :3]-self.center.view(-1,1,3))
        return current_obs 

    def uncanonicalize(self, actions):

        #actions[:, :3] = (actions[:, :3].view(-1, 1, 3) @ self.rot_matrix.float().permute(0, 2, 1)).view(-1, 3)
        if self.vec_env.task.cfg["env"]["driveMode"] in ["ik_rela", "ik_abs"]:
            actions[:, :3] = quaternion_apply(self.quaternion_rot_invert, actions[:, :3])
            actions[:, 3:7] = quaternion_multiply(self.quaternion_rot_invert, actions[:, 3:7])
        return actions

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
                        # Compute the action
                        if self.canonical:
                            current_obs = self.canonicalize(current_obs)
                        actions , feature = self.actor_critic.act(current_obs,deterministic =True)
                        if self.canonical:
                            actions = self.uncanonicalize(actions)
                        #actions, fearure = self.actor_critic.act(current_obs,deterministic =True)
                        # Step the vec_environment
                        next_obs, rews, dones, infos = self.vec_env.step(actions)
                        current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)
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
        # if self.vec_env.task.cfg["env"]["usePreGraspPose"]:
        #     self.pre_grasp()
        current_obs = self.vec_env.reset()
        
        current_states = self.vec_env.get_state()
        if self.is_testing:
            while True:
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                    # Compute the action
                    actions, fearure = self.actor_critic.act(current_obs,deterministic =True)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    current_obs.copy_(next_obs)
        else:
            rewbuffer = deque(maxlen=self.max_episode_length)
            lenbuffer = deque(maxlen=self.max_episode_length)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

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
                # Rollout
                collection_time = 0
                learn_time = 0
                for i in range(self.max_episode_length):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    if self.canonical: 
                        current_obs = self.canonicalize(current_obs)

                    # Compute the action
                    actions, fearure = self.actor_critic.act(current_obs)
                    
                    if self.canonical:
                        actions = self.uncanonicalize(actions)
                    
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    rews *= self.reward_scale
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    # import pdb
                    # pdb.set_trace()
                    self.storage.add_transitions(
                        observations = current_obs.obs[:self.train_env_num], 
                        states = current_obs.state[:self.train_env_num],
                        actions = actions[:self.train_env_num],
                        rewards = rews[:self.train_env_num], 
                        dones = dones[:self.train_env_num],
                        next_obs = next_obs.obs[:self.train_env_num],
                    )
                    current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)
                    current_states.copy_(next_states)
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

                    if self.storage.step >= self.batch_size:
                        self.warm_up = False

                    if self.warm_up == False and (i + 1) % self.num_transitions_per_env == 0:
                        # torch.cuda.synchronize()
                        stop = time.time()
                        collection_time += stop - start

                        # mean_trajectory_length, mean_reward = self.storage.get_statistics()
                        mean_reward = self.storage.rewards.mean()
                        # Learning step
                        start = stop
                        mean_value_loss, mean_surrogate_loss = self.update()
                        # for name, p in self.actor_critic.pi.parameters():
                        #     print(name , p)
                        # print(self.actor_critic.pi.parameters())
                        # print(self.actor_critic.q1.parameters())
                        # print(self.actor_critic.q2.parameters())
                        if self.print_log:
                            stop = time.time()
                            learn_time += stop - start
                            # self.log(locals())

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)


                stop = time.time()
                collection_time += stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                # TODO: need check the buffer size before update
                # add the update within the interaction loop
                # print(self.warm_up)
                if self.warm_up == False:  ###################?
                    # mean_value_loss, mean_surrogate_loss = self.update()

                    stop = time.time()
                    learn_time += stop - start
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
        self.tot_timesteps += self.max_episode_length * self.vec_env.num_envs
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
                    #'Policy/mean_noise_std': mean_std.item(),
                    # 'Policy/lr': self.step_size,
                    'Train/mean_reward/step': locs['mean_reward'],
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
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""" 
                          f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                          f"""{'Mean_part_position_valIntra:':>{pad}} {mean_part_position_valIntra:.5f}\n"""
                          f"""{'Mean_part_position_valInter:':>{pad}} {mean_part_position_valInter:.5f}\n"""
                          )
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
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
        if (locs["i"] + 1) % self.max_episode_length == 0:
            print(log_string)

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


                state_batch = self.storage.states[indices].view(-1, *self.storage.states.size()[2:])
                #pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
                observations_batch = self.storage.observations[indices].view(-1, *self.storage.observations.size()[2:])
                next_observations_batch = self.storage.next_observations[indices].view(-1, *self.storage.observations.size()[2:])
                #next_pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
                
                if not self.use_pc:
                    obs_batch = Observations(points = None, state = state_batch, obs = observations_batch)
                    next_obs_batch = Observations(points = None, state = state_batch, obs = next_observations_batch)
                else:
                    next_pc_batch = self.storage.pointcloud[indices].view(-1, *self.storage.pointcloud.size()[2:])
                    obs_batch = Observations(points = next_pc_batch, state = state_batch, obs = observations_batch)
                    next_obs_batch = Observations(points = next_pc_batch, state = state_batch, obs = next_observations_batch)
                
                # obs_batch = self.storage.observations[indices]
                # nextobs_batch = self.storage.next_observations[indices]
                # if self.asymmetric:
                #     states_batch = self.storage.states[indices]
                # else:
                #     states_batch = None

                actions_batch = self.storage.actions[indices].view(-1, *self.storage.actions.size()[2:])
                rewards_batch = self.storage.rewards[indices].view(-1)
                dones_batch = self.storage.dones[indices].view(-1)
                
                # import pdb
                # pdb.set_trace()
                # print("obs", observations_batch)
                # print("obs_shape", observations_batch.shape)
                # print("next_obs", next_observations_batch)
                # print("next_obs_shape", next_observations_batch.shape)
                # print ("actions", actions_batch)
                # print ("rew", rewards_batch)


                data = {'obs': obs_batch,
                        'act':actions_batch,
                        'r':rewards_batch,
                        'obs2':next_obs_batch,
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

        return mean_value_loss, mean_surrogate_loss

    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'],data['act'], data['r'], data['obs2'], data['done']

        q1 = self.actor_critic.q1(o, a)
        q2 = self.actor_critic.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, _ = self.actor_critic.pi(o2)

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

        contrastive_loss = 0
        # if self.contrastive :
        #     # do contrastive learning
        #     o, pc_out, pc_copy_out = self.actor_critic.prepare_o(o, contrastive=True)
        #     pc_out = pc_out.view(-1, pc_out.shape[-1])
        #     pc_copy_out = pc_copy_out.view(-1, pc_copy_out.shape[-1])
        #     for theta_q, theta_k in zip(self.actor_critic.pointnet_layer.parameters(), self.actor_critic.pointnet_layer_copy.parameters()) :
        #         theta_k = theta_k * self.contrastive_m + theta_q.detach() * (1-self.contrastive_m)
        #     # print(theta_q, theta_k)
        #     label = torch.arange(0, pc_out.shape[0], device=self.device).long()
        #     sim_mat = pc_out @ self.actor_critic.W @ pc_copy_out.T
        #     contrastive_loss = self.actor_critic.contrastive_loss_func(sim_mat, label)
        # else :
        #     o = self.actor_critic.prepare_o(o)

        pi, logp_pi, _ = self.actor_critic.pi(o)


        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.entropy_coef * logp_pi - q_pi).mean()

        return loss_pi + contrastive_loss



