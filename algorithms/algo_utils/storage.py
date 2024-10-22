import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler
import os 
from os.path import join as pjoin 
import numpy as np 

class RolloutStorage:

    def __init__(self, num_envs, n_steps, obs_shape, actions_shape, device, default_succ_value = 0, whole_adv_norm=False, sampler='sequential', tea_obs_shape=None, max_length=None):
        self.device = device
        self.sampler = sampler
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.step = 0
        self.whole_adv_norm = whole_adv_norm
        self.max_episode_length = max_length
        self.first_fill = True 
        self.default_succ_value = default_succ_value

        if tea_obs_shape is not None:   # For dagger, nsteps=1000.0-200 mix, 200-1000 suc.
            self.tea_obs = torch.zeros(self.n_steps*self.num_envs, tea_obs_shape, device=self.device)
            self.observations = torch.zeros(self.n_steps*self.num_envs, obs_shape, device=self.device)
            self.succ_flag = torch.zeros(self.n_steps*self.num_envs, 1, device=self.device)
            self.mix_buf_ind = 0
            self.succ_buf_ind = self.max_episode_length * self.num_envs
            self.cur_buf_size = 0
            self.last_episode_buf_ind = 0
        else:   # For PPO
            self.observations = torch.zeros(self.n_steps, num_envs, obs_shape, device=self.device)
            self.rewards = torch.zeros(self.n_steps, num_envs, 1, device=self.device)
            self.cur_buf_size = self.n_steps * self.num_envs
            self.actions = torch.zeros(self.n_steps, num_envs, actions_shape, device=self.device)
            self.dones = torch.zeros(self.n_steps, num_envs, 1, device=self.device).bool()
            self.succs = torch.zeros(self.n_steps, num_envs, 1, device=self.device).bool()
            self.actions_log_prob = torch.zeros(self.n_steps, num_envs, 1, device=self.device)
            self.values = torch.zeros(self.n_steps, num_envs, 1, device=self.device)
            self.returns = torch.zeros(self.n_steps, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(self.n_steps, num_envs, 1, device=self.device)
            self.mu = torch.zeros(self.n_steps, num_envs, actions_shape, device=self.device)
            self.sigma = torch.zeros(self.n_steps, num_envs, actions_shape, device=self.device)
            self.step_id = torch.zeros(self.n_steps, num_envs, 1, device=self.device)
        
    def add_transitions(self, observations, actions, rewards, dones, succs, values, actions_log_prob, mu, sigma):
        if self.step >= self.n_steps:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(observations)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.succs[self.step].copy_(succs.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)
        self.step = self.step + 1
        return 

    def add_transitions_offline(self, folder, device, add_proprio_obs=False):
        print('Read offline data from ', folder)
        scene_list = list(os.listdir(folder))
        scene_list.sort()
        step_list = list(os.listdir(pjoin(folder, scene_list[0])))
        step_list.sort()
        max_buf_size = self.n_steps * self.num_envs
        for scene in scene_list:
            for step in step_list:
                pth = pjoin(folder, scene, step)
                data = np.load(pth, allow_pickle=True).item()
                tsdf = torch.tensor(data['tsdf']).reshape(-1).to(device)
                if add_proprio_obs:
                    proprio_state = torch.tensor(data['proprio_state']).to(device)
                    stu_obs = torch.cat((tsdf, proprio_state),dim=-1)
                else:
                    stu_obs = tsdf
                tea_obs = torch.tensor(data['tea_obs']).to(device)
                self.observations[self.mix_buf_ind:self.mix_buf_ind+1].copy_(stu_obs)
                self.tea_obs[self.mix_buf_ind:self.mix_buf_ind+1].copy_(tea_obs)
                self.mix_buf_ind = (self.mix_buf_ind + 1) % max_buf_size
                self.last_episode_buf_ind = self.mix_buf_ind
                if self.cur_buf_size < max_buf_size:
                    self.cur_buf_size += 1
        return 

    def add_transitions_dagger(self, stu_obs, tea_obs):
        self.observations[self.mix_buf_ind:self.mix_buf_ind+self.num_envs].copy_(stu_obs)
        self.tea_obs[self.mix_buf_ind:self.mix_buf_ind+self.num_envs].copy_(tea_obs)
        max_buf_size = self.n_steps * self.num_envs
        self.mix_buf_ind = (self.mix_buf_ind + self.num_envs) % max_buf_size
        if self.cur_buf_size < max_buf_size:
            self.cur_buf_size += self.num_envs
        return 

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam): 
        advantage = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_is_not_terminal = (~self.dones[step])
            delta = self.rewards[step] + gamma * next_values - self.values[step]
            advantage = next_is_not_terminal * (delta + gamma * lam * advantage)
            if self.default_succ_value is not None:
                self.returns[step] = (~self.succs[step]) * (advantage + self.values[step]) + self.succs[step] * self.default_succ_value
            else:
                self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        if self.whole_adv_norm:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


    # def get_statistics(self):   # not use
    #     done = self.dones.cpu()
    #     done[-1] = 1 * (done.max() == 0) #如果已经有结束了的, 那么就不用再把最后一个作为1
    #     flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
    #     done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
    #     trajectory_lengths = (done_indices[1:] - done_indices[:-1])
    #     return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.cur_buf_size
        mini_batch_size = min(int(batch_size // num_mini_batches), 2048)
        # mini_batch_size = int(batch_size // num_mini_batches)

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch
