import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler
import random

class RolloutStorage:

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, device='cpu', sampler='sequential', use_imit=False, max_length=0):

        self.device = device
        self.sampler = sampler

        # Core

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        print(self.observations.shape)

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.use_imit = use_imit
        self.max_length = max_length

        self.step = 0

        #for self imitation
        if use_imit:
            self.observations_imit = torch.zeros(max_length, num_envs, *obs_shape, device=self.device)
            self.states_imit = torch.zeros(max_length, num_envs, *states_shape, device=self.device)
            self.actions_imit = torch.zeros(max_length, num_envs, *actions_shape, device=self.device)
            self.rewards_imit = self.rewards = torch.zeros(max_length, num_envs, 1, device=self.device)
            self.imit_step = 0  #整个traj的step
            

    def add_transitions(self, observations, states, actions, rewards, dones, values, actions_log_prob, mu, sigma):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        if self.use_imit:
            self.observations_imit[self.imit_step].copy_(observations)
            self.states_imit[self.imit_step].copy_(states)
            self.actions_imit[self.imit_step].copy_(actions)
            self.rewards_imit[self.imit_step].copy_(rewards.view(-1, 1))
            self.imit_step += 1
            self.imit_step = self.imit_step % self.max_length

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam): #只用了self.value, reward, return, adv是计算出来的
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        #self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)
        #把这部分放到了ppo里面, 方便调整参数

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1 * (done.max() == 0) #如果已经有结束了的, 那么就不用再把最后一个作为1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        #print(done_indices)        
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env                                                                                                                                                         
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size <= 0:
            mini_batch_size = 1

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch

class RolloutStoragePC(RolloutStorage):

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, point_cloud_shape, device='cpu', sampler='sequential', use_imit=False, max_length=0, use_seg = False):

        super().__init__(num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, device, sampler, use_imit, max_length)
        self.use_seg = use_seg
        # self.device = device
        # self.sampler = sampler

        # # Core
        # self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        # self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        # self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.pointcloud = torch.zeros(num_transitions_per_env, num_envs, *point_cloud_shape, device=self.device)
        

        # For PPO
        # self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # self.num_transitions_per_env = num_transitions_per_env
        # self.num_envs = num_envs

        # self.step = 0
        if use_imit:
            self.pointcloud_imit = torch.zeros(self.max_length, num_envs, *point_cloud_shape, device=self.device)

    def add_transitions(self, observations, points, states, actions, rewards, dones, values, actions_log_prob, mu, sigma):

        # if self.step >= self.num_transitions_per_env:
        #     raise AssertionError("Rollout buffer overflow")
        # self.observations[self.step].copy_(observations)
        self.pointcloud[self.step].copy_(points)
        # self.states[self.step].copy_(states)
        # self.actions[self.step].copy_(actions)
        # self.rewards[self.step].copy_(rewards.view(-1, 1))
        # self.dones[self.step].copy_(dones.view(-1, 1))
        # self.values[self.step].copy_(values)
        # self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        # self.mu[self.step].copy_(mu)
        # self.sigma[self.step].copy_(sigma)

        # self.step += 1

        if self.use_imit:
            self.pointcloud_imit[self.imit_step].copy_(points)
        super().add_transitions(observations, states, actions, rewards, dones, values, actions_log_prob, mu, sigma)#必须要在后面调用, 否则self.step + 1

    def update_pointcloud(self, pointcloud):
        self.pointcloud = pointcloud.repeat(self.num_transitions_per_env, 1, 1)

class RolloutStoragePC_feature(RolloutStorage):

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, point_cloud_feature_shape, device='cpu', sampler='sequential', use_imit=False, max_length=0):
        
        super().__init__(num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, device, sampler, use_imit, max_length)
        
        # self.device = device
        # self.sampler = sampler

        # # Core
        # self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        # self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        # self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.pointcloud_features = torch.zeros(num_transitions_per_env, num_envs, point_cloud_feature_shape , device=self.device)
    
        if use_imit:
            self.pointcloud_features_imit = torch.zeros(max_length, num_envs, point_cloud_feature_shape , device=self.device)
        
        # For PPO
        # self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # self.num_transitions_per_env = num_transitions_per_env
        # self.num_envs = num_envs

        # self.step = 0

    def add_transitions(self, observations, points_features, states, actions, rewards, dones, values, actions_log_prob, mu, sigma):
        # if self.step >= self.num_transitions_per_env:
        #     raise AssertionError("Rollout buffer overflow")
        # self.observations[self.step].copy_(observations)
        self.pointcloud_features[self.step].copy_(points_features)
        if self.use_imit:
            self.pointcloud_features_imit[self.step].copy_(points_features)
        # self.states[self.step].copy_(states)
        # self.actions[self.step].copy_(actions)
        # self.rewards[self.step].copy_(rewards.view(-1, 1))
        # self.dones[self.step].copy_(dones.view(-1, 1))
        # self.values[self.step].copy_(values)
        # self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        # self.mu[self.step].copy_(mu)
        # self.sigma[self.step].copy_(sigma)
        super().add_transitions(observations, states, actions, rewards, dones, values, actions_log_prob, mu, sigma)
        # self.step += 1
    
    # def update_(self, pointcloud):
    #     self.pointcloud =pointcloud pointcloud.repeat(self.num_transitions_per_env, 1, 1)

class RolloutStorage_expert:

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, point_cloud_shape, states_shape, actions_shape, device='cpu', sampler='sequential', use_seg = False):

        self.device = device
        self.sampler = sampler
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env 
        self.use_seg = use_seg

        # Core
        # self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        # self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.expert_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.pointcloud = torch.zeros(num_transitions_per_env, num_envs, *point_cloud_shape, device=self.device)
        self.seg_labels = torch.zeros(num_transitions_per_env, num_envs, point_cloud_shape[0], device=self.device)
        if self.use_seg:
            self.sem_labels = torch.zeros(num_transitions_per_env, num_envs, point_cloud_shape[0], device = self.device)
        # self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        # print(self.observations.shape)
        
        # # For PPO
        # self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, points, observations, actions, expert_actions, seg_label = None):
        self.step = self.step%self.num_transitions_per_env
        # if self.step >= self.num_transitions_per_env:
        #     raise AssertionError("Rollout buffer overflow")

        # self.observations[self.step].copy_(observations)
        # self.states[self.step].copy_(states)
        self.actions[self.step] = actions
        self.expert_actions[self.step] = (expert_actions).clone()
        self.observations[self.step] = (observations).clone()
        self.pointcloud[self.step].copy_(points)
        
        # self.dones[self.step].copy_(dones.view(-1, 1))
        # self.values[self.step].copy_(values)
        # self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        # self.mu[self.step].copy_(mu)
        # self.sigma[self.step].copy_(sigma)
        if self.use_seg:
            self.seg_labels[self.step] = seg_label
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam): #只用了self.value, reward, return, adv是计算出来的
        raise NotImplementedError
        # advantage = 0
        # for step in reversed(range(self.num_transitions_per_env)):
        #     if step == self.num_transitions_per_env - 1:
        #         next_values = last_values
        #     else:
        #         next_values = self.values[step + 1]
        #     next_is_not_terminal = 1.0 - self.dones[step].float()
        #     delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
        #     advantage = delta + next_is_not_terminal * gamma * lam * advantage
        #     self.returns[step] = advantage + self.values[step]

        # # Compute and normalize the advantages
        # self.advantages = self.returns - self.values
        # #self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)
        # #把这部分放到了ppo里面, 方便调整参数

    def get_statistics(self):
        raise NotImplementedError
        # done = self.dones.cpu()
        # done[-1] = 1 * (done.max() == 0) #如果已经有结束了的, 那么就不用再把最后一个作为1
        # flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        # done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        # #print(done_indices)        
        # trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        # return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env                                                                                                                                                         
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size <= 0:
            mini_batch_size = 1


        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch
        
class ReplayBuffer:

    def __init__(self, num_envs, replay_size, batch_size, num_transitions_per_env, obs_shape, states_shape, actions_shape, device='cpu', sampler='sequential'):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = torch.zeros(replay_size, num_envs, *obs_shape, device=self.device)
        self.states = torch.zeros(replay_size, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(replay_size, num_envs, 1, device=self.device)
        self.next_observations = torch.zeros(replay_size, num_envs, *obs_shape, device=self.device)
        self.actions = torch.zeros(replay_size, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(replay_size, num_envs, 1, device=self.device).byte()

        self.num_transitions_per_env = num_transitions_per_env
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.fullfill = False

        self.step = 0

    def add_transitions(self, observations, states, actions, rewards, next_obs ,dones):


        self.observations[self.step].copy_(observations.state)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.next_observations[self.step].copy_(next_obs.state)
        self.dones[self.step].copy_(dones.view(-1, 1))

        self.step += 1

        if self.step >= self.replay_size:
            #TODO: 有点bug 清不掉0 后续改下
            self.step = (self.step) % self.replay_size
            self.fullfill = True
            # raise AssertionError("Rollout buffer overflow")


    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards[:self.step].mean()

    def mini_batch_generator(self, num_mini_batches):
        ##### buffer中sample很多个batch
        #TODO: 可以随机选择batch_size
        batch_size = self.batch_size
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size <= 0:
            mini_batch_size = 1
        batch = []
        # if self.sampler == "sequential":
        #     # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
        #     # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
        #     subset = SequentialSampler(range(batch_size))
        # elif self.sampler == "random":
        #     subset = SubsetRandomSampler(range(batch_size))
        for _ in range(num_mini_batches):
            if self.fullfill == True:
                subset = random.sample(range(self.replay_size * self.num_envs), mini_batch_size)
                #print("fullfill")
            else:
                subset = random.sample(range(self.step * self.num_envs), mini_batch_size)
        # batch = BatchSampler(subset, mini_batch_size, drop_last=True)
            batch.append(subset) 
        return batch

class ReplayBuffer_pc(ReplayBuffer):
    def __init__(self, num_envs, replay_size, batch_size, num_transitions_per_env, obs_shape, states_shape, actions_shape, pcs_shape, device='cpu', sampler='sequential'):
        super().__init__(num_envs, replay_size, batch_size, num_transitions_per_env, obs_shape, states_shape, actions_shape, device, sampler)

        self.pointcloud = torch.zeros(replay_size, num_envs, *pcs_shape, device=self.device)
        self.next_pointcloud = torch.zeros(replay_size, num_envs, *pcs_shape, device=self.device)


    def add_transitions(self, observations, states, actions, rewards, next_obs ,dones):
        
        if self.step >= self.replay_size:
            #TODO: 有点bug 清不掉0 后续改下
            self.step = (self.step + 1) % self.replay_size
            self.fullfill = True
            # raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations.state)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.next_observations[self.step].copy_(next_obs.state)
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.pointcloud[self.step].copy_(observations.points)
        self.next_pointcloud[self.step].copy_(next_obs.points)
        self.step += 1