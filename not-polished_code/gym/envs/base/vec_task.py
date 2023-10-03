from gym import spaces
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch
import torch
import numpy as np
from utils.gym_info import clip_actions, clip_observations


class VecTask():
    def __init__(
        self, 
        task, 
        rl_device,
        clip_observations = clip_observations, 
        clip_actions = clip_actions
    ):
        self.task = task
        self.num_environments = task.num_envs
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions


        self.obs_space = spaces.Box(np.ones(self.num_observations) * -np.Inf,\
            np.ones(self.num_observations) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, \
            np.ones(self.num_states) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., \
            np.ones(self.num_actions) * 1.)
        
        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device

        print("RL device", self.rl_device)

    def step(self, actions):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        return self.obs_space
    
    @property
    def action_space(self):
        return self.act_space
    
    @property
    def num_envs(self):
        return self.num_environments
    
    @property
    def num_acts(self):
        return self.num_actions
    
    @property
    def num_obs(self):
        return self.num_observations

class VecTaskPythonArm(VecTask):
    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
    
    def step(self, actions):
        #actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        actions_tensor = actions
        obs, rew, done, _ = self.task.step(actions_tensor)

        return  obs, \
            rew.to(self.rl_device), done.to(self.rl_device), self.task.extras
            #torch.clamp(obs, -self.clip_obs, self.clip_obs).to(self.rl_device), \

    def control_step(self, actions):
        actions_tensor = actions
        #actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.task.control_step(actions_tensor)

    def reset(self):
        # actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], 
        #     dtype=torch.float32, device=self.rl_device))

        self.task.reset()
        # self.task.step(actions)

        return self.task.obs_buf # torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

class VecTaskPythonArmPC(VecTask):
    def __init__(self, task, rl_device, clip_observations=clip_observations, clip_actions=clip_actions):
        super().__init__(task, rl_device, clip_observations, clip_actions)
        self.pc_dim = self.task.cfg["obs"]["pc"]["numPCObservations"]
        self.pc_space = spaces.Box(np.ones((self.task.pc_downsample_num, self.pc_dim))\
            * -np.Inf,np.ones((self.task.pc_downsample_num, self.pc_dim )) * np.Inf)
    
    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):

         #actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        actions_tensor = actions

        obs, rew, done, _ = self.task.step(actions_tensor)

        return obs, \
            rew.to(self.rl_device), done.to(self.rl_device), self.task.extras #torch.clamp(obs, -self.clip_obs, self.clip_obs).to(self.rl_device), 

    def control_step(self, actions):
        
         #actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        actions_tensor = actions
        self.task.control_step(actions_tensor)


    def reset(self):
        #actions = 0.001 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], dtype=torch.float32, device=self.rl_device))

        # step the simulator
        self.task.reset()
        # self.task.step(actions)

        return self.task.obs_buf#torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)