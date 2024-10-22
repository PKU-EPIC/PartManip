import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .network import MLP, Conv3DNet, PoolConv3DNet, PointNet, ResNet, depthResNet

class ActorCritic(nn.Module):
    
    def __init__(self, obs_shape, actions_shape, model_cfg, proprio_shape=0):
        super(ActorCritic, self).__init__()

        net_cfg = model_cfg['network']

        # Policy
        self.actor = eval(net_cfg['name'])(obs_shape, actions_shape, net_cfg, proprio_shape=proprio_shape)

        # Value function
        self.critic = eval(net_cfg['name'])(obs_shape, 1, net_cfg, proprio_shape=proprio_shape)

        # Action noise
        self.log_std = nn.Parameter(np.log(model_cfg['action_std']) * torch.ones(actions_shape))
        
        self.max_action = model_cfg['clipAction']
        assert self.max_action > 0
        self.action_activate = model_cfg['action_activate']

   
    def forward(self):
        raise NotImplementedError

    def cri(self, observations):
        value = self.critic(observations) 
        return value.detach()

    def random_act_cri(self, observations):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(observations) 

        return self.action_activation(actions.detach()), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def random_act(self, observations):
        actions_mean = self.actor(observations)     # (num_envs,tsdf_res^3)->(num_envs,7)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions = distribution.sample()

        return self.action_activation(actions.detach())

    def act(self, observations):
        actions_mean = self.actor(observations)
        return self.action_activation(actions_mean.detach())

    def act_cri(self, observations):
        actions_mean = self.actor(observations)
        value = self.critic(observations) 
        return self.action_activation(actions_mean.detach()), value.detach()

    def update_act(self, observations):
        actions_mean = self.actor(observations)
        return self.action_activation(actions_mean)

    def update_act_cri(self, observations, actions):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(self.action_deactivation(actions))
        entropy = distribution.entropy()

        value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

    def action_activation(self, action):
        if self.action_activate == 'tanh':
            action = torch.tanh(action) * self.max_action
        elif self.action_activate is None:
            action = action
        else:
            raise NotImplementedError
        return action 

    def action_deactivation(self, action):
        if self.action_activate == 'tanh':
            action = torch.atanh(torch.clamp(action / self.max_action, max=1-1e-5, min=-1+1e-5))
        elif self.action_activate is None:
            action = action
        else:
            raise NotImplementedError
        return action 
    