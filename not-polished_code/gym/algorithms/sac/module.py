import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

from ..ppo_utils.backbone import PointNetBackbone, SparseUnetBackbone



LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)




class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True, epsilon=1e-6):
        net_out = self.net(obs.obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = (pi_distribution.log_prob(pi_action) - torch.log(1 - torch.tanh(pi_action).pow(2) + epsilon)).sum(axis=-1, keepdim=True)
            # import pdb
            # pdb.set_trace()
            # logp_pi -= (2*(torch.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1, keepdim = True)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi, None



class SquashedGaussianMLPActorPC(nn.Module):

    def __init__(self, obs_dim, act_dim, feature_dim, hidden_sizes, activation, act_limit, backbone):
        super().__init__()
        self.net = mlp([obs_dim+feature_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.backbone = backbone

    def forward(self, obs, deterministic=False, with_logprob=True, epsilon=1e-6):
        points = obs.points
        feature = self.backbone(points)
        input_obs = obs.obs
        inp = torch.cat((input_obs, feature), dim=1)
        net_out = self.net(inp)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = (pi_distribution.log_prob(pi_action) - torch.log(1 - torch.tanh(pi_action).pow(2) + epsilon)).sum(axis=-1, keepdim=True)
            # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi, feature



class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs.obs, act], dim=-1))
        return q # Critical to ensure q has right shape.
        # return torch.squeeze(q, -1) # Critical to ensure q has right shape.




class MLPQFunctionPC(nn.Module):

    def __init__(self, obs_dim, act_dim, feature_dim, hidden_sizes, activation, backbone):
        super().__init__()
        self.q = mlp([obs_dim + act_dim + feature_dim] + list(hidden_sizes) + [1], activation)
        self.backbone = backbone

    def forward(self, obs, act):
        points = obs.points
        input_obs = obs.obs
        feature = self.backbone(points)
        inp = torch.cat((input_obs, feature, act), dim=1)
        q = self.q(torch.cat([obs.obs, act], dim=-1))
        return q # Critical to ensure q has right shape.
        # return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,hidden_sizes=(256,256),
                 activation=nn.ELU, model_cfg=None):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.use_pc = model_cfg["use_pc"]
        if self.use_pc:
            self.feature_dim = model_cfg['feature_dim']
            self.pc_dim = 3 + model_cfg["task_meta"]["mask_dim"] # N*pc_dim -> point cloud
            self.use_spconv = model_cfg["use_spconv"]
            self.use_pretrain = model_cfg["use_pretrain"]
        else:
            self.feature_dim = 0
            self.pc_dim = 0

        if self.use_pc:
            if self.use_spconv:
                if self.use_pretrain:
                    self.backbone = SparseUnetBackbone(
                        model_cfg["Spconv"]["in_channels"],model_cfg["Spconv"]["num_classes"],model_cfg["Spconv"]["channels"],pretrained_model_path = model_cfg["Spconv"]["ckpt_pth"])
                else:
                    self.backbone = SparseUnetBackbone(
                        model_cfg["Spconv"]["in_channels"],model_cfg["Spconv"]["num_classes"],)
                
            else:
                self.backbone = PointNetBackbone(
                    pc_dim = self.pc_dim,
                    feature_dim = self.feature_dim
                )

        if not self.use_pc:
            # build policy and value functions
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
            self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
            self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        else:
            # build policy and value functions
            self.pi = SquashedGaussianMLPActorPC(obs_dim, act_dim, self.feature_dim, hidden_sizes, activation, act_limit, self.backbone)
            self.q1 = MLPQFunctionPC(obs_dim, act_dim, self.feature_dim, hidden_sizes, activation, self.backbone)
            self.q2 = MLPQFunctionPC(obs_dim, act_dim, self.feature_dim, hidden_sizes, activation, self.backbone)


    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _, feature = self.pi(obs, deterministic, False)
            return a.detach(), feature


