#!/usr/bin/env python3

"""Inverse dynamics models: f(s_t, s_t1) = a_t"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tpi.core.config import cfg


class InvDynMLP(nn.Module):
    """MLP inverse dynamics model."""

    def __init__(self, env_spec, mlp_w=64, seed=None):
        super(InvDynMLP, self).__init__()
        # Set the seed (DAPG style)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        # Compute obs dim in a hacky way
        if cfg.INVDYN_ONPG_SUBSET:
            if cfg.ENV_NAME == 'hammer-v0':
                self.obs_dim = 36
            elif cfg.ENV_NAME == 'door-v0':
                self.obs_dim = 30
            elif cfg.ENV_NAME == 'relocate-v0':
                self.obs_dim = 30
            elif cfg.ENV_NAME == 'pen-v0':
                self.obs_dim = 24
            else:
                assert False
        else:
            self.obs_dim = env_spec.observation_dim
        # Compute act dim in a hacky way
        if cfg.INVDYN_ONPG_ACT_SUBSET:
            assert cfg.CUSTOM_FINGERS
            # arm
            self.act_dim = 6
            # wrist
            self.act_dim += 2
            # index finger
            if cfg.CUSTOM_FINGERS_MASK[1] == '1':
                self.act_dim += 4
            # middle finger
            if cfg.CUSTOM_FINGERS_MASK[2] == '1':
                self.act_dim += 4
            # ring finger
            if cfg.CUSTOM_FINGERS_MASK[3] == '1':
                self.act_dim += 4
            # little finger
            if cfg.CUSTOM_FINGERS_MASK[4] == '1':
                self.act_dim += 5
            # thumb
            if cfg.CUSTOM_FINGERS_MASK[0] == '1':
                self.act_dim += 5
        else:
            self.act_dim = env_spec.action_dim
        # Build the model
        self.fc0 = nn.Linear(self.obs_dim * 2, mlp_w)
        self.fc1 = nn.Linear(mlp_w, mlp_w)
        self.fc2 = nn.Linear(mlp_w, self.act_dim)
        # Make params of the last layer small (following DAPG)
        self.fc2.weight.data *= 1e-2
        self.fc2.bias.data *= 1e-2

    def forward(self, x):
        x = F.tanh(self.fc0(x))
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
