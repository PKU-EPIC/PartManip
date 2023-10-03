#!/usr/bin/env python3

"""Density models."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityMLP(nn.Module):
    """MLP density model."""

    def __init__(self, env_spec, ws=[64, 64], seed=None, obs_only=False):
        super(DensityMLP, self).__init__()
        assert len(ws) == 2, 'Only two-layer MLP is supported'
        # Set the seed (DAPG style)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        # Retrieve observation dim
        self.obs_dim = env_spec.observation_dim
        self.act_dim = env_spec.action_dim
        # Build the model
        if obs_only:
            self.fc0 = nn.Linear(self.obs_dim, ws[0])
        else:
            self.fc0 = nn.Linear(self.obs_dim+self.act_dim, ws[0])
        self.fc1 = nn.Linear(ws[0], ws[1])
        self.fc2 = nn.Linear(ws[1], 1)
        # Make params of the last layer small (following DAPG)
        self.fc2.weight.data *= 1e-2
        self.fc2.bias.data *= 1e-2

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
