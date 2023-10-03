import torch
import numpy as np

def lr_decay(self, total_steps):
    lr_now = self.step_size * (1 - total_steps / self.max_iterations)
    for p in self.optimizer.param_groups:
        p['lr'] = lr_now