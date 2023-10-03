
import numpy as np


class RandomPolicy():
    """Random policy."""

    def __init__(self, env, seed=0):
        self.act_dim = env.action_dim
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high
        np.random.seed(seed)

    def get_action(self, ob):
        action = np.random.uniform(
            low=self.act_low, high=self.act_high, size=(self.act_dim,)
        )
        return [action, {'evaluation': action}]
