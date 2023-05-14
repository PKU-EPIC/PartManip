import os
import torch
import numpy as np

def load(algo, path):
    assert os.path.exists(algo.checkpoint_path)
    checkpoint_dict= torch.load(algo.checkpoint_path, map_location=algo.device)
    try:
        algo.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"], strict = False)
        algo.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        algo.current_learning_iteration = checkpoint_dict["iteration"]
        algo.tot_timesteps = checkpoint_dict["total_steps"]
    except:
        algo.actor_critic.backbone.load_state_dict(checkpoint_dict["model_state_dict"], strict = False)
    # algo.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
   

def load_backbone_only(algo, path):
    import pdb
    pdb.set_trace()
    assert os.path.exists(algo.checkpoint_path)
    checkpoint_dict= torch.load(algo.checkpoint_path, map_location=algo.device)
    algo.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
    algo.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    algo.current_learning_iteration = checkpoint_dict["iteration"]
    algo.tot_timesteps = checkpoint_dict["total_steps"]

def save(self, path, it):    
    torch.save({
                'iteration': it + 1,
                'model_state_dict': self.actor_critic.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps' : self.tot_timesteps,
                }, path)
    