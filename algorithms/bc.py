from utils import path2video
from algorithms.algo_utils import ActorCritic

import os
import time
from os.path import join as pjoin
import numpy as np 
import torch
import torch.optim as optim


class Tsdf_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(Tsdf_Dataset, self).__init__()
        self.data_path = data_path
        self.env_num = len(os.listdir(data_path))
        self.env_lst = os.listdir(data_path)
        self.step_num = len(os.listdir(pjoin(data_path, 'scene_00000')))
        
    def __getitem__(self, index):
        env_ind = self.env_lst[index // self.step_num]
        step_ind = index % self.step_num
        pth = pjoin(self.data_path, f'{env_ind}/step_{str(step_ind).zfill(5)}.npy')
        data = np.load(pth, allow_pickle=True).item()
        tsdf = data['tsdf']
        action = data['action']
        state = data['proprio_state']
        return tsdf, action, state

    def __len__(self):
        return self.env_num * self.step_num

class bc:
    def __init__(self, vec_env, cfg, logger):
        # env infos
        self.vec_env = vec_env
        self.num_envs = cfg['num_envs']
        self.stu_obs_mode = cfg['obs_mode']
        self.stu_num_obs = vec_env.num_obs[self.stu_obs_mode]
        self.num_actions = vec_env.num_actions
        self.max_episode_length = vec_env.max_episode_length
       
        # training
        self.model_cfg = cfg['model']
        self.max_iter = cfg['max_iterations']
        self.device = cfg['device']
        self.data_path = cfg['data_path']
        self.n_minibatches = cfg['n_minibatches']
        self.add_proprio_obs = cfg['add_proprio_obs']

        # eval and save
        self.eval_round = cfg['eval_round']
        self.eval_freq = cfg['eval_frequence']
        self.save_freq = cfg['save_frequence']
        self.test_only = cfg['test_only']
        self.save_pose = cfg['save_pose']
        self.save_video = cfg['save_video']
        self.save_ckpt_dir = logger.save_ckpt_dir

        # learning rate
        self.lr_schedule =  cfg['lr_schedule']
        self.lr = cfg['lr']

        # network
        self.student = ActorCritic(self.stu_num_obs, self.num_actions, self.model_cfg, cfg['add_proprio_obs']*vec_env.num_obs['proprio_state']).to(self.device) 
        print(self.student)
        
        self.optimizer = optim.Adam(self.student.parameters(), lr=self.lr)
        
        # Log
        self.logger = logger
        self.total_time = 0
        self.curr_iter = 0

        # Resume from previous ckpt
        self.resume(cfg['resume'])


    def save(self, it):
        # only save student model     
        os.makedirs(self.save_ckpt_dir, exist_ok=True)
        save_path = pjoin(self.save_ckpt_dir, f'model_{it}.pth')
        save_dict = {
            'iteration': it,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_mode': self.stu_obs_mode,
            'total_steps' : 0,      # useless
            'tricks': {'use_state_norm': False},  # useless
            'teacher': 0    # useless
        }
        torch.save(save_dict, save_path)
        print(f'save ckpt to {save_path}!')
        return 

    def resume(self, ckpt_path):
        if ckpt_path is not None:
            print(f'load student ckpt from {ckpt_path}!')
            assert os.path.exists(ckpt_path)
            ckpt_dict= torch.load(ckpt_path, map_location=self.device)
            self.student.load_state_dict(ckpt_dict["model_state_dict"])
            self.optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
            self.curr_iter = ckpt_dict["iteration"]

            # check infos
            assert ckpt_dict['obs_mode'] == self.stu_obs_mode
        return 

    def run(self):
        if self.test_only:
            raise NotImplementedError
            self.eval()
            self.logger.info(self.log_dict, self.curr_iter)
        else:
            train_dataset = Tsdf_Dataset(self.data_path)
            batch_size = len(train_dataset) // self.n_minibatches
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

            while self.curr_iter < self.max_iter:
                self.curr_iter += 1

                self.log_dict = {}
                mean_loss = 0
                count = 0
                for tsdfs, actions, states in train_loader:
                    tsdfs = tsdfs.to(self.device)
                    actions = actions.to(self.device)
                    states = states.to(self.device).squeeze()
                    
                    if self.add_proprio_obs:
                        model_input = torch.cat([tsdfs.reshape(actions.shape[0], -1), states], dim=-1)
                    else:
                        model_input = tsdfs

                    stu_act = self.student.update_act(model_input)

                    # # L2 seems to be better than L1
                    loss = (actions - stu_act).pow(2).mean()
                    
                    mean_loss += loss.detach()

                    # Gradient step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    count += 1

                mean_loss /= count 

                if self.lr_schedule == 'linear_decay': 
                    lr_now = self.lr * (1 - self.curr_iter / self.max_iter)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_now
                elif self.lr_schedule == 'step_decay':
                    if self.curr_iter < self.max_iter / 2:
                        lr_now = self.lr
                    else:
                        lr_now = self.lr * 0.1
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_now
                elif self.lr_schedule != 'fixed':
                    raise NotImplementedError

                # log 
                self.log_dict['Train/learning_rate'] = self.optimizer.param_groups[0]['lr']
                self.log_dict['Train/bc_loss'] = mean_loss
                self.log_dict['Progress/total_steps'] = self.curr_iter

                # evaluation and save. NOTE eval will update self.log_dict
                # if self.curr_iter % self.eval_freq == 0:
                #     self.eval()
                if self.curr_iter % self.save_freq == 0:
                    self.save(self.curr_iter)
                self.logger.info(self.log_dict, self.curr_iter)

        return 


   