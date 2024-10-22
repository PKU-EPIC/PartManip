from algorithms.algo_utils import RolloutStorage, ActorCritic, Normalization
from utils import path2video

import os
import time
from os.path import join as pjoin
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy 

class dagger:
    def __init__(self, vec_env, cfg, logger):
        # env infos
        self.vec_env = vec_env
        self.num_envs = cfg['num_envs']
        self.stu_obs_mode = cfg['obs_mode']
        self.stu_num_obs = vec_env.num_obs[self.stu_obs_mode]
        self.stu_input_obs = self.stu_num_obs
        self.num_actions = vec_env.num_actions
        self.max_episode_length = vec_env.max_episode_length

        # training
        self.model_cfg = cfg['model']
        self.max_iter = cfg['max_iterations']
        self.n_steps = cfg['n_steps']
        self.n_updates = cfg['n_updates']
        self.num_mini_batches = cfg['n_minibatches']
        self.device = cfg['device']
        self.buf_size = cfg['buf_size']
        self.tea_rew = np.load('teacher_reward.npy')
        self.tea_rew = torch.tensor(self.tea_rew).to(self.device)
        self.reward_reset = cfg['reward_reset']
        self.add_proprio_obs = cfg['add_proprio_obs']
        self.offline_data_pth = cfg['offline_data_pth']

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
        self.student = ActorCritic(self.stu_input_obs, self.num_actions, self.model_cfg, cfg['add_proprio_obs']*vec_env.num_obs['proprio_state']).to(self.device) 
        print("self.student: ", self.student)
        
        self.optimizer = optim.Adam(self.student.parameters(), lr=self.lr)

        # Log
        self.logger = logger
        self.total_envsteps = 0
        self.total_time = 0
        self.curr_iter = 0

        # load teacher 
        self.teacher_path = cfg['teacher']
        assert self.teacher_path is not None and os.path.exists(self.teacher_path)
        print(f'load teacher ckpt from {self.teacher_path}!')
        tea_dict = torch.load(self.teacher_path, map_location=self.device)
        self.tea_obs_mode = tea_dict['obs_mode']
        self.tea_num_obs = vec_env.num_obs[self.tea_obs_mode]
        self.teacher = ActorCritic(self.tea_num_obs, self.num_actions, tea_dict['model_cfg']).to(self.device) 
        self.teacher.load_state_dict(tea_dict["model_state_dict"])
        assert tea_dict['tricks']['use_state_norm'] == False 

        # Resume from previous ckpt
        self.resume(cfg['resume'])
        self.load_pretrain(cfg['pretrain'])

        self.storage = RolloutStorage(self.num_envs, self.buf_size, self.stu_num_obs, self.num_actions, self.device, sampler=cfg['sampler'], tea_obs_shape=self.tea_num_obs, max_length= self.max_episode_length)

    def save(self, it):
        # only save student model     
        os.makedirs(self.save_ckpt_dir, exist_ok=True)
        save_path = pjoin(self.save_ckpt_dir, f'model_{it}.pth')
        save_dict = {
            'iteration': it,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps' : self.total_envsteps,
            'obs_mode': self.stu_obs_mode,
            'teacher': self.teacher_path
        }
        torch.save(save_dict, save_path)
        print(f'save ckpt to {save_path}!')
        return 


    def load_pretrain(self, ckpt_path):
        if ckpt_path is not None:
            print(f'load pretrained ckpt from {ckpt_path}!')
            assert os.path.exists(ckpt_path)
            ckpt_dict= torch.load(ckpt_path, map_location=self.device)
            ckpt_dict['model_state_dict'].pop('log_std')
            self.student.load_state_dict(ckpt_dict["model_state_dict"], strict=False)
        return 

    def resume(self, ckpt_path):
        if ckpt_path is not None:
            print(f'load student ckpt from {ckpt_path}!')
            assert os.path.exists(ckpt_path)
            ckpt_dict= torch.load(ckpt_path, map_location=self.device)
            self.student.load_state_dict(ckpt_dict["model_state_dict"])
            self.optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
            self.curr_iter = ckpt_dict["iteration"]
            self.total_envsteps = ckpt_dict["total_steps"]

            # check infos
            # assert ckpt_dict['teacher'] == self.teacher_path
            # assert ckpt_dict['obs_mode'] == self.stu_obs_mode
        return 

    def eval(self):
        self.student.eval()
        if self.test_only:
            self.log_dict = {}
        with tqdm(total=self.eval_round) as pbar:
            pbar.set_description('Validating:')
            with torch.no_grad() :
                for r in range(self.eval_round):
                    ep_infos = []
                    save_dict_lst = []

                    all_curr_obs = self.vec_env.reset()
                    stu_curr_obs = all_curr_obs[self.stu_obs_mode]
                    tea_curr_obs = all_curr_obs[self.tea_obs_mode]
                    for i in range(self.max_episode_length) :

                        actions = self.student.act(stu_curr_obs)
                        # gt_act = self.teacher.act(tea_curr_obs)
                        # print(f'TA: {gt_act[0].data}, SA: {actions[0].data}')
                        # print(f'step: {i}, state: {tea_curr_obs[0,:3].data} {tea_curr_obs[0,7:10].data}, loss: {(gt_act[0]-actions[0]).pow(2).mean()}')
                        if self.save_video:
                            save_image_path = pjoin(self.logger.save_video_dir,  f"Iter{self.curr_iter}", f"{i}.png")
                        else:
                            save_image_path = None 

                        next_obs, rews, dones, infos = self.vec_env.step(actions, save_image_path=save_image_path)

                        infos['action_t'] = actions[:, :3].mean(dim=-1)
                        infos['action_r'] = actions[:, 3:6].mean(dim=-1)
                        infos['action_gripper'] = actions[:, -1]
                        infos['reward'] = rews
                        ep_infos.append(deepcopy(infos))

                        if self.save_pose:
                            save_dict = self.vec_env.save_scene_pose(pjoin(self.logger.save_pose_dir, f"Iter{self.curr_iter}", f"{i}.npy"))
                            save_dict['state'] = stu_curr_obs.cpu().numpy()
                            save_dict['action'] = actions.cpu().numpy()
                            save_dict_lst.append(deepcopy(save_dict))
                        stu_curr_obs = next_obs[self.stu_obs_mode]
                        tea_curr_obs = next_obs[self.tea_obs_mode]

                    if self.save_pose:
                        print(ep_infos[-1]['obj_up_flag'].float().mean())
                        for i in range(self.max_episode_length) :
                            save_dict_lst[i]['success'] = ep_infos[-1]['obj_up_flag'].cpu().numpy()
                            np.save(pjoin(self.logger.save_pose_dir, f"Iter{self.curr_iter}", f"{i}.npy"), save_dict_lst[i])

                    if self.save_video and r == self.eval_round-1:
                        save_image_path = pjoin(self.logger.save_video_dir,  f"Iter{self.curr_iter}")
                        path2video(save_image_path)

                    # log
                    mode = 'Test' if self.test_only else 'Val'
                    self.use_info_update_logdict(ep_infos, mode)   
                    ep_infos.clear()
                    pbar.update(1)
        return

    def run(self):
        if self.test_only:
            self.eval()
            self.logger.info(self.log_dict, self.curr_iter)
        else:

            if self.offline_data_pth is not None:
                self.storage.add_transitions_offline(self.offline_data_pth, self.device, self.add_proprio_obs)

            # reset
            all_curr_obs = self.vec_env.reset()  
            tea_curr_obs = all_curr_obs[self.tea_obs_mode]  
            stu_curr_obs = all_curr_obs[self.stu_obs_mode]  

            debug_lst = None 

            while self.curr_iter < self.max_iter:
                self.curr_iter += 1

                # intialization
                self.student.train()
                self.teacher.eval()

                self.log_dict = {}
                ep_infos = []
                start = time.time()
                # avg_rew = 0

                # Rollout
                for i in range(self.n_steps):
                    actions = self.student.random_act(stu_curr_obs)
                    gt_act = self.teacher.act(tea_curr_obs)
                    # if debug_lst is not None:
                    #     print(debug_lst)
                    #     print(f'TA: {gt_act[debug_lst,:3].data}, SA: {actions[debug_lst,:3].data} loss: {(gt_act[debug_lst]-actions[debug_lst]).pow(2).mean()} rew: {rews[debug_lst]} thre: {self.tea_rew[self.vec_env.progress_buf[debug_lst]]}')
                    
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    print(f'TrainIter: {self.curr_iter}, MaxStepid: {self.vec_env.progress_buf.max()}, Finish: {dones.int().sum()}, Rew: {self.vec_env.rew_buf.mean()}')
                    print(tea_curr_obs[0,:14], gt_act[0,:3], actions[0,:3], rews[0])
                    print('sample loss', (gt_act-actions).pow(2).mean())
                    # avg_rew += rews.cpu().numpy().mean() / self.n_steps
                    
                    self.storage.add_transitions_dagger(stu_curr_obs, tea_curr_obs)

                    infos['action_t'] = actions[:, :3].mean(dim=-1)
                    infos['action_r'] = actions[:, 3:6].mean(dim=-1)
                    infos['action_gripper'] = actions[:, -1]

                    tea_curr_obs = next_obs[self.tea_obs_mode]
                    stu_curr_obs = next_obs[self.stu_obs_mode]

                    ep_infos.append(deepcopy(infos))

                    # reward based reset
                    if self.reward_reset:
                        delta_step = 10
                        self.vec_env.dagger_reward_reset = (self.vec_env.progress_buf > delta_step) & (rews < self.tea_rew[self.vec_env.progress_buf-delta_step])
                        rew_reset_env_ids = torch.nonzero(self.vec_env.dagger_reward_reset).squeeze(-1)
                        if len(rew_reset_env_ids) != 0:
                            print(rews[rew_reset_env_ids], infos['reaching_reward'][rew_reset_env_ids], self.tea_rew[self.vec_env.progress_buf[rew_reset_env_ids]-delta_step])
                
                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = time.time()
                self.update(self.curr_iter)
                stop = time.time()
                learn_time = stop - start

                # log
                self.total_envsteps += self.n_steps * self.vec_env.num_envs
                self.total_time += collection_time + learn_time
                mean_std = self.student.log_std.exp().mean()
                fps = int(self.n_steps * self.vec_env.num_envs / (collection_time + learn_time))
                self.log_dict['Progress/total_steps'] = self.curr_iter
                self.log_dict['Progress/collection_time'] = collection_time
                self.log_dict['Progress/learn_time'] = learn_time
                self.log_dict['Progress/FPS'] = fps
                self.log_dict['Train/mean_action_noise_std'] = mean_std.item()
                self.log_dict['Train/cur_buf_size'] = self.storage.cur_buf_size
                self.log_dict['Train/succ_buf_ind'] = self.storage.succ_buf_ind
                self.log_dict['Train/mix_buf_ind'] = self.storage.mix_buf_ind
                self.use_info_update_logdict(ep_infos, 'Train')            

                # evaluation and save. NOTE eval will update self.log_dict
                if self.curr_iter % self.eval_freq == 0:
                    self.eval()
                    all_curr_obs = self.vec_env.reset()
                    tea_curr_obs = all_curr_obs[self.tea_obs_mode]
                    stu_curr_obs = all_curr_obs[self.stu_obs_mode]

                if self.curr_iter % self.save_freq == 0:
                    self.save(self.curr_iter)
                    
                self.logger.info(self.log_dict, self.curr_iter)
                
                ep_infos.clear()
        return 

    def use_info_update_logdict(self, info_lst, mode):
        for key in info_lst[0].keys():
            assert len(info_lst[0][key].shape) == 1, f"{key}: {info_lst[0][key].shape}"
            for i, info in enumerate(info_lst):
                if i == 0:
                    all_info_train = info[key].unsqueeze(-1).float()
                else:
                    all_info_train = torch.cat((all_info_train, info[key].unsqueeze(-1).float()), dim=-1)
            if mode != 'Train':
                if str(f'{mode}/{key}_mean') not in self.log_dict.keys():
                    self.log_dict[f'{mode}/{key}_mean'] = 0
                    self.log_dict[f'{mode}/{key}_max'] = 0
                self.log_dict[f'{mode}/{key}_mean'] += torch.mean(all_info_train) / self.eval_round
                self.log_dict[f'{mode}/{key}_max'] += torch.mean(all_info_train.max(dim=-1)[0]) / self.eval_round
            else:
                self.log_dict[f'{mode}/{key}_mean'] = torch.mean(all_info_train)
                self.log_dict[f'{mode}/{key}_max'] = torch.mean(all_info_train.max(dim=-1)[0])
        return 

    def update(self, it):
        mean_loss = 0
        count = 0
        if self.storage.cur_buf_size < 16:
            return 
        for epoch in range(self.n_updates):
            batch = self.storage.mini_batch_generator(self.num_mini_batches)
            for indices in batch:
                stu_obs_batch = self.storage.observations.view(-1, self.storage.observations.size(-1))[indices]     # (n_steps*n_env/n_minibatch, obs_dim)
                tea_obs_batch = self.storage.tea_obs.view(-1, self.storage.tea_obs.size(-1))[indices]                   # (n_steps*n_env/n_minibatch, obs_dim)

                tea_act = self.teacher.act(tea_obs_batch)
                # tea_act, tea_value = self.teacher.act_cri(tea_obs_batch)
                stu_act = self.student.update_act(stu_obs_batch)

                loss_dagger = (tea_act - stu_act).pow(2).mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss_dagger.backward()
                self.optimizer.step()

                mean_loss += loss_dagger.detach()
                count += 1

        mean_loss /= count 
        print('update loss', mean_loss)
        if self.lr_schedule == 'linear_decay': 
            lr_now = self.lr * max(1 - it / self.max_iter * 1.8, 0.1)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_now
        elif self.lr_schedule != 'fixed':
            raise NotImplementedError

        # log 
        self.log_dict['Train/learning_rate'] = self.optimizer.param_groups[0]['lr']
        self.log_dict['Train/dagger_loss'] = mean_loss

        return 
    
