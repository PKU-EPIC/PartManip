import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
from mjrl.utils.logger import DataLog
import trimesh
import sys
from tqdm import tqdm

class BC:
    def __init__(self, expert_paths,
                 policy,
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 optimizer = None,
                 category=None,
                 object_scale=1.,
                 max_batch_size=512):

        self.policy = policy
        self.expert_paths = expert_paths
        self.epochs = epochs
        self.mb_size = batch_size
        self.logger = DataLog()
        self.category = category
        self.object_scale = object_scale
        self.max_batch_size = max_batch_size

        # get transformations
        observations = np.concatenate([path["observations"] for path in expert_paths])
        actions = np.concatenate([path["actions"] for path in expert_paths])
        in_shift, in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
        out_shift, out_scale = np.mean(actions, axis=0), np.std(actions, axis=0)

        # set scalings in the target policy
        self.policy.model.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.policy.old_model.set_transformations(in_shift, in_scale, out_shift, out_scale)

        # set the variance of gaussian policy based on out_scale
        params = self.policy.get_param_values()
        params[-self.policy.m:] = np.log(out_scale + 1e-12)
        self.policy.set_param_values(params)

        ### use cuda
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.policy.to_cuda()
        else:
            self.device = torch.device("cpu")
            self.policy.to_cpu()
        ###

        # construct optimizer
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr) if optimizer is None else optimizer

        # loss criterion is MSE for maximum likelihood estimation
        self.loss_function = torch.nn.MSELoss()

    def change_parameters(self, epochs=None, batch_size=None, lr=None):
        self.epochs = epochs if epochs is not None else self.epochs
        self.mb_size = batch_size if batch_size is not None else self.mb_size
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr) if lr is not None else self.optimizer

    def loss(self, obs, act):
        #obs_var = Variable(torch.from_numpy(obs).float(), requires_grad=False).cuda()
        #act_var = Variable(torch.from_numpy(act).float(), requires_grad=False).cuda()
        #obs_var = Variable(torch.from_numpy(obs).float(), requires_grad=False)
        #act_var = Variable(torch.from_numpy(act).float(), requires_grad=False)
        if obs.shape[0] > self.max_batch_size:
            idx = np.random.choice(obs.shape[0], self.max_batch_size)
            obs = obs[idx, :]
            act = act[idx, :]
    

        obs_var = torch.from_numpy(obs).float().to(self.device)
        act_var = torch.from_numpy(act).float().to(self.device)
        
        act_hat = self.policy.model(obs_var)
        return self.loss_function(act_hat, act_var.detach())

    def get_groundtruth_pointcloud(self, x, num_pc=1000):
        print(f'shape of obs: {x.shape}')
        original_obs = x[:, :-1]
        idx = x[:, -1]
       
        # load original pointclouds
        object_names = []
        pointclouds = {}
        for i in range(40):
            if i < 10:
                object_names.append('000' + str(i))
            else:
                object_names.append('00' + str(i))

        for object_name in object_names:
            obj = trimesh.load(f'{sys.path[-1]}/hand_imitation/hand_imitation/env/models/assets/shapenet_{self.category}/visual/{object_name}/model_transform_scaled.obj')
            pointclouds[int(object_name)] = np.asarray(obj.vertices) * self.object_scale

        pc_array = []
        for i in idx:
            pc = pointclouds[int(i)]
            random_idx = np.random.choice(pc.shape[0], num_pc)
            pc = pc[random_idx, :].ravel()
            pc_array.append(pc)
        pc_array = np.array(pc_array)
        print(f'shape of groundtruth pointclouds: {pc_array.shape}')
        print(f'shape of original obs: {original_obs.shape}')
        obs = np.concatenate((original_obs, pc_array), axis=1)

        return obs

    def train(self, random_embedding=False, transform_from_idx=False, expert_paths=None, num_samples=5000, finetune=False):
        if expert_paths is not None:
            self.expert_paths = expert_paths
        #self.policy.to_cuda()
        self.policy.set_pc_idx(False)
        self.policy.unfreeze_pointnet()
        observations = np.concatenate([path["observations"] for path in self.expert_paths])
        actions = np.concatenate([path["actions"] for path in self.expert_paths])
        if transform_from_idx:
            observations = self.get_groundtruth_pointcloud(observations)
            random_idx = np.random.choice(observations.shape[0], num_samples)
            observations = observations[random_idx, :]
            actions = actions[random_idx, :]

        print(f'shape of observations: {observations.shape}')

        params_before_opt = self.policy.get_param_values()
        ts = timer.time()
        num_samples = observations.shape[0]
        for ep in tqdm(range(self.epochs)):
            self.logger.log_kv('epoch', ep)
            loss_val = self.loss(observations, actions).cpu().data.numpy().ravel()[0]
            self.logger.log_kv('loss', loss_val)
            self.logger.log_kv('time', (timer.time()-ts))
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                obs = observations[rand_idx]
                act = actions[rand_idx]
                self.optimizer.zero_grad()
                loss = self.loss(obs, act)
                loss.backward()
                self.optimizer.step()
            #print(f'episode: {ep+1}, loss_val: {loss_val}')
        params_after_opt = self.policy.get_param_values()
        if finetune:
            self.policy.set_pointnet_values(params_after_opt, set_new=True, set_old=True)
        else:
            self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        self.logger.log_kv('epoch', self.epochs)
        loss_val = self.loss(observations, actions).cpu().data.numpy().ravel()[0]
        self.logger.log_kv('loss', loss_val)
        self.logger.log_kv('time', (timer.time()-ts))
        
        # estimate point cloud embedding and save it as an attribute of policy
        if self.category is None:
            return
        object_names = []
        embedding = {}
        for i in range(40):
            if i < 10:
                object_names.append('000' + str(i))
            else:
                object_names.append('00' + str(i))

        for object_name in object_names:
            obj = trimesh.load(f'{sys.path[-1]}/hand_imitation/hand_imitation/env/models/assets/shapenet_{self.category}/visual/{object_name}/model_transform_scaled.obj')
            pointcloud_original = np.asarray(obj.vertices) * self.object_scale
            #random_idx = np.random.choice(pointcloud_original.shape[0], self.policy.model.pc_dim//3)
            #sampled_pointcloud = pointcloud_original[random_idx, :].ravel()
            pc_input = torch.from_numpy(pointcloud_original.ravel()).float().to(self.device)
            pred_emb = self.policy.model.get_embedding(pc_input).detach()
            if random_embedding:
                embedding[int(object_name)] = torch.rand(pred_emb.shape).to(pred_emb.device)
            else:
                embedding[int(object_name)] = pred_emb



        #print(f'embedding: {embedding}')
        self.policy.set_embedding(embedding)    
        self.policy.set_pc_idx(use_pc_idx=True) 
        #self.policy.to_cpu()
        self.policy.freeze_pointnet()
