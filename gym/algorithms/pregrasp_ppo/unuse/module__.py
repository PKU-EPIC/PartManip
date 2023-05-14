import numpy as np
from functools import reduce
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
# from algorithms.pn_utils.model_3d import PointNet2Feature
from algorithms.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet, getNaivePointNet
from algorithms.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNetWithInstanceInfo
# from perception.structures.point_cloud import PointCloud
# from perception.datasets.gapartnet_new import apply_voxelization, apply_voxelization_batch
import ipdb
import importlib
import spconv.pytorch as spconv
from ..ppo_utils.backbone import PointNetBackbone, SparseUnetBackbone, GAPartNet, SegSpconvBackbone
from ..ppo_utils.pointgroup_utils import estimate_pose_from_npcs

class MyNetState(nn.Module):
    def __init__(self, fc):
        super(MyNetState, self).__init__()
        self.fc = fc
    
    def forward(self, input):
        input_obs = input.obs
        y2 = self.fc(input_obs)
        others = {}
        return y2, others

class MyNetPC(nn.Module):
    def __init__(self, backbone, fc):
        super(MyNetPC, self).__init__()
        self.backbone = backbone
        self.fc = fc
    
    def forward(self, input):
        input_pc = input.points[:,:,:] 
        input_obs = input.obs
        y1 = self.backbone(input_pc)
        inp = torch.cat((input_obs, y1), dim=1)
        y2 = self.fc(inp)
        others = {"feature": y1.detach()}
        return y2, others

class MyNetPC_new(nn.Module):
    def __init__(self, backbone, fc):
        super(MyNetPC_new, self).__init__()
        self.backbone = backbone
        self.fc = fc
    
    def forward(self, input):
        input_pc = input.points[:,:,:6] 
        input_obs = input.obs
        y1, seg_preds = self.backbone(input_pc)
        inp = torch.cat((input_obs, y1.detach()), dim=1)
        y2 = self.fc(inp)
        others = {"feature": y1.detach(), "seg_preds": seg_preds}
        return y2, others


class ActorCriticPC(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, device = None):
        super(ActorCriticPC, self).__init__()

        # self.state_dim = model_cfg["task_meta"]["state_dim"] # feature_dim + state_dim = MLP input_dim
        #self.qpose_num = qpose_num
        self.use_pc = model_cfg["use_pc"]
        if self.use_pc:
            self.feature_dim = model_cfg['feature_dim']
            self.pc_dim = 3 + model_cfg["task_meta"]["mask_dim"] # N*pc_dim -> point cloud
            self.backbone_type = model_cfg["backbone_type"]
            self.use_pretrain = model_cfg["use_pretrain"]
        else:
            self.feature_dim = 0
            self.pc_dim = 0
        
        self.device = device

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        self.freeze_backbone = model_cfg["freeze"]

        # Policy
        # obs_shape need to be (point_num, 3+feat_dim)
        if self.use_pc:
            if self.backbone_type == "spconv":
                if self.use_pretrain:
                    self.backbone = SparseUnetBackbone(
                        model_cfg["Spconv"]["in_channels"],model_cfg["Spconv"]["num_classes"],model_cfg["Spconv"]["channels"],pretrained_model_path = model_cfg["Spconv"]["ckpt_pth"])
                else:
                    self.backbone = SparseUnetBackbone(
                        model_cfg["Spconv"]["in_channels"],model_cfg["Spconv"]["num_classes"],)
                
            elif self.backbone_type == "pn":
                self.backbone = PointNetBackbone(
                    pc_dim = self.pc_dim,
                    feature_dim = self.feature_dim
                )
            elif self.backbone_type == "gapartnet":
                if self.use_pretrain:
                    self.backbone = GAPartNet(
                        model_cfg["Spconv"]["in_channels"],model_cfg["Spconv"]["num_classes"],model_cfg["Spconv"]["channels"],pretrained_model_path = model_cfg["Spconv"]["ckpt_pth"])
                else:
                    self.backbone = GAPartNet(
                        model_cfg["Spconv"]["in_channels"],model_cfg["Spconv"]["num_classes"],)
            elif self.backbone_type == "segspconv":
                if self.use_pretrain and model_cfg["SegSpconv"]["ckpt_pth"] != "None":
                    self.backbone = SegSpconvBackbone(
                        model_cfg["SegSpconv"]["in_channels"],model_cfg["SegSpconv"]["num_classes"],model_cfg["SegSpconv"]["channels_down"],model_cfg["SegSpconv"]["channels_up"],block_repeat=model_cfg["SegSpconv"]["block_repeat"],pretrained_model_path = model_cfg["SegSpconv"]["ckpt_pth"])
                else:
                    self.backbone = SegSpconvBackbone(
                        model_cfg["SegSpconv"]["in_channels"],model_cfg["SegSpconv"]["num_classes"],model_cfg["SegSpconv"]["channels_down"],model_cfg["SegSpconv"]["channels_up"])
            else:
                print("Unknow Backbone!")
                exit(123)

        if self.freeze_backbone:
            self.backbone.freeze()
            self.backbone.eval()

        actor_layers = []
        actor_layers.append(nn.Linear(obs_shape[0] + self.feature_dim, actor_hidden_dim[0]))

        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor1 = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(obs_shape[0] + self.feature_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic1 = nn.Sequential(*critic_layers)

        
        if self.use_pc:
            if self.backbone_type == "segspconv":
                self.actor = MyNetPC_new(self.backbone, self.actor1)
                self.critic = MyNetPC_new(self.backbone, self.critic1)
            else:
                self.actor = MyNetPC(self.backbone, self.actor1)
                self.critic = MyNetPC(self.backbone, self.critic1)

        else:
            self.actor = MyNetState(self.actor1)
            self.critic = MyNetState(self.critic1)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        # actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        # actor_weights.append(0.01)
        # critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        # critic_weights.append(1.0)
        # self.init_weights(self.actor, actor_weights)
        # self.init_weights(self.critic, critic_weights)


    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    
    def forward(self):
        raise NotImplementedError

    def get_features(self, observations_pc):
        features = self.backbone(observations_pc)
        return features.detach()

    def act(self, observations, require_grad= True):
        # tmp_observations = {}
        # tmp_observations["pc"] = observations["pc"]
        # tmp_observations["state"] = observations["state"][:, self.qpose_num:](
        
        # if not require_grad:
        #     with torch.no_grad():
        #         actions_mean, others = self.actor(observations)
        # else:
        actions_mean, others = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)
        value, _ = self.critic(observations)
        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach(), others

    def inference_bbox(self, proposals):
        pt_xyz = proposals.pt_xyz.detach()
        batch_indices = proposals.batch_indices
        proposal_offsets = proposals.proposal_offsets
        num_points_per_proposal = proposals.num_points_per_proposal
        num_proposals = num_points_per_proposal.shape[0]
        npcs_preds = proposals.npcs_preds
        score_preds= proposals.score_preds


        # indices = torch.arange(sem_preds.shape[0], dtype=torch.int64, device=device)
        # propsoal_indices = indices[proposals.valid_mask][proposals.sorted_indices]

        bboxes = [[] for _ in range(batch_indices.max()+1)]
        for i in range(num_proposals):
            offset_begin = proposal_offsets[i].item()
            offset_end = proposal_offsets[i + 1].item()

            batch_idx = batch_indices[offset_begin]
            xyz_i = pt_xyz[offset_begin:offset_end]
            npcs_i = npcs_preds[offset_begin:offset_end]

            npcs_i = npcs_i - 0.5

            bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                estimate_pose_from_npcs(xyz_i.cpu().numpy(), npcs_i.cpu().numpy())

            bboxes[batch_idx].append(bbox_xyz.tolist())

        return bboxes


    def act_inference(self, observations):
        # tmp_observations = {}
        # tmp_observations["pc"] = observations["pc"]
        # tmp_observations["state"] = observations["state"][:, self.qpose_num:]
        # with torch.no_grad():
            # proposals = self.actor(observations)
            
            # import pdb
            # pdb.set_trace()
            # bboxes = self.inference_bbox(proposals)
        actions_mean, others = self.actor(observations)
        return actions_mean.detach(), others


    def evaluate(self, observations,  actions):

        actions_mean, _ = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value, _ = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


class Discriminator(nn.Module):

    def __init__(self, model_cfg, obs_shape, actions_shape, feature_dim = 0, device=None) -> None:
        super(Discriminator, self).__init__()
        self.device = device
        self.model_cfg = model_cfg
        self.state_dim = model_cfg["task_meta"]["state_dim"] # feature_dim + state_dim = MLP input_dim
        hidden_dim = model_cfg['hid_sizes']
        activation = get_activation(model_cfg['activation'])
        self.use_feature = model_cfg["use_feature"]
        self.feature_dim = feature_dim
        self.use_gt_obs = model_cfg["use_gt_obs"]

        input_dim = actions_shape[0]
        if self.use_feature:
            input_dim += self.feature_dim
        if self.use_gt_obs:
            input_dim += obs_shape[0]

        actor_layers = []
        actor_layers.append(nn.Linear(input_dim, hidden_dim[0]))
        actor_layers.append(activation)

        for l in range(len(hidden_dim)):

            if l == len(hidden_dim) - 1:
                actor_layers.append(nn.Linear(hidden_dim[l], 1))
            else:
                if model_cfg["use_bn"]:
                    actor_layers.append(nn.BatchNorm1d(hidden_dim[l]))
                if model_cfg["use_dropout"]:
                    actor_layers.append(nn.Dropout(model_cfg["dropout_p"]))
                actor_layers.append(nn.Linear(hidden_dim[l], hidden_dim[l + 1]))
                actor_layers.append(activation)
                
        
       
        self.disc = nn.Sequential(*actor_layers).to(self.device)

        # Initialize the weights like in stable baselines
        self.init_weights(self.disc)

    def act(self, observations, actions, features = None):
        raise NotImplementedError

    def forward(self, observations, actions, features):
        
        input_ = actions
        if self.use_feature:
            input_ = torch.cat((features, input_), dim = 1)
        if self.use_gt_obs:
            input_ = torch.cat((observations, input_), dim = 1)
        value = self.disc(input_)
        return value

    @staticmethod
    def init_weights(sequential):
        [torch.nn.init.xavier_normal_(module.weight) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape).cuda()
        self.S = torch.zeros(shape).cuda()
        self.std = torch.sqrt(self.S).cuda()

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)

        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = torch.zeros(self.shape).cuda()

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = torch.zeros(self.shape).cuda()