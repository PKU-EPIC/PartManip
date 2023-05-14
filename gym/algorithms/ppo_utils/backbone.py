import functools
from pyexpat import features
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.autograd import Function

from data_structure.pointcloud import PointCloud      
import copy
from epic_ops.voxelize import voxelize
from epic_ops.reduce import segmented_maxpool

from .sparse_unet_backbone import SparseUNet
from algorithms.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet
from .pointgroup_utils import (apply_nms, cluster_proposals, compute_ap,
                               compute_npcs_loss, filter_invalid_proposals,
                               get_gt_scores, segmented_voxelize)

from data_structure.instances import Instances

class SparseUnetBackbone(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: List[int],
        block_repeat: int = 2,
        pretrained_model_path: Optional[str] = None,
        skip_link: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.skip_link = skip_link

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        
        # if self.skip_link:
            
        # else: 
        #     self.unet = SparseUNet_NoSkip.build(in_channels, channels, block_repeat, norm_fn)
        self.unet = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
            
        self.seg_head = nn.Linear(channels[0], num_classes)


        if pretrained_model_path != None and pretrained_model_path != "None":
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            

    def forward(self, input_pc):
        batch_size = input_pc.shape[0]
        input_pc[:, :,:3] = input_pc[:, :,:3] - input_pc[:, :,:3].mean(0, keepdim=True).mean(1, keepdim=True)
        input_pc[:, :,0] = -input_pc[:, :,0]
        pc = PointCloud(scene_id=["train"],points=input_pc[:, :,:6],)
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = self.apply_voxelization_batch(pc,  voxel_size=[1. / 100, 1. / 100, 1. / 100])
        voxel_coords_range = (torch.max(voxel_coords, dim=0)[0] + 1).clamp(128, 100000000)
        voxel_coords = torch.cat((batch_indices.unsqueeze(1), voxel_coords, ), dim=1).int()
        voxel_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords,spatial_shape=voxel_coords_range.tolist(),batch_size=batch_size,)

        voxel_features, global_feats = self.unet(voxel_tensor)
        pt_features = voxel_features.features[pc_voxel_id]

        res = torch.zeros((batch_size, global_feats._features.shape[-1]), device = global_feats._features.device)
        others = {}
        for i in range(batch_size):
            if (global_feats.indices[:,0] == i).sum() == 0:
                continue ## ? have some bug, maybe
            res[i] = global_feats._features[global_feats.indices[:,0] == i].max(0)[0]
        return res, others

    def apply_voxelization_batch(
        self, pc: PointCloud, *, voxel_size: Tuple[float, float, float]
    ) -> PointCloud:
        pc = copy.copy(pc)

        batch_size = pc.points.shape[0]
        num_points = pc.points.shape[1]
        pt_xyz = pc.points[:, :, :3].reshape(-1,3)
        points_range_min = pt_xyz.min(0)[0] - 1e-4
        points_range_max = pt_xyz.max(0)[0] + 1e-4
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = voxelize(
            pt_xyz, pc.points.reshape((-1, 6)),
            batch_offsets=torch.as_tensor(list(range(batch_size+1)), dtype=torch.int64, device = pt_xyz.device)*num_points,
            voxel_size=torch.tensor([0.01, 0.01, 0.01], device = pt_xyz.device),
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            reduction="mean",)
        return voxel_features, voxel_coords, batch_indices, pc_voxel_id

class PointNetBackbone(pl.LightningModule):
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = getPointNet({
                'input_feature_dim': self.pc_dim,
                'feat_dim': self.feature_dim
            })

        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            
    
    def forward(self, input_pc):
        others = {}
        return self.backbone(input_pc), others

class SparseUnetBackbone_seg(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: List[int],
        block_repeat: int = 2,
        pretrained_model_path: Optional[str] = None,
        skip_link: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.skip_link = skip_link

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.unet = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        self.seg_head = nn.Linear(channels[0], num_classes)


        if pretrained_model_path != None and pretrained_model_path != "None":
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            

    def forward(self, input_pc):
        batch_size = input_pc.shape[0]
        input_pc[:, :,:3] = input_pc[:, :,:3] - input_pc[:, :,:3].mean(0, keepdim=True).mean(1, keepdim=True)
        input_pc[:, :,0] = -input_pc[:, :,0]
        pc = PointCloud(scene_id=["train"],points=input_pc[:, :,:6],)
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = self.apply_voxelization_batch(pc,  voxel_size=[1. / 100, 1. / 100, 1. / 100])
        voxel_coords_range = (torch.max(voxel_coords, dim=0)[0] + 1).clamp(128, 100000000)
        voxel_coords = torch.cat((batch_indices.unsqueeze(1), voxel_coords, ), dim=1).int()

        voxel_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords,spatial_shape=voxel_coords_range.tolist(),batch_size=batch_size,)

        voxel_features = self.unet(voxel_tensor)
        pt_features = voxel_features.features[pc_voxel_id]
        sem_logits = self.seg_head(voxel_features.features)[pc_voxel_id]
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)

        res = torch.zeros((batch_size, global_feats._features.shape[-1]), device = global_feats._features.device)
        
        others = {}
        others["sem_logits"] = sem_logits
        others["sem_preds"] = sem_preds
        
        
        for i in range(batch_size):
            if (global_feats.indices[:,0] == i).sum() == 0:
                continue ## ? have some bug, maybe
            res[i] = global_feats._features[global_feats.indices[:,0] == i].max(0)[0]
        return res, others

    def apply_voxelization_batch(
        self, pc: PointCloud, *, voxel_size: Tuple[float, float, float]
    ) -> PointCloud:
        pc = copy.copy(pc)

        batch_size = pc.points.shape[0]
        num_points = pc.points.shape[1]
        pt_xyz = pc.points[:, :, :3].reshape(-1,3)
        points_range_min = pt_xyz.min(0)[0] - 1e-4
        points_range_max = pt_xyz.max(0)[0] + 1e-4
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = voxelize(
            pt_xyz, pc.points.reshape((-1, pc.points.shape[-1])),
            batch_offsets=torch.as_tensor(list(range(batch_size+1)), dtype=torch.int64, device = pt_xyz.device)*num_points,
            voxel_size=torch.tensor([0.01, 0.01, 0.01], device = pt_xyz.device),
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            reduction="mean",)
        return voxel_features, voxel_coords, batch_indices, pc_voxel_id

class SparseUnetBackbone_pn(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: List[int],
        block_repeat: int = 2,
        pretrained_model_path: Optional[str] = None,
        use_domain_discrimination: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.unet = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        self.seg_head = nn.Linear(channels[0], num_classes)
        self.pn = getPointNet({
                'input_feature_dim': 6 + channels[0],
                'feat_dim': channels[-1],
            })

        # self.pn = getPointNet({
        #         'input_feature_dim': 6,
        #         'feat_dim': channels[-1],
        #     })
        
        # self.domain_head = DomainHead(input_dim, classes)


        if pretrained_model_path != None and pretrained_model_path != "None":
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            

    def forward(self, input_pc):
        batch_size = input_pc.shape[0]
        input_pc[:, :,:3] = input_pc[:, :,:3] - input_pc[:, :,:3].mean(0, keepdim=True).mean(1, keepdim=True)
        input_pc[:, :,0] = -input_pc[:, :,0]

        pc = PointCloud(scene_id=["train"],points=input_pc)

        voxel_features, voxel_coords, batch_indices, pc_voxel_id = self.apply_voxelization_batch(pc,  voxel_size=[1. / 100, 1. / 100, 1. / 100])
        voxel_coords_range = (torch.max(voxel_coords, dim=0)[0] + 1).clamp(128, 100000000)
        voxel_coords = torch.cat((batch_indices.unsqueeze(1), voxel_coords, ), dim=1).int()

        voxel_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords,spatial_shape=voxel_coords_range.tolist(),batch_size=batch_size,)

        voxel_features = self.unet(voxel_tensor)
        pt_features = voxel_features.features[pc_voxel_id]
        sem_logits = self.seg_head(voxel_features.features)[pc_voxel_id]
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)
        # import pdb
        # pdb.set_trace()
        # pt_features[:] = 0.
        res = self.pn(torch.cat((input_pc[..., :6], pt_features.reshape(batch_size, -1, pt_features.shape[-1])), dim = -1))
        # res = self.pn(input_pc[..., :6])

        others = {}
        others["sem_logits"] = sem_logits
        others["sem_preds"] = sem_preds
        
        
        # for i in range(batch_size):
        #     if (global_feats.indices[:,0] == i).sum() == 0:
        #         continue ## ? have some bug, maybe
        #     res[i] = global_feats._features[global_feats.indices[:,0] == i].max(0)[0]
        return res, others

    def apply_voxelization_batch(
        self, pc: PointCloud, *, voxel_size: Tuple[float, float, float]
    ) -> PointCloud:
        pc = copy.copy(pc)

        batch_size = pc.points.shape[0]
        num_points = pc.points.shape[1]
        pt_xyz = pc.points[:, :, :3].reshape(-1,3)
        points_range_min = pt_xyz.min(0)[0] - 1e-4

        points_range_max = pt_xyz.max(0)[0] + 1e-4
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = voxelize(
            pt_xyz, pc.points.reshape((-1, pc.points.shape[-1])),
            batch_offsets=torch.as_tensor(list(range(batch_size+1)), dtype=torch.int64, device = pt_xyz.device)*num_points,
            voxel_size=torch.tensor([0.01, 0.01, 0.01], device = pt_xyz.device),
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            reduction="mean",)
        return voxel_features, voxel_coords, batch_indices, pc_voxel_id

class GAPartNet_(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: List[int],
        block_repeat: int = 2,
        pretrained_model_path: Optional[str] = None,
        # ignore_sem_label: int = -100,
        # ignore_instance_label: int = -100,
        ball_query_radius: float = 0.03,
        max_num_points_per_query: int = 50,
        max_num_points_per_query_shift: int = 300,
        min_num_points_per_proposal: int = 50,
        score_fullscale: float = 14,
        score_scale: float = 50,
        # symmetry_indices: Optional[List[int]] = None,
        val_score_threshold: float = 0.09,
        val_min_num_points_per_proposal: int = 3,
        val_nms_iou_threshold: float = 0.3,
        val_ap_iou_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.ball_query_radius = ball_query_radius
        self.max_num_points_per_query = max_num_points_per_query
        self.max_num_points_per_query_shift = max_num_points_per_query_shift
        self.min_num_points_per_proposal = min_num_points_per_proposal
        self.score_fullscale = score_fullscale
        self.score_scale = score_scale

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.unet = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        self.sem_seg_head = nn.Linear(channels[0], num_classes)
        self.offset_head = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            norm_fn(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], 3),
        )

        self.score_unet = SparseUNet.build(
            channels[0], channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.score_head = nn.Linear(channels[0], num_classes - 1)

        self.npcs_unet = SparseUNet.build(
            channels[0], channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.npcs_head = nn.Linear(channels[0], 3 * (num_classes - 1))



        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            
        self.val_score_threshold = val_score_threshold
        self.val_min_num_points_per_proposal = val_min_num_points_per_proposal
        self.val_nms_iou_threshold = val_nms_iou_threshold
        self.val_ap_iou_threshold = val_ap_iou_threshold



    def cluster_proposals_and_revoxelize(
        self,
        pt_xyz: torch.Tensor,
        batch_indices: torch.Tensor,
        pt_features: torch.Tensor,
        sem_preds: torch.Tensor,
        pt_offsets: torch.Tensor,
        instance_labels: Optional[torch.Tensor],
    ):
        device = pt_xyz.device

        # get rid of stuff classes (e.g. wall)
        if instance_labels is not None:
            valid_mask = (sem_preds > 0) & (instance_labels >= 0)
        else:
            valid_mask = sem_preds > 0

        pt_xyz = pt_xyz[valid_mask]
        if pt_xyz.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[valid_mask]
        pt_features = pt_features[valid_mask]
        sem_preds = sem_preds[valid_mask].int()
        pt_offsets = pt_offsets[valid_mask]
        if instance_labels is not None:
            instance_labels = instance_labels[valid_mask]

        # get batch offsets (csr) from batch indices
        _, batch_indices_compact, num_points_per_batch = torch.unique_consecutive(
            batch_indices, return_inverse=True, return_counts=True
        )
        batch_indices_compact = batch_indices_compact.int()
        batch_offsets = torch.zeros(
            (num_points_per_batch.shape[0] + 1,), dtype=torch.int32, device=device
        )
        batch_offsets[1:] = num_points_per_batch.cumsum(0)

        # cluster proposals
        sorted_cc_labels, sorted_indices = cluster_proposals(
            pt_xyz, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query,
        )

        sorted_cc_labels_shift, sorted_indices_shift = cluster_proposals(
            pt_xyz + pt_offsets, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query_shift,
        )

        # combine clusters
        sorted_cc_labels = torch.cat([
            sorted_cc_labels,
            sorted_cc_labels_shift + sorted_cc_labels.shape[0],
        ], dim=0)
        sorted_indices = torch.cat([sorted_indices, sorted_indices_shift], dim=0)

        # compact the proposal ids
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            sorted_cc_labels, return_inverse=True, return_counts=True
        )

        # remove small proposals
        valid_proposal_mask = (
            num_points_per_proposal >= self.min_num_points_per_proposal
        )
        # proposal to point
        valid_point_mask = valid_proposal_mask[proposal_indices]

        sorted_indices = sorted_indices[valid_point_mask]
        if sorted_indices.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[sorted_indices]
        pt_xyz = pt_xyz[sorted_indices]
        pt_features = pt_features[sorted_indices]
        sem_preds = sem_preds[sorted_indices]
        if instance_labels is not None:
            instance_labels = instance_labels[sorted_indices]

        # re-compact the proposal ids
        proposal_indices = proposal_indices[valid_point_mask]
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            proposal_indices, return_inverse=True, return_counts=True
        )
        num_proposals = num_points_per_proposal.shape[0]

        # get proposal batch offsets
        proposal_offsets = torch.zeros(
            num_proposals + 1, dtype=torch.int32, device=device
        )
        proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

        # voxelization
        voxel_features, voxel_coords, pc_voxel_id = segmented_voxelize(
            pt_xyz, pt_features,
            proposal_offsets, proposal_indices,
            num_points_per_proposal,
            self.score_fullscale, self.score_scale,
        )
        voxel_tensor = spconv.SparseConvTensor(
            voxel_features, voxel_coords.int(),
            spatial_shape=[self.score_fullscale] * 3,
            batch_size=num_proposals,
        )
        
        assert (pc_voxel_id >= 0).all()

        proposals = Instances(
            valid_mask=valid_mask,
            sorted_indices=sorted_indices,
            pt_xyz=pt_xyz,
            batch_indices=batch_indices,
            proposal_offsets=proposal_offsets,
            proposal_indices=proposal_indices,
            num_points_per_proposal=num_points_per_proposal,
            sem_preds=sem_preds,
            instance_labels=instance_labels,
        )

        return voxel_tensor, pc_voxel_id, proposals

    def forward(self, input_pc):
        batch_size = input_pc.shape[0]
        input_pc[:, :,:3] = input_pc[:, :,:3] - input_pc[:, :,:3].mean(0, keepdim=True).mean(1, keepdim=True)
        input_pc[:, :,0] = -input_pc[:, :,0]
        
        pc = PointCloud(scene_id=["train"],points=input_pc[:, :,:6],)
        voxel_features, voxel_coords, voxel_batch_indices, pc_voxel_id = self.apply_voxelization_batch(pc,  voxel_size=[1. / 100, 1. / 100, 1. / 100])
        voxel_coords_range = (torch.max(voxel_coords, dim=0)[0] + 1).clamp(128, 100000000)
        voxel_coords = torch.cat((voxel_batch_indices.unsqueeze(1), voxel_coords, ), dim=1).int()
        voxel_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords,spatial_shape=voxel_coords_range.tolist(),batch_size=batch_size,)

        voxel_features, global_feats = self.unet(voxel_tensor)
        
        pt_features = voxel_features.features[pc_voxel_id]
        sem_logits = self.sem_seg_head(voxel_features.features)[pc_voxel_id]
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)

        batch_indices = (torch.ones(pc.points.shape[:2], dtype=torch.int32, device=pc.points.device) * torch.arange(pc.points.shape[0], dtype=torch.int32, device=pc.points.device).view(-1, 1)).reshape(-1)
        pt_xyz = pc.points.reshape(-1, 6)[:,:3]
        pt_offsets = self.offset_head(voxel_features.features)[pc_voxel_id]
        (voxel_tensor, pc_voxel_id, proposals) = self.cluster_proposals_and_revoxelize(pt_xyz, batch_indices, pt_features,sem_preds, pt_offsets, None,)

        # score
        proposal_offsets = proposals.proposal_offsets
        proposal_offsets_begin = proposal_offsets[:-1]
        proposal_offsets_end = proposal_offsets[1:]

        score_features, _ = self.score_unet(voxel_tensor)
        score_features = score_features.features[pc_voxel_id]
        pooled_score_features, _ = segmented_maxpool(
            score_features, proposal_offsets_begin, proposal_offsets_end
        )
        score_logits = self.score_head(pooled_score_features)
        proposal_offsets_begin = proposals.proposal_offsets[:-1].long()
        if proposals.sem_labels is not None:
            proposal_sem_labels = proposals.sem_labels[proposal_offsets_begin].long()
        else:
            proposal_sem_labels = proposals.sem_preds[proposal_offsets_begin].long()
        score_logits = score_logits.gather(
            1, proposal_sem_labels[:, None] - 1
        ).squeeze(1)
        proposals.score_preds = score_logits.detach().sigmoid()
        npcs_features = self.npcs_unet(voxel_tensor)
        npcs_logits = self.npcs_head(npcs_features.features)
        npcs_logits = npcs_logits[pc_voxel_id]
        npcs_preds = npcs_logits.detach()
        npcs_preds = rearrange(npcs_preds, "n (k c) -> n k c", c=3)
        npcs_preds = npcs_preds.gather(
            1, index=repeat(proposals.sem_preds.long() - 1, "n -> n one c", one=1, c=3)
        ).squeeze(1)
        proposals.npcs_preds = npcs_preds
        return proposals

    def apply_voxelization_batch(
        self, pc: PointCloud, *, voxel_size: Tuple[float, float, float]
    ) -> PointCloud:
        pc = copy.copy(pc)

        batch_size = pc.points.shape[0]
        num_points = pc.points.shape[1]
        pt_xyz = pc.points[:, :, :3].reshape(-1,3)
        points_range_min = pt_xyz.min(0)[0] - 1e-4
        points_range_max = pt_xyz.max(0)[0] + 1e-4
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = voxelize(
            pt_xyz, pc.points.reshape((-1, 6)),
            batch_offsets=torch.as_tensor(list(range(batch_size+1)), dtype=torch.int64, device = pt_xyz.device)*num_points,
            voxel_size=torch.tensor([0.01, 0.01, 0.01], device = pt_xyz.device),
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            reduction="mean",)
        return voxel_features, voxel_coords, batch_indices, pc_voxel_id

class GAPartNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: List[int],
        block_repeat: int = 2,
        pretrained_model_path: Optional[str] = None,
        # ignore_sem_label: int = -100,
        # ignore_instance_label: int = -100,
        ball_query_radius: float = 0.03,
        max_num_points_per_query: int = 50,
        max_num_points_per_query_shift: int = 300,
        min_num_points_per_proposal: int = 50,
        score_fullscale: float = 14,
        score_scale: float = 50,
        # symmetry_indices: Optional[List[int]] = None,
        val_score_threshold: float = 0.09,
        val_min_num_points_per_proposal: int = 3,
        val_nms_iou_threshold: float = 0.3,
        val_ap_iou_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.ball_query_radius = ball_query_radius
        self.max_num_points_per_query = max_num_points_per_query
        self.max_num_points_per_query_shift = max_num_points_per_query_shift
        self.min_num_points_per_proposal = min_num_points_per_proposal
        self.score_fullscale = score_fullscale
        self.score_scale = score_scale

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.unet = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        self.sem_seg_head = nn.Linear(channels[0], num_classes)
        self.offset_head = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            norm_fn(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], 3),
        )

        self.score_unet = SparseUNet.build(
            channels[0], channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.score_head = nn.Linear(channels[0], num_classes - 1)

        self.npcs_unet = SparseUNet.build(
            channels[0], channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.npcs_head = nn.Linear(channels[0], 3 * (num_classes - 1))



        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            
        self.val_score_threshold = val_score_threshold
        self.val_min_num_points_per_proposal = val_min_num_points_per_proposal
        self.val_nms_iou_threshold = val_nms_iou_threshold
        self.val_ap_iou_threshold = val_ap_iou_threshold


    def cluster_proposals_and_revoxelize(
        self,
        pt_xyz: torch.Tensor,
        batch_indices: torch.Tensor,
        pt_features: torch.Tensor,
        sem_preds: torch.Tensor,
        pt_offsets: torch.Tensor,
        instance_labels: Optional[torch.Tensor],
    ):
        device = pt_xyz.device

        # get rid of stuff classes (e.g. wall)
        if instance_labels is not None:
            valid_mask = (sem_preds > 0) & (instance_labels >= 0)
        else:
            valid_mask = sem_preds > 0

        pt_xyz = pt_xyz[valid_mask]
        if pt_xyz.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[valid_mask]
        pt_features = pt_features[valid_mask]
        sem_preds = sem_preds[valid_mask].int()
        pt_offsets = pt_offsets[valid_mask]
        if instance_labels is not None:
            instance_labels = instance_labels[valid_mask]

        # get batch offsets (csr) from batch indices
        _, batch_indices_compact, num_points_per_batch = torch.unique_consecutive(
            batch_indices, return_inverse=True, return_counts=True
        )
        batch_indices_compact = batch_indices_compact.int()
        batch_offsets = torch.zeros(
            (num_points_per_batch.shape[0] + 1,), dtype=torch.int32, device=device
        )
        batch_offsets[1:] = num_points_per_batch.cumsum(0)

        # cluster proposals
        sorted_cc_labels, sorted_indices = cluster_proposals(
            pt_xyz, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query,
        )

        sorted_cc_labels_shift, sorted_indices_shift = cluster_proposals(
            pt_xyz + pt_offsets, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query_shift,
        )

        # combine clusters
        sorted_cc_labels = torch.cat([
            sorted_cc_labels,
            sorted_cc_labels_shift + sorted_cc_labels.shape[0],
        ], dim=0)
        sorted_indices = torch.cat([sorted_indices, sorted_indices_shift], dim=0)

        # compact the proposal ids
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            sorted_cc_labels, return_inverse=True, return_counts=True
        )

        # remove small proposals
        valid_proposal_mask = (
            num_points_per_proposal >= self.min_num_points_per_proposal
        )
        # proposal to point
        valid_point_mask = valid_proposal_mask[proposal_indices]

        sorted_indices = sorted_indices[valid_point_mask]
        if sorted_indices.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[sorted_indices]
        pt_xyz = pt_xyz[sorted_indices]
        pt_features = pt_features[sorted_indices]
        sem_preds = sem_preds[sorted_indices]
        if instance_labels is not None:
            instance_labels = instance_labels[sorted_indices]

        # re-compact the proposal ids
        proposal_indices = proposal_indices[valid_point_mask]
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            proposal_indices, return_inverse=True, return_counts=True
        )
        num_proposals = num_points_per_proposal.shape[0]

        # get proposal batch offsets
        proposal_offsets = torch.zeros(
            num_proposals + 1, dtype=torch.int32, device=device
        )
        proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

        # voxelization
        voxel_features, voxel_coords, pc_voxel_id = segmented_voxelize(
            pt_xyz, pt_features,
            proposal_offsets, proposal_indices,
            num_points_per_proposal,
            self.score_fullscale, self.score_scale,
        )
        voxel_tensor = spconv.SparseConvTensor(
            voxel_features, voxel_coords.int(),
            spatial_shape=[self.score_fullscale] * 3,
            batch_size=num_proposals,
        )
        assert (pc_voxel_id >= 0).all()

        proposals = Instances(
            valid_mask=valid_mask,
            sorted_indices=sorted_indices,
            pt_xyz=pt_xyz,
            batch_indices=batch_indices,
            proposal_offsets=proposal_offsets,
            proposal_indices=proposal_indices,
            num_points_per_proposal=num_points_per_proposal,
            sem_preds=sem_preds,
            instance_labels=instance_labels,
        )

        return voxel_tensor, pc_voxel_id, proposals

    def forward_sem_seg(
        self,
        voxel_tensor: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
    ) -> Tuple[spconv.SparseConvTensor, torch.Tensor, torch.Tensor]:
        voxel_features= self.unet(voxel_tensor)
        sem_logits = self.sem_seg_head(voxel_features.features)

        pt_features = voxel_features.features[pc_voxel_id]
        sem_logits = sem_logits[pc_voxel_id]

        return voxel_features, pt_features, sem_logits

    def forward_pt_offset(
        self,
        voxel_features: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
    ) -> torch.Tensor:
        pt_offsets = self.offset_head(voxel_features.features)
        return pt_offsets[pc_voxel_id]

    def forward_proposal_score(
        self,
        voxel_tensor: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
        proposals: Instances,
    ):
        proposal_offsets = proposals.proposal_offsets
        proposal_offsets_begin = proposal_offsets[:-1]
        proposal_offsets_end = proposal_offsets[1:]

        score_features = self.score_unet(voxel_tensor)
        score_features = score_features.features[pc_voxel_id]
        pooled_score_features, _ = segmented_maxpool(
            score_features, proposal_offsets_begin, proposal_offsets_end
        )
        score_logits = self.score_head(pooled_score_features)

        return score_logits

    def forward_proposal_npcs(
        self,
        voxel_tensor: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
    ) -> torch.Tensor:
        npcs_features = self.npcs_unet(voxel_tensor)
        npcs_logits = self.npcs_head(npcs_features.features)
        npcs_logits = npcs_logits[pc_voxel_id]

        return npcs_logits

    def forward(self, input_pc):
        batch_size = input_pc.shape[0]
        input_pc[:, :,:3] = input_pc[:, :,:3] - input_pc[:, :,:3].mean(0, keepdim=True).mean(1, keepdim=True)
        input_pc[:, :,0] = -input_pc[:, :,0]
        
        pc = PointCloud(scene_id=["train"],points=input_pc[:, :,:6],)
        voxel_features, voxel_coords, voxel_batch_indices, pc_voxel_id = self.apply_voxelization_batch(pc,  voxel_size=[1. / 100, 1. / 100, 1. / 100])
        voxel_coords_range = (torch.max(voxel_coords, dim=0)[0] + 1).clamp(128, 100000000)
        voxel_coords = torch.cat((voxel_batch_indices.unsqueeze(1), voxel_coords, ), dim=1).int()
        voxel_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords,spatial_shape=voxel_coords_range.tolist(),batch_size=batch_size,)

        # semantic segmentation
        voxel_features, pt_features, sem_logits = self.forward_sem_seg(
            voxel_tensor, pc_voxel_id
        )
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)

        # point offset
        pt_offsets = self.forward_pt_offset(voxel_features, pc_voxel_id)
        batch_indices = (torch.ones(pc.points.shape[:2], dtype=torch.int32, device=pc.points.device) * torch.arange(pc.points.shape[0], dtype=torch.int32, device=pc.points.device).view(-1, 1)).reshape(-1)
        pt_xyz = pc.points.reshape(-1, 6)[:,:3]

        (
            voxel_tensor, pc_voxel_id, proposals
        ) = self.cluster_proposals_and_revoxelize(
            pt_xyz, batch_indices, pt_features,
            sem_preds, pt_offsets, None,
        )

        # clustering and scoring
        score_logits = self.forward_proposal_score(
            voxel_tensor, pc_voxel_id, proposals
        )
        proposal_offsets_begin = proposals.proposal_offsets[:-1].long()

        if proposals.sem_labels is not None:
            proposal_sem_labels = proposals.sem_labels[proposal_offsets_begin].long()
        else:
            proposal_sem_labels = proposals.sem_preds[proposal_offsets_begin].long()
        score_logits = score_logits.gather(
            1, proposal_sem_labels[:, None] - 1
        ).squeeze(1)
        proposals.score_preds = score_logits.detach().sigmoid()

        npcs_logits = self.forward_proposal_npcs(
            voxel_tensor, pc_voxel_id
        )
        npcs_preds = npcs_logits.detach()
        npcs_preds = rearrange(npcs_preds, "n (k c) -> n k c", c=3)
        npcs_preds = npcs_preds.gather(
            1, index=repeat(proposals.sem_preds.long() - 1, "n -> n one c", one=1, c=3)
        ).squeeze(1)
        proposals.npcs_preds = npcs_preds
        return proposals


    def apply_voxelization_batch(
        self, pc: PointCloud, *, voxel_size: Tuple[float, float, float]
    ) -> PointCloud:
        pc = copy.copy(pc)

        batch_size = pc.points.shape[0]
        num_points = pc.points.shape[1]
        pt_xyz = pc.points[:, :, :3].reshape(-1,3)
        points_range_min = pt_xyz.min(0)[0] - 1e-4
        points_range_max = pt_xyz.max(0)[0] + 1e-4
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = voxelize(
            pt_xyz, pc.points.reshape((-1, 6)),
            batch_offsets=torch.as_tensor(list(range(batch_size+1)), dtype=torch.int64, device = pt_xyz.device)*num_points,
            voxel_size=torch.tensor([0.01, 0.01, 0.01], device = pt_xyz.device),
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            reduction="mean",)
        return voxel_features, voxel_coords, batch_indices, pc_voxel_id

class GAPartNet_(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: List[int],
        block_repeat: int = 2,
        pretrained_model_path: Optional[str] = None,
        # ignore_sem_label: int = -100,
        # ignore_instance_label: int = -100,
        # ball_query_radius: float = 0.03,
        # max_num_points_per_query: int = 50,
        # max_num_points_per_query_shift: int = 300,
        # min_num_points_per_proposal: int = 50,
        # score_fullscale: float = 14,
        # score_scale: float = 50,
        # symmetry_indices: Optional[List[int]] = None,
        val_score_threshold: float = 0.09,
        val_min_num_points_per_proposal: int = 3,
        val_nms_iou_threshold: float = 0.3,
        val_ap_iou_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.unet = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        self.sem_seg_head = nn.Linear(channels[0], num_classes)
        self.offset_head = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            norm_fn(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], 3),
        )

        self.score_unet = SparseUNet.build(
            channels[0], channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.score_head = nn.Linear(channels[0], num_classes - 1)

        self.npcs_unet = SparseUNet.build(
            channels[0], channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.npcs_head = nn.Linear(channels[0], 3 * (num_classes - 1))



        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            
        self.val_score_threshold = val_score_threshold
        self.val_min_num_points_per_proposal = val_min_num_points_per_proposal
        self.val_nms_iou_threshold = val_nms_iou_threshold
        self.val_ap_iou_threshold = val_ap_iou_threshold



    def cluster_proposals_and_revoxelize(
        self,
        pt_xyz: torch.Tensor,
        batch_indices: torch.Tensor,
        pt_features: torch.Tensor,
        sem_preds: torch.Tensor,
        pt_offsets: torch.Tensor,
        instance_labels: Optional[torch.Tensor],
    ):
        device = pt_xyz.device

        # get rid of stuff classes (e.g. wall)
        if instance_labels is not None:
            valid_mask = (sem_preds > 0) & (instance_labels >= 0)
        else:
            valid_mask = sem_preds > 0

        pt_xyz = pt_xyz[valid_mask]
        if pt_xyz.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[valid_mask]
        pt_features = pt_features[valid_mask]
        sem_preds = sem_preds[valid_mask].int()
        pt_offsets = pt_offsets[valid_mask]
        if instance_labels is not None:
            instance_labels = instance_labels[valid_mask]

        # get batch offsets (csr) from batch indices
        _, batch_indices_compact, num_points_per_batch = torch.unique_consecutive(
            batch_indices, return_inverse=True, return_counts=True
        )
        batch_indices_compact = batch_indices_compact.int()
        batch_offsets = torch.zeros(
            (num_points_per_batch.shape[0] + 1,), dtype=torch.int32, device=device
        )
        batch_offsets[1:] = num_points_per_batch.cumsum(0)

        # cluster proposals
        sorted_cc_labels, sorted_indices = cluster_proposals(
            pt_xyz, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query,
        )

        sorted_cc_labels_shift, sorted_indices_shift = cluster_proposals(
            pt_xyz + pt_offsets, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query_shift,
        )

        # combine clusters
        sorted_cc_labels = torch.cat([
            sorted_cc_labels,
            sorted_cc_labels_shift + sorted_cc_labels.shape[0],
        ], dim=0)
        sorted_indices = torch.cat([sorted_indices, sorted_indices_shift], dim=0)

        # compact the proposal ids
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            sorted_cc_labels, return_inverse=True, return_counts=True
        )

        # remove small proposals
        valid_proposal_mask = (
            num_points_per_proposal >= self.min_num_points_per_proposal
        )
        # proposal to point
        valid_point_mask = valid_proposal_mask[proposal_indices]

        sorted_indices = sorted_indices[valid_point_mask]
        if sorted_indices.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[sorted_indices]
        pt_xyz = pt_xyz[sorted_indices]
        pt_features = pt_features[sorted_indices]
        sem_preds = sem_preds[sorted_indices]
        if instance_labels is not None:
            instance_labels = instance_labels[sorted_indices]

        # re-compact the proposal ids
        proposal_indices = proposal_indices[valid_point_mask]
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            proposal_indices, return_inverse=True, return_counts=True
        )
        num_proposals = num_points_per_proposal.shape[0]

        # get proposal batch offsets
        proposal_offsets = torch.zeros(
            num_proposals + 1, dtype=torch.int32, device=device
        )
        proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

        # voxelization
        voxel_features, voxel_coords, pc_voxel_id = segmented_voxelize(
            pt_xyz, pt_features,
            proposal_offsets, proposal_indices,
            num_points_per_proposal,
            self.score_fullscale, self.score_scale,
        )
        voxel_tensor = spconv.SparseConvTensor(
            voxel_features, voxel_coords.int(),
            spatial_shape=[self.score_fullscale] * 3,
            batch_size=num_proposals,
        )
        assert (pc_voxel_id >= 0).all()

        proposals = Instances(
            valid_mask=valid_mask,
            sorted_indices=sorted_indices,
            pt_xyz=pt_xyz,
            batch_indices=batch_indices,
            proposal_offsets=proposal_offsets,
            proposal_indices=proposal_indices,
            num_points_per_proposal=num_points_per_proposal,
            sem_preds=sem_preds,
            instance_labels=instance_labels,
        )

        return voxel_tensor, pc_voxel_id, proposals

    def forward(self, input_pc):
        batch_size = input_pc.shape[0]
        input_pc[:, :,:3] = input_pc[:, :,:3] - input_pc[:, :,:3].mean(0, keepdim=True).mean(1, keepdim=True)
        input_pc[:, :,0] = -input_pc[:, :,0]
        
        pc = PointCloud(scene_id=["train"],points=input_pc[:, :,:6],)
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = self.apply_voxelization_batch(pc,  voxel_size=[1. / 100, 1. / 100, 1. / 100])
        voxel_coords_range = (torch.max(voxel_coords, dim=0)[0] + 1).clamp(128, 100000000)
        voxel_coords = torch.cat((batch_indices.unsqueeze(1), voxel_coords, ), dim=1).int()
        voxel_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords,spatial_shape=voxel_coords_range.tolist(),batch_size=batch_size,)

        voxel_features, global_feats = self.unet(voxel_tensor)
        pt_features = voxel_features.features[pc_voxel_id]
        sem_logits = self.sem_seg_head(voxel_features.features)[pc_voxel_id]
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)

        res = torch.zeros((batch_size, global_feats._features.shape[-1]), device = global_feats._features.device)
        for i in range(batch_size):
            if (global_feats.indices[:,0] == i).sum() == 0:
                continue ## ? have some bug, maybe
            res[i] = global_feats._features[global_feats.indices[:,0] == i].max(0)[0]
        return res
        
        return res, sem_preds

    def apply_voxelization_batch(
        self, pc: PointCloud, *, voxel_size: Tuple[float, float, float]
    ) -> PointCloud:
        pc = copy.copy(pc)

        batch_size = pc.points.shape[0]
        num_points = pc.points.shape[1]
        pt_xyz = pc.points[:, :, :3].reshape(-1,3)
        points_range_min = pt_xyz.min(0)[0] - 1e-4
        points_range_max = pt_xyz.max(0)[0] + 1e-4
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = voxelize(
            pt_xyz, pc.points.reshape((-1, 6)),
            batch_offsets=torch.as_tensor(list(range(batch_size+1)), dtype=torch.int64, device = pt_xyz.device)*num_points,
            voxel_size=torch.tensor([0.01, 0.01, 0.01], device = pt_xyz.device),
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            reduction="mean",)
        return voxel_features, voxel_coords, batch_indices, pc_voxel_id

class SegSpconvBackbone(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels_down: List[int],
        channels_up: List[int],
        block_repeat: int = 2,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels_down = channels_down
        self.channels_up = channels_up

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.unet = SparseUNet.build(in_channels, channels_down, channels_up, block_repeat, norm_fn)
        self.seg_head = nn.Linear(channels_up[0], num_classes)


        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            

    def forward(self, input_pc):
        batch_size = input_pc.shape[0]
        input_pc[:, :,:3] = input_pc[:, :,:3] - input_pc[:, :,:3].mean(0, keepdim=True).mean(1, keepdim=True)
        input_pc[:, :,0] = -input_pc[:, :,0]
        pc = PointCloud(scene_id=["train"],points=input_pc[:, :,:6],)
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = self.apply_voxelization_batch(pc,  voxel_size=[1. / 100, 1. / 100, 1. / 100])
        voxel_coords_range = (torch.max(voxel_coords, dim=0)[0] + 1).clamp(128, 100000000)
        voxel_coords = torch.cat((batch_indices.unsqueeze(1), voxel_coords, ), dim=1).int()
        voxel_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords,spatial_shape=voxel_coords_range.tolist(),batch_size=batch_size,)

        voxel_features = self.unet(voxel_tensor)
        sem_logits = self.seg_head(voxel_features.features)[pc_voxel_id]
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)

        res = torch.zeros((batch_size, voxel_features._features.shape[-1]), device = voxel_features._features.device)

        others = {}
        others["sem_logits"] = sem_logits
        others["sem_preds"] = sem_preds
        for i in range(batch_size):
            # if (voxel_features.indices[:,0] == i).sum() == 0:
            #     continue ## ? have some bug, maybe
            res[i] = voxel_features._features[voxel_features.indices[:,0] == i].max(0)[0]
        return res, others

    def apply_voxelization_batch(
        self, pc: PointCloud, *, voxel_size: Tuple[float, float, float]
    ) -> PointCloud:
        pc = copy.copy(pc)

        batch_size = pc.points.shape[0]
        num_points = pc.points.shape[1]
        pt_xyz = pc.points[:, :, :3].reshape(-1,3)
        points_range_min = pt_xyz.min(0)[0] - 1e-4
        points_range_max = pt_xyz.max(0)[0] + 1e-4
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = voxelize(
            pt_xyz, pc.points.reshape((-1, 6)),
            batch_offsets=torch.as_tensor(list(range(batch_size+1)), dtype=torch.int64, device = pt_xyz.device)*num_points,
            voxel_size=torch.tensor([0.01, 0.01, 0.01], device = pt_xyz.device),
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            reduction="mean",)
        return voxel_features, voxel_coords, batch_indices, pc_voxel_id
    
class DomainHead(pl.LightningModule):
    def __init__(self, input_dim, classes):
        self.input_dim = input_dim
        self.classes = classes
        self.pn = getPointNet({
                'input_feature_dim': self.pc_dim,
                'feat_dim': 256
            })
        
        self.cls_head =  nn.Linear(256, classes)
    
    def forward(input):
        return self.cls_head(self.pn(ReverseLayerF(input)))


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha = 0.1):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
