import functools
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import spconv.pytorch as spconv
import torch
import torch.nn as nn

from perception.structures.point_cloud import PointCloud      
import copy
from epic_ops.voxelize import voxelize

import torch.nn.functional as F
from einops import rearrange, repeat
from epic_ops.iou import batch_instance_seg_iou
from epic_ops.reduce import segmented_maxpool

from perception.metrics.segmentation import mean_iou, pixel_accuracy
from perception.structures.instances import Instances
from perception.structures.point_cloud import PointCloud
from perception.structures.segmentation import Segmentation
from perception.utils.symmetry_matrix import get_symmetry_matrix
from perception.losses.focal_loss import focal_loss
from perception.losses.dice_loss import dice_loss
from perception.datasets.gapartnet_new import apply_voxelization, apply_voxelization_batch
# from .pointgroup_utils import (apply_nms, cluster_proposals, compute_ap,
#                                compute_npcs_loss, filter_invalid_proposals,
#                                get_gt_scores, segmented_voxelize)
from .sparse_unet_backbone import SparseUNet


class PointGroup(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: List[int],
        block_repeat: int = 2,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.unet = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        self.sem_seg_head = nn.Linear(channels[0], num_classes)


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
            
    # def forward_backbone(
    #     self,
    #     voxel_tensor: spconv.SparseConvTensor,
    #     pc_voxel_id: torch.Tensor,
    # ) -> Tuple[spconv.SparseConvTensor, torch.Tensor, torch.Tensor]:
    #     voxel_features, global_feats = self.unet(voxel_tensor)
    #     pt_features = voxel_features.features[pc_voxel_id]
    #     return voxel_features, pt_features, global_feats, 

    # def _training_or_validation_step(
    #     self,
    #     points,
    #     voxel_tensor,
    #     pc_voxel_id,
    #     batch_size, 
    #     mask_list: List,
    #     batch_idx: int,
    #     running_mode: str,
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     pt_xyz = points[:, :3]
    #     assert (pc_voxel_id >= 0).all()

    #     voxel_features, pt_features, global_feats = self.forward_backbone(
    #         voxel_tensor, pc_voxel_id
    #     )
    #     res = torch.zeros((batch_size, global_feats._features.shape[-1]), device = global_feats._features.device)
    #     for i in range(batch_size):
    #         res[i] = global_feats._features[global_feats.indices[:,0] == i].max(0)[0]
    #     return res

    def forward(self, input_pc, mask_list: List = None, batch_idx: int = 0, dataloader_idx: int = 0):
        batch_size = input_pc.shape[0]
        points = input_pc[:, :,:6].reshape((-1, 6))
        pc = PointCloud(scene_id=["train"],points=input_pc[:, :,:6],)
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = self.apply_voxelization_batch(pc,  voxel_size=[1. / 100, 1. / 100, 1. / 100])
        voxel_coords_range = (torch.max(voxel_coords, dim=0)[0] + 1).clamp(128, 100000000)
        voxel_coords = torch.cat((batch_indices.unsqueeze(1), voxel_coords, ), dim=1).int()
        voxel_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords,spatial_shape=voxel_coords_range.tolist(),batch_size=batch_size,)

        voxel_features, global_feats = self.unet(voxel_tensor)
        pt_features = voxel_features.features[pc_voxel_id]

        res = torch.zeros((batch_size, global_feats._features.shape[-1]), device = global_feats._features.device)
        for i in range(batch_size):
            res[i] = global_feats._features[global_feats.indices[:,0] == i].max(0)[0]
        return res

    def apply_voxelization_batch(
        pc: PointCloud, *, voxel_size: Tuple[float, float, float]
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
    # def validation_step(self, points, voxel_tensor, pc_voxel_id, mask_list: List = None, batch_idx: int = 0, dataloader_idx: int = 0):
    #     split = ["val", "intra", "inter"]
    #     global_features = self._training_or_validation_step(
    #         points, voxel_tensor, pc_voxel_id, mask_list, batch_idx, split[dataloader_idx]
    #     )
    #     return global_features

    def forward(self, points, voxel_tensor, pc_voxel_id, batch_size, mask_list: List = None, batch_idx: int = 0, dataloader_idx: int = 0):
        batch_size = points.shape[0]
        pc = PointCloud(scene_id=["train"],points=points[:, :, :6],)
        voxel_features, voxel_coords, batch_indices, pc_voxel_id = apply_voxelization_batch(pc,  voxel_size=[1. / 100, 1. / 100, 1. / 100])
        voxel_coords_range = (torch.max(voxel_coords, dim=0)[0] + 1).clamp(128, 100000000)
        voxel_coords = torch.cat((batch_indices.unsqueeze(1), voxel_coords, ), dim=1).int()
        voxel_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords,spatial_shape=voxel_coords_range.tolist(),batch_size=batch_size,)
        return self._training_or_validation_step(
            points, voxel_tensor, pc_voxel_id, batch_size, mask_list, batch_idx, "train"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        
