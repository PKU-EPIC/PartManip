import functools
from sqlite3 import dbapi2
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import spconv.pytorch as spconv
import torch
import torch.nn as nn
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

from .pointgroup_utils import (apply_nms, cluster_proposals, compute_ap,
                               compute_npcs_loss, filter_invalid_proposals,
                               get_gt_scores, segmented_voxelize)
from .sparse_unet_backbone import SparseUNet


class PointGroup(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: List[int],
        block_repeat: int = 2,
        learning_rate: float = 1e-3,
        ignore_sem_label: int = -100,
        ignore_instance_label: int = -100,
        ball_query_radius: float = 0.03,
        max_num_points_per_query: int = 50,
        max_num_points_per_query_shift: int = 300,
        min_num_points_per_proposal: int = 50,
        score_net_start_at: int = 100,
        score_fullscale: float = 14,
        score_scale: float = 50,
        npcs_net_start_at: int = 100,
        symmetry_indices: Optional[List[int]] = None,
        pretrained_model_path: Optional[str] = None,
        loss_sem_seg_weight: Optional[List[float]] = None,
        use_focal_loss: bool = False,
        use_dice_loss: bool = False,
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

        self.learning_rate = learning_rate
        self.ignore_sem_label = ignore_sem_label
        self.ignore_instance_label = ignore_instance_label
        self.ball_query_radius = ball_query_radius
        self.max_num_points_per_query = max_num_points_per_query
        self.max_num_points_per_query_shift = max_num_points_per_query_shift
        self.min_num_points_per_proposal = min_num_points_per_proposal

        self.score_net_start_at = score_net_start_at
        self.score_fullscale = score_fullscale
        self.score_scale = score_scale

        self.npcs_net_start_at = npcs_net_start_at
        self.register_buffer(
            "symmetry_indices", torch.as_tensor(symmetry_indices, dtype=torch.int64)
        )
        if symmetry_indices is not None:
            assert len(symmetry_indices) == num_classes, (symmetry_indices, num_classes)

            (
                symmetry_matrix_1, symmetry_matrix_2, symmetry_matrix_3
            ) = get_symmetry_matrix()
            self.register_buffer("symmetry_matrix_1", symmetry_matrix_1)
            self.register_buffer("symmetry_matrix_2", symmetry_matrix_2)
            self.register_buffer("symmetry_matrix_3", symmetry_matrix_3)

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

        if loss_sem_seg_weight is None:
            self.loss_sem_seg_weight = loss_sem_seg_weight
        else:
            assert len(loss_sem_seg_weight) == num_classes
            self.register_buffer(
                "loss_sem_seg_weight",
                torch.as_tensor(loss_sem_seg_weight, dtype=torch.float32),
                persistent=False,
            )
        self.use_focal_loss = use_focal_loss
        self.use_dice_loss = use_dice_loss

        self.cluster_proposals_start_at = min(
            self.score_net_start_at, self.npcs_net_start_at
        )

        self.val_score_threshold = val_score_threshold
        self.val_min_num_points_per_proposal = val_min_num_points_per_proposal
        self.val_nms_iou_threshold = val_nms_iou_threshold
        self.val_ap_iou_threshold = val_ap_iou_threshold

    def forward_backbone(
        self,
        voxel_tensor: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
    ) -> Tuple[spconv.SparseConvTensor, torch.Tensor, torch.Tensor]:
        # import pdb
        # pdb.set_trace()
        voxel_features, global_feats = self.unet(voxel_tensor)
        # exit(123)
        # import pdb
        # pdb.set_trace()
        # sem_logits = self.sem_seg_head(voxel_features.features)

        pt_features = voxel_features.features[pc_voxel_id]
        # sem_logits = sem_logits[pc_voxel_id]

        return voxel_features, pt_features, global_feats, 

    def _training_or_validation_step(
        self,
        point_clouds: List[PointCloud],
        mask_list: List,
        batch_idx: int,
        running_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # print(running_mode, batch_idx)
        batch_size = len(point_clouds)
        (
            scene_ids, num_points, points, batch_indices, sem_labels, instance_labels, gt_npcs,
            num_instances, gt_affordance, gt_affordance_pose, gt_forward_axis, affordance_mask,
            part_level_scores, instance_regions, num_points_per_instance, instance_sem_labels,
            voxel_tensor, pc_voxel_id,
        ) = PointCloud.collate(point_clouds)
        pt_xyz = points[:, :3]
        assert (pc_voxel_id >= 0).all()

        # semantic segmentation
        voxel_features, pt_features, global_feats = self.forward_backbone(
            voxel_tensor, pc_voxel_id
        )
        # voxel_features.c
        res = torch.zeros((batch_size, global_feats._features.shape[-1]), device = global_feats._features.device)

        # import pdb
        # pdb.set_trace()
        for i in range(batch_size):
            res[i] = global_feats._features[global_feats.indices[:,0] == i].max(0)[0]

        return res

    def validation_step(self, point_clouds: List[PointCloud], mask_list: List = None, batch_idx: int = 0, dataloader_idx: int = 0):
        split = ["val", "intra", "inter"]
        global_features = self._training_or_validation_step(
            point_clouds, mask_list, batch_idx, split[dataloader_idx]
        )

        return global_features

    def validation_epoch_end(self, validation_step_outputs_list):
        import pdb
        pdb.set_trace()

    def forward(self, point_clouds: List[PointCloud], mask_list: List = None):
        return self.validation_step(point_clouds, mask_list, 0)

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
