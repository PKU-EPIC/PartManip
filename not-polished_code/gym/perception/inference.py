from calendar import c
import importlib
from typing import List

import numpy as np

from perception.structures.point_cloud import PointCloud
from perception.datasets.gapartnet_new import apply_voxelization
from perception.utils.pose_fitting import estimate_pose_from_npcs

from envs.utils.get_running_bbox import _draw_bbox_tensor, get_bbox_pt
from envs.utils.misc import _draw_line
# from scipy.spatial.transform import Rotation as R
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
     quat_mul, tensor_clamp
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert


class Perception:
    def __init__(
        self,
        cfg
    ):
        self.cfg = cfg
        self.perception_model = self._load_perception_model(self.cfg["perception_model"])
    
    def _load_perception_model(self, perception_model_cfg):
        class_path = perception_model_cfg["class_path"]
        ckpt_pth = perception_model_cfg["ckpt_pth"]
        device = perception_model_cfg["device"]

        module_name = ".".join(class_path.split(".")[:-1])
        class_name = class_path.split(".")[-1]

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        net = cls.load_from_checkpoint(ckpt_pth)

        net.cluster_proposals_start_at = 0
        net.score_net_start_at = 0
        net.npcs_net_start_at = 0
        net.freeze()
        net.eval()
        net.to(device)

        return net


    def _inference_perception_model(self, points_list: List[torch.Tensor]):
        device = self.perception_model.device

        pcs = []
        for points in points_list:
            pc = PointCloud(
                scene_id=["eval"],
                points=points,
            )
            pc = apply_voxelization(
                pc,  voxel_size=(1. / 100, 1. / 100, 1. / 100)
            )
            pc = pc.to(device=device)
            pcs.append(pc)

        with torch.no_grad():
            scene_ids, segmentations, proposals = self.perception_model(pcs)

        sem_preds = segmentations.sem_preds

        pt_xyz = proposals.pt_xyz
        batch_indices = proposals.batch_indices
        proposal_offsets = proposals.proposal_offsets
        num_points_per_proposal = proposals.num_points_per_proposal
        num_proposals = num_points_per_proposal.shape[0]
        npcs_preds = proposals.npcs_preds
        score_preds= proposals.score_preds

        # indices = torch.arange(sem_preds.shape[0], dtype=torch.int64, device=device)
        # propsoal_indices = indices[proposals.valid_mask][proposals.sorted_indices]

        bboxes = [[] for _ in range(len(points_list))]
        for i in range(num_proposals):
            offset_begin = proposal_offsets[i].item()
            offset_end = proposal_offsets[i + 1].item()

            batch_idx = batch_indices[offset_begin]
            xyz_i = pt_xyz[offset_begin:offset_end]
            npcs_i = npcs_preds[offset_begin:offset_end]

            npcs_i = npcs_i - 0.5

            bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                estimate_pose_from_npcs(xyz_i.numpy(), npcs_i.numpy())

            bboxes[batch_idx].append(bbox_xyz.tolist())

        return bboxes


    def _inference_perception_model_batch(self, points: torch.Tensor):
        device = self.perception_model.device

        pcs = []
        for points in points_list:
            pc = PointCloud(
                scene_id=["eval"],
                points=points,
            )
            pc = apply_voxelization(
                pc,  voxel_size=(1. / 100, 1. / 100, 1. / 100)
            )
            pc = pc.to(device=device)
            pcs.append(pc)

        with torch.no_grad():
            scene_ids, segmentations, proposals = self.perception_model(pcs)

        sem_preds = segmentations.sem_preds

        pt_xyz = proposals.pt_xyz
        batch_indices = proposals.batch_indices
        proposal_offsets = proposals.proposal_offsets
        num_points_per_proposal = proposals.num_points_per_proposal
        num_proposals = num_points_per_proposal.shape[0]
        npcs_preds = proposals.npcs_preds
        score_preds= proposals.score_preds

        # indices = torch.arange(sem_preds.shape[0], dtype=torch.int64, device=device)
        # propsoal_indices = indices[proposals.valid_mask][proposals.sorted_indices]

        bboxes = [[] for _ in range(len(points_list))]
        for i in range(num_proposals):
            offset_begin = proposal_offsets[i].item()
            offset_end = proposal_offsets[i + 1].item()

            batch_idx = batch_indices[offset_begin]
            xyz_i = pt_xyz[offset_begin:offset_end]
            npcs_i = npcs_preds[offset_begin:offset_end]

            npcs_i = npcs_i - 0.5

            bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                estimate_pose_from_npcs(xyz_i.numpy(), npcs_i.numpy())

            bboxes[batch_idx].append(bbox_xyz.tolist())

        return bboxes