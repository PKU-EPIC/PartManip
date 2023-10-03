from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Instances:
    valid_mask: torch.Tensor
    sorted_indices: torch.Tensor
    pt_xyz: torch.Tensor

    batch_indices: torch.Tensor
    proposal_offsets: torch.Tensor
    proposal_indices: torch.Tensor
    num_points_per_proposal: torch.Tensor

    sem_preds: Optional[torch.Tensor] = None
    score_preds: Optional[torch.Tensor] = None
    npcs_preds: Optional[torch.Tensor] = None

    sem_labels: Optional[torch.Tensor] = None
    instance_labels: Optional[torch.Tensor] = None
    instance_sem_labels: Optional[torch.Tensor] = None
    num_points_per_instance: Optional[torch.Tensor] = None
    gt_npcs: Optional[torch.Tensor] = None

    npcs_valid_mask: Optional[torch.Tensor] = None

    ious: Optional[torch.Tensor] = None

    affordance_preds: Optional[torch.Tensor] = None
    affordance_pose_preds: Optional[torch.Tensor] = None
    forward_axis_preds: Optional[torch.Tensor] = None
    gt_affordance: Optional[torch.Tensor] = None
    gt_affordance_pose: Optional[torch.Tensor] = None
    gt_forward_axis: Optional[torch.Tensor] = None
    affordance_mask: Optional[torch.Tensor] = None
