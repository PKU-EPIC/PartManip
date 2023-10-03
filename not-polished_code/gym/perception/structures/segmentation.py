from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Segmentation:
    batch_size: int
    num_points: List[int]

    sem_preds: torch.Tensor
    sem_labels: Optional[torch.Tensor] = None

    affordance_preds: Optional[torch.Tensor] = None
    affordance_pose_preds: Optional[torch.Tensor] = None
    affordance_pose1_preds: Optional[torch.Tensor] = None
    affordance_pose2_preds: Optional[torch.Tensor] = None
    forward_axis_preds: Optional[torch.Tensor] = None
    gt_affordance: Optional[torch.Tensor] = None
    gt_affordance_pose: Optional[torch.Tensor] = None
    gt_forward_axis: Optional[torch.Tensor] = None
    affordance_mask: Optional[torch.Tensor] = None
