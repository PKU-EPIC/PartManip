from dataclasses import dataclass

import torch


@dataclass
class PartPC:
    scene_id: str

    points: torch.Tensor
    rgb: torch.Tensor
    npcs: torch.Tensor
    affordance: torch.Tensor
