from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class Observations:
    state: torch.Tensor
    obs:torch.Tensor
    points: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    imgs: Optional[torch.Tensor] = None
    