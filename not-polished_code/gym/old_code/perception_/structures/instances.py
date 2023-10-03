from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Instances:
    valid_mask: torch.Tensor # [80000]
    sorted_indices: torch.Tensor # [102044]
    pt_xyz: torch.Tensor # [102044, 3]

    batch_indices: torch.Tensor # [102044]
    proposal_offsets: torch.Tensor # [128]
    proposal_indices: torch.Tensor # [102044]
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


# Instances(
# valid_mask=tensor([False, False, False,  ...,  True, False,  True], device='cuda:2'), sorted_indices=tensor([    0,     1,     2,  ..., 44715, 48963, 50689], device='cuda:2'), pt_xyz=tensor([[-0.1079, -0.1249, -0.5972],
# [-0.1029, -0.7435, -0.0442],
# [-0.1070,  0.7303, -0.1368],
# ...,
# [-0.0919, -0.3902,  0.1197],
# [-0.0920, -0.3982,  0.1337],
# [-0.0931, -0.3880,  0.1258]], device='cuda:2'), batch_indices=tensor([0, 0, 0,  ..., 3, 3, 3], device='cuda:2', dtype=torch.int32), proposal_offsets=tensor([     0,  10954,  11092,  11095,  11238,  11272,  11275,  22218,  22351,
# 22484,  22522,  32711,  33997,  34517,  34554,  35064,  36611,  36658,
# 36699,  36735,  36760,  36763,  36818,  36826,  36829,  36835,  47117,
# 47176,  47674,  48173,  49644,  50933,  50937,  50984,  51029,  51060,
# 51082,  51115,  56258,  61921,  62044,  62047,  62130,  62265,  62274,
# 62278,  62288,  62295,  62305,  62309,  62312,  62318,  62329,  62337,
# 62340,  62343,  62349,  62354,  62357,  62360,  67450,  73155,  73276,
# 73402,  73500,  73532,  73546,  73554,  73557,  73573,  73579,  73583,
# 77098,  78895,  82623,  82729,  85484,  85507,  87563,  87610,  87623,
# 87647,  87685,  87693,  87746,  87749,  87753,  87766,  87772,  87781,
# 87786,  87791,  87796,  87799,  87802,  87806,  87812,  87815,  87821,
# 87824,  91526,  94326,  97966,  98019,  99982, 101760, 101764, 101808,
# 101830, 101846, 101889, 101892, 101978, 101985, 101991, 101997, 102003,
# 102008, 102013, 102018, 102022, 102025, 102028, 102031, 102034, 102037,
# 102041, 102044], device='cuda:2', dtype=torch.int32), proposal_indices=tensor([  0,   0,   0,  ..., 126, 126, 126], device='cuda:2'), num_points_per_proposal=tensor([10954,   138,     3,   143,    34,     3, 10943,   133,   133,    38,
# 10189,  1286,   520,    37,   510,  1547,    47,    41,    36,    25,
# 3,    55,     8,     3,     6, 10282,    59,   498,   499,  1471,
# 1289,     4,    47,    45,    31,    22,    33,  5143,  5663,   123,
# 3,    83,   135,     9,     4,    10,     7,    10,     4,     3,
# 6,    11,     8,     3,     3,     6,     5,     3,     3,  5090,
# 5705,   121,   126,    98,    32,    14,     8,     3,    16,     6,
# 4,  3515,  1797,  3728,   106,  2755,    23,  2056,    47,    13,
# 24,    38,     8,    53,     3,     4,    13,     6,     9,     5,
# 5,     5,     3,     3,     4,     6,     3,     6,     3,  3702,
# 2800,  3640,    53,  1963,  1778,     4,    44,    22,    16,    43,
# 3,    86,     7,     6,     6,     6,     5,     5,     5,     4,
# 3,     3,     3,     3,     3,     4,     3], device='cuda:2'), sem_preds=tensor([4, 4, 4,  ..., 5, 5, 5], device='cuda:2', dtype=torch.int32), score_preds=tensor([9.3249e-01, 9.9992e-01, 6.9229e-03, 9.9994e-01, 1.8997e-02, 1.1665e-04,
# 8.6874e-01, 9.9995e-01, 9.9996e-01, 5.4314e-03, 6.1730e-01, 9.8452e-01,
# 5.9566e-01, 9.8518e-01, 8.3966e-01, 8.8297e-01, 9.9829e-01, 9.9114e-01,
# 9.9126e-01, 9.6619e-01, 1.1800e-04, 9.9929e-01, 5.1141e-03, 9.4771e-02,
# 2.7368e-03, 6.6097e-01, 9.9753e-01, 7.8700e-01, 6.4902e-01, 8.7739e-01,
# 9.8372e-01, 7.6656e-04, 9.9815e-01, 9.9312e-01, 9.7858e-01, 9.2760e-01,
# 9.9625e-01, 9.9997e-01, 9.9999e-01, 9.9992e-01, 6.9229e-03, 1.3587e-02,
# 9.9995e-01, 1.1489e-03, 1.4471e-03, 4.7745e-04, 6.1044e-06, 3.1520e-04,
# 9.6889e-05, 2.4535e-05, 3.6280e-05, 1.9629e-04, 9.5551e-04, 7.0108e-06,
# 1.4070e-05, 7.5813e-06, 1.1366e-04, 1.4340e-03, 8.5150e-04, 9.9997e-01,
# 9.9999e-01, 9.9997e-01, 9.9994e-01, 5.7999e-02, 1.3111e-03, 5.0420e-04,
# 9.6944e-04, 2.5229e-05, 5.8156e-04, 1.0952e-03, 2.4667e-05, 9.9966e-01,
# 9.9716e-01, 6.7665e-01, 6.5667e-02, 9.9648e-01, 7.3809e-01, 9.9891e-01,
# 9.9738e-01, 1.6213e-03, 8.4733e-01, 9.2664e-01, 1.1331e-04, 9.9772e-01,
# 5.5737e-05, 4.0000e-04, 8.5270e-02, 1.0723e-04, 8.8937e-05, 2.8144e-03,
# 2.3432e-03, 9.2157e-04, 8.1338e-06, 1.8599e-04, 1.7258e-03, 9.3393e-04,
# 9.5721e-06, 2.7368e-03, 2.9205e-05, 9.9833e-01, 9.9980e-01, 7.9783e-01,
# 9.8833e-01, 9.9951e-01, 9.9260e-01, 7.6656e-04, 9.9800e-01, 8.7830e-01,
# 8.3166e-01, 9.4797e-01, 2.1401e-03, 5.2423e-02, 6.2765e-03, 1.1602e-03,
# 7.7810e-05, 1.1900e-03, 9.6812e-04, 2.1925e-04, 1.1119e-03, 3.7581e-04,
# 3.1147e-04, 4.9941e-04, 1.8499e-04, 4.9966e-04, 1.2742e-04, 2.1514e-04,
# 8.2492e-04], device='cuda:2'), npcs_preds=tensor([[0.7592, 0.7488, 0.4990],
# [0.1606, 0.3271, 0.4991],
# [0.3884, 0.2534, 0.4943],
# ...,
# [0.9452, 0.1854, 0.4444],
# [0.9536, 0.2001, 0.4079],
# [0.9494, 0.1678, 0.4376]], device='cuda:2'), sem_labels=None, instance_labels=None, instance_sem_labels=None, num_points_per_instance=None, gt_npcs=None, npcs_valid_mask=None, ious=None, affordance_preds=None, affordance_pose_preds=None, forward_axis_preds=None, gt_affordance=None, gt_affordance_pose=None, gt_forward_axis=None, affordance_mask=None)