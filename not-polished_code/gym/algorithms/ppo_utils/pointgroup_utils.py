from typing import List, Tuple

import torch
from epic_ops.ball_query import ball_query
from epic_ops.ccl import connected_components_labeling
from epic_ops.nms import nms
from epic_ops.reduce import segmented_reduce
from epic_ops.voxelize import voxelize

from perception.structures.instances import Instances


@torch.jit.script
def compute_npcs_loss(
    npcs_preds: torch.Tensor,
    gt_npcs: torch.Tensor,
    proposal_indices: torch.Tensor,
    symmetry_matrix: torch.Tensor,
) -> torch.Tensor:
    _, num_points_per_proposal = torch.unique_consecutive(
        proposal_indices, return_counts=True
    )

    # gt_npcs: n, 3 -> n, 1, 1, 3
    # symmetry_matrix: n, m, 3, 3
    gt_npcs = gt_npcs[:, None, None, :] @ symmetry_matrix
    # n, m, 1, 3 -> n, m, 3
    gt_npcs = gt_npcs.squeeze(2)

    # npcs_preds: n, 3 -> n, 1, 3
    dist2 = (npcs_preds[:, None, :] - gt_npcs - 0.5) ** 2
    # n, m, 3 -> n, m
    dist2 = dist2.sum(dim=-1)

    loss = torch.where(
        dist2 <= 0.01,
        5 * dist2, torch.sqrt(dist2) - 0.05,
    )
    loss = torch.segment_reduce(
        loss, "mean", lengths=num_points_per_proposal
    )
    loss, _ = loss.min(dim=-1)
    return loss.mean()


@torch.jit.script
def segmented_voxelize(
    pt_xyz: torch.Tensor,
    pt_features: torch.Tensor,
    segment_offsets: torch.Tensor,
    segment_indices: torch.Tensor,
    num_points_per_segment: torch.Tensor,
    score_fullscale: float,
    score_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_offsets_begin = segment_offsets[:-1]
    segment_offsets_end = segment_offsets[1:]

    segment_coords_mean = segmented_reduce(
        pt_xyz, segment_offsets_begin, segment_offsets_end, mode="sum"
    ) / num_points_per_segment[:, None]

    centered_points = pt_xyz - segment_coords_mean[segment_indices]

    segment_coords_min = segmented_reduce(
        centered_points, segment_offsets_begin, segment_offsets_end, mode="min"
    )
    segment_coords_max = segmented_reduce(
        centered_points, segment_offsets_begin, segment_offsets_end, mode="max"
    )

    score_fullscale = 28.
    score_scale = 50.
    segment_scales = 1. / (
        (segment_coords_max - segment_coords_min) / score_fullscale
    ).max(-1)[0] - 0.01
    segment_scales = torch.clamp(segment_scales, min=None, max=score_scale)

    min_xyz = segment_coords_min * segment_scales[..., None]
    max_xyz = segment_coords_max * segment_scales[..., None]

    segment_scales = segment_scales[segment_indices]
    scaled_points = centered_points * segment_scales[..., None]

    range_xyz = max_xyz - min_xyz
    offsets = -min_xyz + torch.clamp(
        score_fullscale - range_xyz - 0.001, min=0
    ) * torch.rand(3, dtype=min_xyz.dtype, device=min_xyz.device) + torch.clamp(
        score_fullscale - range_xyz + 0.001, max=0
    ) * torch.rand(3, dtype=min_xyz.dtype, device=min_xyz.device)
    scaled_points += offsets[segment_indices]

    voxel_features, voxel_coords, voxel_batch_indices, pc_voxel_id = voxelize(
        scaled_points,
        pt_features,
        batch_offsets=segment_offsets.long(),
        voxel_size=torch.as_tensor([1., 1., 1.]),
        points_range_min=torch.as_tensor([0., 0., 0.]),
        points_range_max=torch.as_tensor([score_fullscale, score_fullscale, score_fullscale]),
        reduction="mean",
    )
    voxel_coords = torch.cat([voxel_batch_indices[:, None], voxel_coords], dim=1)

    return voxel_features, voxel_coords, pc_voxel_id


@torch.jit.script
def cluster_proposals(
    pt_xyz: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_offsets: torch.Tensor,
    sem_preds: torch.Tensor,
    ball_query_radius: float,
    max_num_points_per_query: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = pt_xyz.device
    index_dtype = batch_indices.dtype

    clustered_indices, num_points_per_query = ball_query(
        pt_xyz,
        pt_xyz,
        batch_indices,
        batch_offsets,
        ball_query_radius,
        max_num_points_per_query,
        point_labels=sem_preds,
        query_labels=sem_preds,
    )

    ccl_indices_begin = torch.arange(
        pt_xyz.shape[0], dtype=index_dtype, device=device
    ) * max_num_points_per_query
    ccl_indices_end = ccl_indices_begin + num_points_per_query
    ccl_indices = torch.stack([ccl_indices_begin, ccl_indices_end], dim=1)
    cc_labels = connected_components_labeling(
        ccl_indices.view(-1), clustered_indices.view(-1), compacted=False
    )

    sorted_cc_labels, sorted_indices = torch.sort(cc_labels)
    return sorted_cc_labels, sorted_indices


@torch.jit.script
def get_gt_scores(
    ious: torch.Tensor, fg_thresh: float = 0.75, bg_thresh: float = 0.25
) -> torch.Tensor:
    fg_mask = ious > fg_thresh
    bg_mask = ious < bg_thresh
    intermidiate_mask = ~(fg_mask | bg_mask)

    gt_scores = fg_mask.float()
    k = 1 / (fg_thresh - bg_thresh)
    b = bg_thresh / (bg_thresh - fg_thresh)
    gt_scores[intermidiate_mask] = ious[intermidiate_mask] * k + b

    return gt_scores


def filter_invalid_proposals(
    proposals: Instances,
    score_threshold: float,
    min_num_points_per_proposal: int,
) -> Instances:
    score_preds = proposals.score_preds
    proposal_indices = proposals.proposal_indices
    num_points_per_proposal = proposals.num_points_per_proposal

    valid_proposals_mask = (
        score_preds > score_threshold
    ) & (num_points_per_proposal > min_num_points_per_proposal)
    valid_points_mask = valid_proposals_mask[proposal_indices]

    proposal_indices = proposal_indices[valid_points_mask]
    _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
        proposal_indices, return_inverse=True, return_counts=True
    )
    num_proposals = num_points_per_proposal.shape[0]

    proposal_offsets = torch.zeros(
        num_proposals + 1, dtype=torch.int32, device=proposal_indices.device
    )
    proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

    if proposals.npcs_valid_mask is not None:
        valid_npcs_mask = valid_points_mask[proposals.npcs_valid_mask]
    else:
        valid_npcs_mask = valid_points_mask

    return Instances(
        valid_mask=proposals.valid_mask,
        sorted_indices=proposals.sorted_indices[valid_points_mask],
        pt_xyz=proposals.pt_xyz[valid_points_mask],
        batch_indices=proposals.batch_indices[valid_points_mask],
        proposal_offsets=proposal_offsets,
        proposal_indices=proposal_indices,
        num_points_per_proposal=num_points_per_proposal,
        sem_preds=proposals.sem_preds[valid_points_mask],
        score_preds=proposals.score_preds[valid_proposals_mask],
        npcs_preds=proposals.npcs_preds[
            valid_npcs_mask
        ] if proposals.npcs_preds is not None else None,
        sem_labels=proposals.sem_labels[
            valid_points_mask
        ] if proposals.sem_labels is not None else None,
        instance_labels=proposals.instance_labels[
            valid_points_mask
        ] if proposals.instance_labels is not None else None,
        instance_sem_labels=proposals.instance_sem_labels,
        num_points_per_instance=proposals.num_points_per_instance,
        gt_npcs=proposals.gt_npcs[
            valid_npcs_mask
        ] if proposals.gt_npcs is not None else None,
        npcs_valid_mask=proposals.npcs_valid_mask[valid_points_mask] \
            if proposals.npcs_valid_mask is not None else None,
        ious=proposals.ious[
            valid_proposals_mask
        ] if proposals.ious is not None else None,
        affordance_preds=proposals.affordance_preds[
            valid_points_mask
        ] if proposals.affordance_preds is not None else None,
        affordance_pose_preds=proposals.affordance_pose_preds[
            valid_points_mask
        ] if proposals.affordance_pose_preds is not None else None,
        forward_axis_preds=proposals.forward_axis_preds[
            valid_points_mask
        ] if proposals.forward_axis_preds is not None else None,
        gt_affordance=proposals.gt_affordance[
            valid_points_mask
        ] if proposals.gt_affordance is not None else None,
        gt_affordance_pose=proposals.gt_affordance_pose[
            valid_points_mask
        ] if proposals.gt_affordance_pose is not None else None,
        gt_forward_axis=proposals.gt_forward_axis[
            valid_points_mask
        ] if proposals.gt_forward_axis is not None else None,
        affordance_mask=proposals.affordance_mask[
            valid_points_mask
        ] if proposals.affordance_mask is not None else None,
    )


def apply_nms(
    proposals: Instances,
    iou_threshold: float = 0.3,
):
    score_preds = proposals.score_preds
    sorted_indices = proposals.sorted_indices
    proposal_offsets = proposals.proposal_offsets
    proposal_indices = proposals.proposal_indices
    num_points_per_proposal = proposals.num_points_per_proposal

    values = torch.ones(
        sorted_indices.shape[0], dtype=torch.float32, device=sorted_indices.device
    )
    csr = torch.sparse_csr_tensor(
        proposal_offsets.int(), sorted_indices.int(), values,
        dtype=torch.float32, device=sorted_indices.device,
    )
    intersection = csr @ csr.t()
    intersection = intersection.to_dense()
    union = num_points_per_proposal[:, None] + num_points_per_proposal[None, :]
    union = union - intersection

    ious = intersection / (union + 1e-8)
    keep = nms(ious.cuda(), score_preds.cuda(), iou_threshold)
    keep = keep.to(score_preds.device)

    valid_proposals_mask = torch.zeros(
        ious.shape[0], dtype=torch.bool, device=score_preds.device
    )
    valid_proposals_mask[keep] = True
    valid_points_mask = valid_proposals_mask[proposal_indices]

    proposal_indices = proposal_indices[valid_points_mask]
    _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
        proposal_indices, return_inverse=True, return_counts=True
    )
    num_proposals = num_points_per_proposal.shape[0]

    proposal_offsets = torch.zeros(
        num_proposals + 1, dtype=torch.int32, device=proposal_indices.device
    )
    proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

    if proposals.npcs_valid_mask is not None:
        valid_npcs_mask = valid_points_mask[proposals.npcs_valid_mask]
    else:
        valid_npcs_mask = valid_points_mask

    return Instances(
        valid_mask=proposals.valid_mask,
        sorted_indices=proposals.sorted_indices[valid_points_mask],
        pt_xyz=proposals.pt_xyz[valid_points_mask],
        batch_indices=proposals.batch_indices[valid_points_mask],
        proposal_offsets=proposal_offsets,
        proposal_indices=proposal_indices,
        num_points_per_proposal=num_points_per_proposal,
        sem_preds=proposals.sem_preds[valid_points_mask],
        score_preds=proposals.score_preds[valid_proposals_mask],
        npcs_preds=proposals.npcs_preds[
            valid_npcs_mask
        ] if proposals.npcs_preds is not None else None,
        sem_labels=proposals.sem_labels[
            valid_points_mask
        ] if proposals.sem_labels is not None else None,
        instance_labels=proposals.instance_labels[
            valid_points_mask
        ] if proposals.instance_labels is not None else None,
        instance_sem_labels=proposals.instance_sem_labels,
        num_points_per_instance=proposals.num_points_per_instance,
        gt_npcs=proposals.gt_npcs[
            valid_npcs_mask
        ] if proposals.gt_npcs is not None else None,
        npcs_valid_mask=proposals.npcs_valid_mask[valid_points_mask] \
            if proposals.npcs_valid_mask is not None else None,
        ious=proposals.ious[
            valid_proposals_mask
        ] if proposals.ious is not None else None,
        affordance_preds=proposals.affordance_preds[
            valid_points_mask
        ] if proposals.affordance_preds is not None else None,
        affordance_pose_preds=proposals.affordance_pose_preds[
            valid_points_mask
        ] if proposals.affordance_pose_preds is not None else None,
        forward_axis_preds=proposals.forward_axis_preds[
            valid_points_mask
        ] if proposals.forward_axis_preds is not None else None,
        gt_affordance=proposals.gt_affordance[
            valid_points_mask
        ] if proposals.gt_affordance is not None else None,
        gt_affordance_pose=proposals.gt_affordance_pose[
            valid_points_mask
        ] if proposals.gt_affordance_pose is not None else None,
        gt_forward_axis=proposals.gt_forward_axis[
            valid_points_mask
        ] if proposals.gt_forward_axis is not None else None,
        affordance_mask=proposals.affordance_mask[
            valid_points_mask
        ] if proposals.affordance_mask is not None else None,
    )


@torch.jit.script
def voc_ap(
    rec: torch.Tensor,
    prec: torch.Tensor,
    use_07_metric: bool = False,
) -> float:
    if use_07_metric:
        # 11 point metric
        ap = torch.as_tensor(0, dtype=prec.dtype, device=prec.device)
        for t in range(0, 11, 1):
            t /= 10.0
            if torch.sum(rec >= t) == 0:
                p = torch.as_tensor(0, dtype=prec.dtype, device=prec.device)
            else:
                p = torch.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = torch.cat([
            torch.as_tensor([0.0], dtype=rec.dtype, device=rec.device),
            rec,
            torch.as_tensor([1.0], dtype=rec.dtype, device=rec.device),
        ], dim=0)
        mpre = torch.cat([
            torch.as_tensor([0.0], dtype=prec.dtype, device=prec.device),
            prec,
            torch.as_tensor([0.0], dtype=prec.dtype, device=prec.device),
        ], dim=0)

        # compute the precision envelope
        for i in range(mpre.shape[0] - 1, 0, -1):
            mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = torch.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap.item())


@torch.jit.script
def _compute_ap_per_class(
    tp: torch.Tensor, fp: torch.Tensor, num_gt_instances: int
) -> float:
    if tp.shape[0] == 0:
        return 0.

    tp = tp.cumsum(0)
    fp = fp.cumsum(0)
    rec = tp / num_gt_instances
    prec = tp / (tp + fp + 1e-8)

    return voc_ap(rec, prec)


@torch.jit.script
def _compute_ap(
    confidence: torch.Tensor,
    classes: torch.Tensor,
    sorted_indices: torch.Tensor,
    batch_indices: torch.Tensor,
    sample_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    matched: List[torch.Tensor],
    instance_sem_labels: List[torch.Tensor],
    ious: List[torch.Tensor],
    num_classes: int,
    iou_threshold: float,
):
    sorted_indices_cpu = sorted_indices.cpu()

    num_proposals = confidence.shape[0]
    tp = torch.zeros(num_proposals, dtype=torch.float32)
    fp = torch.zeros(num_proposals, dtype=torch.float32)
    for i in range(num_proposals):
        idx = sorted_indices_cpu[i]

        class_idx = classes[idx]
        batch_idx = batch_indices[idx].item()
        sample_idx = sample_indices[idx]
        proposal_idx = proposal_indices[idx]

        instance_sem_labels_i = instance_sem_labels[batch_idx][sample_idx]
        invalid_instance_mask = instance_sem_labels_i != class_idx

        ious_i = ious[batch_idx][proposal_idx].clone()
        ious_i[invalid_instance_mask] = 0.
        if ious_i.shape[0] == 0:
            max_iou, max_idx = 0., 0
        else:
            max_iou, max_idx = ious_i.max(0)
            max_iou, max_idx = max_iou.item(), int(max_idx.item())

        if max_iou > iou_threshold:
            if not matched[batch_idx][sample_idx, max_idx].item():
                tp[i] = 1.0
                matched[batch_idx][sample_idx, max_idx] = True
            else:
                fp[i] = 1.0
        else:
            fp[i] = 1.0

    tp = tp.to(device=confidence.device)
    fp = fp.to(device=confidence.device)

    sorted_classes = classes[sorted_indices]
    gt_classes = torch.cat([x.view(-1) for x in instance_sem_labels], dim=0)
    aps: List[float] = []
    for c in range(1, num_classes):
        num_gt_instances = (gt_classes == c).sum()
        mask = sorted_classes == c
        ap = _compute_ap_per_class(tp[mask], fp[mask], num_gt_instances)
        aps.append(ap)
    return aps


def compute_ap(
    proposals: List[Instances],
    num_classes: int = 9,
    iou_threshold: float = 0.5,
    device="cpu",
):
    confidence = torch.cat([
        p.score_preds for p in proposals
    ], dim=0).to(device=device)
    classes = torch.cat([
        p.sem_preds[p.proposal_offsets[:-1].long()]
        for p in proposals
    ], dim=0).to(device=device)
    sorted_indices = torch.argsort(confidence, descending=True)

    batch_indices = torch.cat([
        torch.full((p.score_preds.shape[0],), i, dtype=torch.int64)
        for i, p in enumerate(proposals)
    ], dim=0)
    sample_indices = torch.cat([
        p.batch_indices[p.proposal_offsets[:-1].long()].long()
        for p in proposals
    ], dim=0).cpu()
    proposal_indices = torch.cat([
        torch.arange(p.score_preds.shape[0], dtype=torch.int64)
        for p in proposals
    ], dim=0)

    matched = [
        torch.zeros_like(p.instance_sem_labels, dtype=torch.bool, device="cpu")
        for p in proposals
    ]

    return _compute_ap(
        confidence,
        classes,
        sorted_indices,
        batch_indices,
        sample_indices,
        proposal_indices,
        matched,
        [p.instance_sem_labels.to(device=device) for p in proposals],
        [p.ious.to(device=device) for p in proposals],
        num_classes,
        iou_threshold,
    )


import numpy as np


def estimate_similarity_umeyama(source_hom: np.ndarray, target_hom: np.ndarray):
    num_points = source_hom.shape[1]

    source_centroid = np.mean(source_hom[:3, :], axis=1)
    target_centroid = np.mean(target_hom[:3, :], axis=1)

    centered_source = source_hom[:3, :] - np.tile(source_centroid, (num_points, 1)).transpose()
    centered_target = target_hom[:3, :] - np.tile(target_centroid, (num_points, 1)).transpose()

    cov = np.matmul(centered_target, np.transpose(centered_source)) / num_points

    if np.isnan(cov).any():
        raise RuntimeError("There are NANs in the input.")

    U, D, Vh = np.linalg.svd(cov, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    var_P = np.var(source_hom[:3, :], axis=1).sum()
    scale_factor = 1 / var_P * np.sum(D)
    scale = np.array([scale_factor, scale_factor, scale_factor])
    scale_matrix = np.diag(scale)

    rotation = np.matmul(U, Vh).T

    translation = target_hom[:3, :].mean(axis=1) - source_hom[:3, :].mean(axis=1).dot(
        scale_factor * rotation
    )

    out_transform = np.identity(4)
    out_transform[:3, :3] = scale_matrix @ rotation
    out_transform[:3, 3] = translation

    return scale, rotation, translation, out_transform


def evaluate_model(
    out_transform: np.ndarray, source_hom: np.ndarray, target_hom: np.ndarray, pass_thrsh: float
):
    diff = target_hom - np.matmul(out_transform, source_hom)
    residual_vec = np.linalg.norm(diff[:3, :], axis=0)
    residual = np.linalg.norm(residual_vec)
    inlier_idx = np.where(residual_vec < pass_thrsh)
    num_inliers = np.count_nonzero(inlier_idx)
    inlier_ratio = num_inliers / source_hom.shape[1]
    return residual, inlier_ratio, inlier_idx[0]


def get_RANSAC_inliers(
    source_hom: np.ndarray, target_hom: np.ndarray,
    max_iters: int, pass_thrsh: float, stop_thrsh: float,
):
    best_residual = 1e10
    best_inlier_ratio = 0
    best_inlier_idx = np.arange(source_hom.shape[1])

    for i in range(max_iters):
        # Pick 5 random (but corresponding) points from source and target
        rand_idx = np.random.randint(source_hom.shape[1], size=5)
        _, _, _, out_transform = estimate_similarity_umeyama(
            source_hom[:, rand_idx], target_hom[:, rand_idx]
        )

        residual, inlier_ratio, inlier_idx = evaluate_model(
            out_transform, source_hom, target_hom, pass_thrsh
        )
        if residual < best_residual:
            best_residual = residual
            best_inlier_ratio = inlier_ratio
            best_inlier_idx = inlier_idx

        if best_residual < stop_thrsh:
            break

    return best_inlier_ratio, best_inlier_idx


def estimate_similarity_transform(
    source: np.ndarray, target: np.ndarray,
    stop_thrsh: float = 0.5,
    max_iters: int = 100,
):
    if source.shape[0] == 1:
        source = np.repeat(source, 2, axis=0)
        target = np.repeat(target, 2, axis=0)

    source_hom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    target_hom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))

    # Auto-parameter selection based on source-target heuristics
    source_norm = np.mean(np.linalg.norm(source, axis=1))
    target_norm = np.mean(np.linalg.norm(target, axis=1))

    ratio_st = (source_norm / target_norm)
    ratio_ts = (target_norm / source_norm)
    pass_thrsh = ratio_st if ratio_st > ratio_ts else ratio_ts

    best_inlier_ratio, best_inlier_idx = \
        get_RANSAC_inliers(
            source_hom, target_hom, max_iters=max_iters,
            pass_thrsh=pass_thrsh, stop_thrsh=stop_thrsh,
        )
    source_inliers_hom = source_hom[:, best_inlier_idx]
    target_inliers_hom = target_hom[:, best_inlier_idx]

    if best_inlier_ratio < 0.01:
        return None, np.asarray([None, None, None]), None, None, None

    scale, rotation, translation, out_transform = estimate_similarity_umeyama(
        source_inliers_hom, target_inliers_hom
    )

    return scale, rotation, translation, out_transform, best_inlier_idx


def estimate_pose_from_npcs(xyz, npcs):
    scale, rotation, translation, out_transform, best_inlier_idx = \
        estimate_similarity_transform(npcs, xyz)

    rotation_inv = np.linalg.pinv(rotation)
    trans_seg = np.dot((xyz - translation), rotation_inv) / scale[0]
    npcs_max = abs(trans_seg[best_inlier_idx]).max(0)

    bbox_raw = np.asarray([
        [-npcs_max[0], -npcs_max[1], -npcs_max[2]],
        [npcs_max[0], -npcs_max[1], -npcs_max[2]],
        [-npcs_max[0], npcs_max[1], -npcs_max[2]],
        [-npcs_max[0], -npcs_max[1], npcs_max[2]],
        [npcs_max[0], npcs_max[1], -npcs_max[2]],
        [npcs_max[0], -npcs_max[1], npcs_max[2]],
        [-npcs_max[0], npcs_max[1], npcs_max[2]],
        [npcs_max[0], npcs_max[1], npcs_max[2]],
    ])
    bbox_trans = np.dot((bbox_raw * scale[0]), rotation) + translation

    return bbox_trans, scale, rotation, translation, out_transform, best_inlier_idx
