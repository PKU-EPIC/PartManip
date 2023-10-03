from pathlib import Path
from PIL import Image
import argparse
import json

from tqdm import tqdm
import numpy as np
import open3d as o3d
import open3d.ml.torch as ml3d
import torch
from torch_cluster import fps, knn


@torch.jit.script
def deproject_depth_images(
    depth_images: torch.Tensor,
    proj_matrices: torch.Tensor,
    inv_view_matrices: torch.Tensor
):
    bs, h, w = depth_images.shape
    dtype, device = depth_images.dtype, depth_images.device

    v, u = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device),
        indexing="ij"
    )
    u = u[None].expand(bs, h, w)
    v = v[None].expand(bs, h, w)
    d = depth_images

    fu = 2 / proj_matrices[:, 0, 0]
    fv = 2 / proj_matrices[:, 1, 1]

    x = (w / 2 - u) / w * d * fu[:, None, None]
    y = (v - h / 2) / h * d * fv[:, None, None]

    xyz_cam = torch.stack([x, y, d, torch.ones_like(x)], dim=-1)
    points = xyz_cam[:, :, :, None, :] @ inv_view_matrices[:, None, None, :, :]
    points = points.squeeze(3)
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")

    root_dir = Path("datasets/GAPartNetRaw/pla2/isaac_tmp")

    file_paths = []
    for file_path in sorted(root_dir.glob("*_00_*.pth")):
        file_paths.append(file_path)

    local_file_paths = []
    for i in range(args.local_rank, len(file_paths), args.world_size):
        local_file_paths.append(file_paths[i])

    for file_path in tqdm(local_file_paths):
        render_id = int(file_path.stem.split("_")[-1])

        data = torch.load(str(file_path))
        cam_rgb_tensor = data["cam_rgb_tensor"]
        cam_seg_tensor = data["cam_seg_tensor"]
        cam_depth_tensor = data["cam_depth_tensor"].to(device=device)
        cam_flow_tensor = data["cam_flow_tensor"]
        proj_matrix = data["proj_matrix"].to(device=device)
        view_matrix = data["view_matrix"].to(device=device)
        inv_view_matrix = torch.linalg.inv(view_matrix)

        im = Image.fromarray(cam_rgb_tensor.numpy(), "RGBA")
        im.save(f"datasets/GAPartNetRaw/pla2/images/{file_path.stem}.png")

        points = deproject_depth_images(
            cam_depth_tensor[None],
            proj_matrix[None], inv_view_matrix[None],
        )
        points = points[0]
        points = points[cam_seg_tensor > 0]
        points = points[:, :3] / points[:, 3:4]

        cam_rgb_tensor = cam_rgb_tensor.to(device=device)
        rgb = cam_rgb_tensor[cam_seg_tensor > 0]

        segmentation = np.load(
            f"datasets/GAPartNetRaw/annotation/segmentation/{file_path.stem}.npz"
        )
        semantic_segmentation = segmentation["semantic_segmentation"]
        instance_segmentation = segmentation["instance_segmentation"]
        sem_labels = torch.from_numpy(semantic_segmentation).to(dtype=torch.int64, device=device)
        instance_labels = torch.from_numpy(instance_segmentation).to(dtype=torch.int64, device=device)
        valid_mask_raw = sem_labels >= 0
        sem_labels = sem_labels[valid_mask_raw]
        instance_labels = instance_labels[valid_mask_raw]

        npcs_map = np.load(
            f"datasets/GAPartNetRaw/annotation/npcs/{file_path.stem}.npz"
        )["npcs_map"]
        npcs_map = torch.from_numpy(npcs_map).to(device=device)
        npcs_map = npcs_map[valid_mask_raw]

        depth_raw = np.load(
            f"datasets/GAPartNetRaw/depth/{file_path.stem}.npz"
        )["depth_map"]
        depth_raw = torch.from_numpy(depth_raw).to(device=device)
        depth_raw = -depth_raw
        depth_raw[~valid_mask_raw] = -1000

        points_raw = deproject_depth_images(
            depth_raw[None],
            proj_matrix[None], inv_view_matrix[None],
        )
        points_raw = points_raw[0]
        points_raw = points_raw[valid_mask_raw]
        points_raw = points_raw[:, :3] / points_raw[:, 3:4]

        neighbors_index = knn(points_raw, points, k=1)[1]

        sem_labels = sem_labels[neighbors_index]
        instance_labels = instance_labels[neighbors_index]
        npcs_map = npcs_map[neighbors_index]

        with open(
            f"datasets/GAPartNetRaw/metafile/{file_path.stem}.json"
        ) as f:
            data = json.load(f)
            camera_position = data["camera_position"]

        points_numpy = points.cpu().numpy()
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_numpy)
        pc.estimate_normals()
        pc.orient_normals_towards_camera_location(np.asarray(camera_position))
        normals = np.asarray(pc.normals)
        normals = torch.from_numpy(normals).to(dtype=torch.float32)

        batch_indices = torch.zeros(points.shape[0], dtype=torch.int64, device=device)
        sampled_indices = fps(points, batch_indices, ratio=20000. / points.shape[0], random_start=False)
        sampled_indices = sampled_indices
        sampled_indices_cpu = sampled_indices.cpu()

        points = points[sampled_indices].cpu()
        rgb = rgb[sampled_indices].cpu()
        normals = normals[sampled_indices_cpu]
        sem_labels = sem_labels[sampled_indices].cpu()
        instance_labels = instance_labels[sampled_indices].cpu()
        npcs_map = npcs_map[sampled_indices].cpu()

        row_id = render_id % 8
        col_id = render_id // 8

        offsets = torch.as_tensor([
            row_id * 6, col_id * 6, 0
        ], dtype=torch.float32)[None]

        points -= offsets

        torch.save(
            {
                "points": points,
                "rgb": rgb,
                "normals": normals,
                "sem_labels": sem_labels,
                "instance_labels": instance_labels,
                "npcs_map": npcs_map,
            },
            f"datasets/GAPartNetRaw/pla2/pc/{file_path.stem}.pth"
        )


if __name__ == "__main__":
    main()