import copy
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchdata.datapipes as dp
from epic_ops.voxelize import voxelize
from torch.utils.data import DataLoader

from perception.structures.point_cloud import PointCloud
from perception.utils import data as data_utils


def apply_augmentations(
    pc: PointCloud,
    *,
    pos_jitter: float = 0.,
    color_jitter: float = 0.,
    flip_prob: float = 0.,
    rotate_prob: float = 0.,
) -> PointCloud:
    pc = copy.copy(pc)

    m = np.eye(3)
    if pos_jitter > 0:
        m += np.random.randn(3, 3) * pos_jitter

    if flip_prob > 0:
        if np.random.rand() < flip_prob:
            m[0, 0] = -m[0, 0]

    if rotate_prob > 0:
        if np.random.rand() < flip_prob:
            theta = np.random.rand() * np.pi * 2
            m = m @ np.asarray([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ])

    pc.points = pc.points.copy()
    pc.points[:, :3] = pc.points[:, :3] @ m

    if color_jitter > 0:
        pc.points[:, 3:] += np.random.randn(
            1, pc.points.shape[1] - 3
        ) * color_jitter

    return pc


def downsample(pc: PointCloud, *, max_points: int = 20000) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]

    if num_points > max_points:
        assert False, (num_points, max_points)
        indices = np.random.choice(num_points, max_points, replace=False)
        pc.points = pc.points[indices]
        pc.sem_labels = pc.sem_labels[indices]
        pc.instance_labels = pc.instance_labels[indices]
        if pc.gt_npcs is not None:
            pc.gt_npcs = pc.gt_npcs[indices]

    return pc


def compact_instance_labels(pc: PointCloud) -> PointCloud:
    pc = copy.copy(pc)

    valid_mask = pc.instance_labels >= 0
    instance_labels = pc.instance_labels[valid_mask]
    original_labels, instance_labels = np.unique(instance_labels, return_inverse=True)
    pc.instance_labels[valid_mask] = instance_labels

    if pc.part_level_scores_dict is not None:
        part_level_scores = []
        for original_label in original_labels.tolist():
            original_label = original_label + 1
            if original_label in pc.part_level_scores_dict:
                score = pc.part_level_scores_dict[original_label]
            else:
                score = 0
            if original_label >= 0:
                part_level_scores.append(score)
        pc.part_level_scores = np.asarray(part_level_scores, dtype=np.float32)

    return pc


def generate_inst_info(pc: PointCloud) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]

    num_instances = int(pc.instance_labels.max()) + 1
    instance_regions = np.zeros((num_points, 9), dtype=np.float32)
    num_points_per_instance = []
    instance_sem_labels = []

    for i in range(num_instances):
        indices = np.where(pc.instance_labels == i)[0]

        xyz_i = pc.points[indices, :3]
        min_i = xyz_i.min(0)
        max_i = xyz_i.max(0)
        mean_i = xyz_i.mean(0)
        instance_regions[indices, 0:3] = mean_i
        instance_regions[indices, 3:6] = min_i
        instance_regions[indices, 6:9] = max_i

        num_points_per_instance.append(indices.shape[0])
        instance_sem_labels.append(int(pc.sem_labels[indices[0]]))

    pc.num_instances = num_instances
    pc.instance_regions = instance_regions
    pc.num_points_per_instance = np.asarray(num_points_per_instance, dtype=np.int32)
    pc.instance_sem_labels = np.asarray(instance_sem_labels, dtype=np.int32)

    return pc


def apply_voxelization(
    pc: PointCloud, *, voxel_size: Tuple[float, float, float]
) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]
    pt_xyz = pc.points[:, :3]
    points_range_min = pt_xyz.min(0)[0] - 1e-4
    points_range_max = pt_xyz.max(0)[0] + 1e-4
    voxel_features, voxel_coords, _, pc_voxel_id = voxelize(
        pt_xyz, pc.points,
        batch_offsets=torch.as_tensor([0, num_points], dtype=torch.int64),
        voxel_size=voxel_size,
        points_range_min=points_range_min.tolist(),
        points_range_max=points_range_max.tolist(),
        reduction="mean",
    )
    assert (pc_voxel_id >= 0).all()

    voxel_coords_range = (voxel_coords.max(0)[0] + 1).clamp(min=128, max=None)

    pc.voxel_features = voxel_features
    pc.voxel_coords = voxel_coords
    pc.voxel_coords_range = voxel_coords_range.tolist()
    pc.pc_voxel_id = pc_voxel_id

    return pc


def load_data(file_path: str):
    pc_data = torch.load(file_path)

    scene_id = file_path.split("/")[-1].split(".")[0]

    return PointCloud(
        scene_id=scene_id,
        points=np.concatenate(
            [pc_data[0], pc_data[1]],
            axis=-1, dtype=np.float32,
        ),
        sem_labels=pc_data[2].astype(np.int64),
        instance_labels=pc_data[3].astype(np.int32),
        gt_npcs=pc_data[4].astype(np.float32),
    )


def from_folder(
    root_dir: Union[str, Path] = "",
    shuffle: bool = False,
    max_points: int = 20000,
    augmentation: bool = False,
    voxel_size: Tuple[float, float, float] = (1 / 100, 1 / 100, 1 / 100),
    few_shot = False,
    few_shot_num = 256,
):
    pipe = dp.iter.FileLister(str(root_dir))
    pipe = pipe.filter(filter_fn=lambda x: x.endswith(".pth"))

    pipe = pipe.distributed_sharding_filter()
    if shuffle:
        pipe = pipe.shuffle()
    if few_shot:
        pipe = pipe.header(few_shot_num)
        # print("??", len(pipe))
        # exit(123)

    # Load data
    pipe = pipe.map(load_data)
    # Remove empty samples
    pipe = pipe.filter(filter_fn=lambda x: bool((x.instance_labels != -100).any()))

    # Downsample
    # TODO: Crop
    pipe = pipe.map(partial(downsample, max_points=max_points))
    pipe = pipe.map(compact_instance_labels)

    # Augmentations
    if augmentation:
        pipe = pipe.map(partial(apply_augmentations, color_jitter=0.1))

    # Generate instance info
    pipe = pipe.map(generate_inst_info)

    # To tensor
    pipe = pipe.map(lambda pc: pc.to_tensor())

    # Voxelization
    pipe = pipe.map(partial(apply_voxelization, voxel_size=voxel_size))

    return pipe


class GAPartNetInst(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        max_points: int = 20000,
        voxel_size: Tuple[float, float, float] = (1 / 100, 1 / 100, 1 / 100),
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_dir = root_dir
        self.max_points = max_points
        self.voxel_size = voxel_size

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit", "validate"):
            self.train_data_pipe = from_folder(
                Path(self.root_dir) / "train" / "pth",
                shuffle=True,
                max_points=self.max_points,
                augmentation=True,
                voxel_size=self.voxel_size,
                few_shot = True,
            )

            self.val_data_pipe = from_folder(
                Path(self.root_dir) / "val" / "pth",
                shuffle=False,
                max_points=self.max_points,
                augmentation=False,
                voxel_size=self.voxel_size,
                few_shot = True
            )

            self.intra_data_pipe = from_folder(
                Path(self.root_dir) / "test_intra" / "pth",
                shuffle=False,
                max_points=self.max_points,
                augmentation=False,
                voxel_size=self.voxel_size,
                few_shot = True
            )

            self.inter_data_pipe = from_folder(
                Path(self.root_dir) / "test_inter" / "pth",
                shuffle=False,
                max_points=self.max_points,
                augmentation=False,
                voxel_size=self.voxel_size,
                few_shot = True
            )

        if stage in (None, "test"):
            self.test_data_pipe = from_folder(
                Path(self.root_dir) / "test" / "pth",
                shuffle=False,
                max_points=self.max_points,
                augmentation=False,
                voxel_size=self.voxel_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data_pipe,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=data_utils.trivial_batch_collator,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_data_pipe,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=data_utils.trivial_batch_collator,
                pin_memory=True,
                drop_last=False,
            ),
            DataLoader(
                self.intra_data_pipe,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=data_utils.trivial_batch_collator,
                pin_memory=True,
                drop_last=False,
            ),
            DataLoader(
                self.inter_data_pipe,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=data_utils.trivial_batch_collator,
                pin_memory=True,
                drop_last=False,
            ),
        ]
    def test_dataloader(self):
        return DataLoader(
            self.test_data_pipe,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_utils.trivial_batch_collator,
            pin_memory=True,
            drop_last=False,
        )
