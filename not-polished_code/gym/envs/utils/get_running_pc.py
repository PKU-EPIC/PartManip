from scipy.spatial.transform import Rotation as R
import numpy as np
import isaacgym
from .misc import _draw_line
from utils.gym_info import *
import torch
import pytorch3d as p3d
from pytorch3d.transforms import axis_angle_to_matrix

def get_pc_for_now(pc_part, axis_xyz, axis_dir, qpos, ):
    rotation_matrix = R.from_rotvec(qpos * axis_dir).as_matrix()
    return np.dot(rotation_matrix, (pc_part - axis_xyz).T).T + axis_xyz

# def get_bbox_from_world_to_isaac(bbox):
#     matrix = R.from_quat(object_init_pose_r_np).as_matrix()
#     return np.matmul(matrix, np.array(bbox).T).T+object_init_pose_p_np

# def get_axis_xyz_from_world_to_isaac(axis_xyz):
#     matrix = R.from_quat(object_init_pose_r_np).as_matrix()
#     return np.matmul(matrix, np.array(axis).T).T+object_init_pose_p_np

def get_pc_isaac(task, qpos, asset_id, env_id):
    return get_pc_for_now(task.pseudo_pc_part[asset_id], task.part_axis_xyz_tensor[env_id], task.part_axis_dir_tensor[env_id], qpos), \
        get_pc_for_now(task.pseudo_pc_handle[asset_id], task.part_axis_xyz_tensor[env_id], task.part_axis_dir_tensor[env_id], qpos), 


def get_pc_for_now_tensor(pc, axis_xyz, axis_dir, qpos, type = type):
    # import pdb
    # pdb.set_trace()
    pc_num = pc.shape[1]
    pc_new = pc.clone()
    # import pdb
    # pdb.set_trace()
    if type == 0: #door 
        rotation_matrix = axis_angle_to_matrix((qpos * axis_dir.T).T)
        pc_canon = (pc[:, :, :3] - axis_xyz.reshape(-1,1,3)).reshape(-1,pc_num,3,1)
        pc_new[:, :, :3] = torch.matmul(rotation_matrix.reshape(-1,1,3,3), pc_canon).reshape(-1,pc_num,3) + axis_xyz.reshape(-1,1,3)
        return pc_new
    elif type == 1: # drawer
        pc_new[:, :, :3] = (pc[:, :, :3] - axis_xyz.reshape(-1,1,3))#.reshape(-1,pc_num,3,1)
        pc_new[:, :, :3] = pc_new[:, :, :3] + qpos.reshape(-1,1,1) * axis_dir.reshape(-1,1,3) + axis_xyz.reshape(-1,1,3)
        return pc

def get_pc_isaac_tensor(task, qpos_tensor, type = 0):
    return get_pc_for_now_tensor(task.pseudo_pc_init, task.part_axis_xyz_tensor, task.part_axis_dir_tensor, qpos_tensor, type = type)
        # get_bbox_for_now_tensor(
        # task.pseudo_pc, task.part_axis_xyz_tensor_init, task.part_axis_dir_tensor_init, qpos_tensor, type = type)

def get_bbox_pt(task):
    return task.part_pt_bbox_tensor, task.handle_pt_bbox_tensor

if __name__ == "__main__":
    print("yes")
    bbox_tensor = torch.rand((5,8,3))
    axis_xyz_tensor = torch.rand((5,3))
    axis_dir_tensor = torch.rand((5,3))
    qpos_tensor = torch.rand(5)
    print(torch.permute(bbox_tensor,(1,0,2)).shape)
    print(get_bbox_for_now_tensor(bbox_tensor, axis_xyz_tensor, axis_dir_tensor,qpos_tensor)[0])
    print(get_bbox_for_now(bbox_tensor[0].numpy(), axis_xyz_tensor[0].numpy(), axis_dir_tensor[0].numpy(),qpos_tensor[0].numpy()))