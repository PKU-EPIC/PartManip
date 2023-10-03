from scipy.spatial.transform import Rotation as R
import numpy as np
import isaacgym
from .misc import _draw_line
from utils.gym_info import *
import torch
import pytorch3d as p3d
from pytorch3d.transforms import axis_angle_to_matrix

def get_bbox_for_now(bbox, axis_xyz, axis_dir, qpos, ):
    rotation_matrix = R.from_rotvec(qpos * axis_dir).as_matrix()
    return np.dot(rotation_matrix, (bbox - axis_xyz).T).T + axis_xyz

def get_bbox_from_world_to_isaac(bbox):
    matrix = R.from_quat(object_init_pose_r_np).as_matrix()
    return np.matmul(matrix, np.array(bbox).T).T+object_init_pose_p_np

# def get_axis_xyz_from_world_to_isaac(axis_xyz):
#     matrix = R.from_quat(object_init_pose_r_np).as_matrix()
#     return np.matmul(matrix, np.array(axis).T).T+object_init_pose_p_np

def get_bbox_isaac(task, qpos, asset_id):
    return get_bbox_from_world_to_isaac(get_bbox_for_now(
        task.part_bbox[asset_id], task.part_axis_xyz[asset_id], task.part_axis_dir[asset_id], qpos)), \
        get_bbox_from_world_to_isaac(get_bbox_for_now(
        task.handle_bbox[asset_id], task.part_axis_xyz[asset_id], task.part_axis_dir[asset_id], qpos))

def _draw_bbox(task, bbox, env_id = 0):
    _draw_line(task, bbox[0], bbox[1], clear = False, env_id=env_id) 
    _draw_line(task, bbox[1], bbox[2], clear = False, env_id=env_id) 
    _draw_line(task, bbox[2], bbox[3], clear = False, env_id=env_id) 
    _draw_line(task, bbox[3], bbox[0], clear = False, env_id=env_id) 
    _draw_line(task, bbox[4], bbox[5], clear = False, env_id=env_id) 
    _draw_line(task, bbox[6], bbox[5], clear = False, env_id=env_id) 
    _draw_line(task, bbox[6], bbox[7], clear = False, env_id=env_id) 
    _draw_line(task, bbox[4], bbox[7], clear = False, env_id=env_id) 
    _draw_line(task, bbox[4], bbox[0], clear = False, env_id=env_id) 
    _draw_line(task, bbox[2], bbox[6], clear = False, env_id=env_id) 
    _draw_line(task, bbox[3], bbox[7], clear = False, env_id=env_id) 
    _draw_line(task, bbox[1], bbox[5], clear = False, env_id=env_id) 

def get_bbox_for_now_tensor(bbox, axis_xyz, axis_dir, qpos, type = type):
    if type == 0: #door 
        rotation_matrix = axis_angle_to_matrix((qpos * axis_dir.T).T)
        bbox_canon = (bbox - axis_xyz.reshape(-1,1,3)).reshape(-1,8,3,1)
        return torch.matmul(rotation_matrix.reshape(-1,1,3,3), bbox_canon).reshape(-1,8,3) + axis_xyz.reshape(-1,1,3)
    elif type == 1: # drawer
        bbox_canon = (bbox - axis_xyz.reshape(-1,1,3))#.reshape(-1,8,3,1)
        return bbox_canon + qpos.reshape(-1,1,1) * axis_dir.reshape(-1,1,3) + axis_xyz.reshape(-1,1,3)

def get_bbox_from_world_to_isaac_tensor(task, bbox):
    # print(torch.matmul(task.object_init_pose_r_matrix_tensor, bbox.reshape(-1,8,3,1)).shape)
    return torch.matmul(task.object_init_pose_r_matrix_tensor, bbox.reshape(-1,8,3,1)).reshape(-1,8,3)+task.object_init_pose_p_tensor

def get_bbox_isaac_tensor(task, qpos_tensor, type = 0):
    return get_bbox_from_world_to_isaac_tensor(task, get_bbox_for_now_tensor(
        task.part_bbox_tensor_init, task.part_axis_xyz_tensor_init, task.part_axis_dir_tensor_init, qpos_tensor, type = type)), \
        get_bbox_from_world_to_isaac_tensor(task, get_bbox_for_now_tensor(
        task.handle_bbox_tensor_init, task.part_axis_xyz_tensor_init, task.part_axis_dir_tensor_init, qpos_tensor, type = type))

def get_bbox_isaac_tensor_nohandle(task, qpos_tensor, type = 0):
    return get_bbox_from_world_to_isaac_tensor(task, get_bbox_for_now_tensor(
        task.part_bbox_tensor_init, task.part_axis_xyz_tensor_init, task.part_axis_dir_tensor_init, qpos_tensor, type = type))
        # get_bbox_from_world_to_isaac_tensor(task, get_bbox_for_now_tensor(
        # task.handle_bbox_tensor_init, task.part_axis_xyz_tensor_init, task.part_axis_dir_tensor_init, qpos_tensor, type = type))



def get_bbox_pt(task):
    return task.part_pt_bbox_tensor, task.handle_pt_bbox_tensor

def get_bbox_pt_nohandle(task):
    return task.part_pt_bbox_tensor

def _draw_bbox_pt(task, bbox_tensor):
    for _id in range(bbox_tensor.shape[0]):
       _draw_line(task, bbox_tensor[_id][0].cpu().numpy(), bbox_tensor[_id][1].cpu().numpy(), clear = False, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][1].cpu().numpy(), bbox_tensor[_id][5].cpu().numpy(), clear = False, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][5].cpu().numpy(), bbox_tensor[_id][3].cpu().numpy(), clear = False, env_id=_id)
       _draw_line(task, bbox_tensor[_id][3].cpu().numpy(), bbox_tensor[_id][0].cpu().numpy(), clear = False, env_id=_id)  
       
       _draw_line(task, bbox_tensor[_id][4].cpu().numpy(), bbox_tensor[_id][2].cpu().numpy(), clear = False, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][2].cpu().numpy(), bbox_tensor[_id][6].cpu().numpy(), clear = False, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][6].cpu().numpy(), bbox_tensor[_id][7].cpu().numpy(), clear = False, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][7].cpu().numpy(), bbox_tensor[_id][4].cpu().numpy(), clear = False, env_id=_id) 
       
       _draw_line(task, bbox_tensor[_id][4].cpu().numpy(), bbox_tensor[_id][1].cpu().numpy(), clear = False, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][2].cpu().numpy(), bbox_tensor[_id][0].cpu().numpy(), clear = False, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][3].cpu().numpy(), bbox_tensor[_id][6].cpu().numpy(), clear = False, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][7].cpu().numpy(), bbox_tensor[_id][5].cpu().numpy(), clear = False, env_id=_id) 
       
def _draw_bbox_tensor(task, bbox_tensor, clear=False):
    for _id in range(bbox_tensor.shape[0]):
       _draw_line(task, bbox_tensor[_id][0].cpu().numpy(), bbox_tensor[_id][1].cpu().numpy(), clear = clear, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][1].cpu().numpy(), bbox_tensor[_id][2].cpu().numpy(), clear = clear, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][2].cpu().numpy(), bbox_tensor[_id][3].cpu().numpy(), clear = clear, env_id=_id)
       _draw_line(task, bbox_tensor[_id][3].cpu().numpy(), bbox_tensor[_id][0].cpu().numpy(), clear = clear, env_id=_id)  
       _draw_line(task, bbox_tensor[_id][4].cpu().numpy(), bbox_tensor[_id][5].cpu().numpy(), clear = clear, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][6].cpu().numpy(), bbox_tensor[_id][5].cpu().numpy(), clear = clear, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][6].cpu().numpy(), bbox_tensor[_id][7].cpu().numpy(), clear = clear, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][4].cpu().numpy(), bbox_tensor[_id][7].cpu().numpy(), clear = clear, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][4].cpu().numpy(), bbox_tensor[_id][0].cpu().numpy(), clear = clear, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][2].cpu().numpy(), bbox_tensor[_id][6].cpu().numpy(), clear = clear, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][3].cpu().numpy(), bbox_tensor[_id][7].cpu().numpy(), clear = clear, env_id=_id) 
       _draw_line(task, bbox_tensor[_id][1].cpu().numpy(), bbox_tensor[_id][5].cpu().numpy(), clear = clear, env_id=_id) 
       
if __name__ == "__main__":
    print("yes")
    bbox_tensor = torch.rand((5,8,3))
    axis_xyz_tensor = torch.rand((5,3))
    axis_dir_tensor = torch.rand((5,3))
    qpos_tensor = torch.rand(5)
    print(torch.permute(bbox_tensor,(1,0,2)).shape)
    print(get_bbox_for_now_tensor(bbox_tensor, axis_xyz_tensor, axis_dir_tensor,qpos_tensor)[0])
    print(get_bbox_for_now(bbox_tensor[0].numpy(), axis_xyz_tensor[0].numpy(), axis_dir_tensor[0].numpy(),qpos_tensor[0].numpy()))