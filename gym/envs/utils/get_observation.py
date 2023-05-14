from random import shuffle
from sre_parse import State
from utils.gym_info import *
from pathlib import Path
import torch
from os.path import join as pjoin
import numpy as np
from isaacgym.gymtorch import wrap_tensor
from .compute import *
from .get_running_bbox import get_bbox_isaac_tensor, _draw_bbox_tensor,  get_bbox_pt, get_bbox_pt_nohandle ,get_bbox_isaac_tensor_nohandle
from pytorch3d.transforms import quaternion_invert, matrix_to_quaternion, quaternion_apply, quaternion_multiply
from data_structure.observation import Observations


def _get_base_observation(task, suggested_gt=None) :
    task.dof_state_tensor_all = wrap_tensor(task.gym.acquire_dof_state_tensor(task.sim))
    task.rigid_body_tensor_all = wrap_tensor(task.gym.acquire_rigid_body_state_tensor(task.sim))
    task.dof_state_tensor_used = task.dof_state_tensor_all[task.dof_state_mask]
    task.rigid_body_tensor_used = task.rigid_body_tensor_all[task.rigid_state_mask]

    task.hand_rigid_body_tensor = task.rigid_body_tensor_used\
        [:, task.hand_rigid_body_index, :] # N*13
    task.franka_dof_tensor = task.dof_state_tensor_used\
        [:, :task.franka_num_dofs, :] # N*11*2
    task.cabinet_dof_tensor = task.dof_state_tensor_used\
        [:, task.cabinet_dof_index, :] # N*2
    task.cabinet_dof_tensor_spec = task.cabinet_dof_tensor.view(
        task.cabinet_num, task.env_per_asset, -1) # M*(N/M)*2
        
    task.cabinet_base_rigid_body_tensor = task.rigid_body_tensor_used[:, task.franka_num_dofs, :] 
    task.cabinet_part_rigid_body_tensor = task.rigid_body_tensor_used[:, task.franka_num_dofs+1, :] 
    task.cabinet_handle_rigid_body_tensor = task.rigid_body_tensor_used[:, task.franka_num_dofs+2, :] 

    task.cabinet_handle_pos_tensor = task.cabinet_handle_rigid_body_tensor[:,:3]


    hand_rot = task.hand_rigid_body_tensor[..., 3:7]
    hand_down_dir = quat_axis(hand_rot, 2)
    task.hand_tip_pos = task.hand_rigid_body_tensor[..., 0:3] + hand_down_dir * 0.130    # calculating middle of two fingers
    task.hand_mid_pos = task.hand_rigid_body_tensor[..., 0:3] + hand_down_dir * 0.07 # TODO modify
    task.hand_rot = hand_rot
    
    # if task.cfg["task"]["use_first_bbox"] is None:
    #     if task.target_part == "door":
    #         part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0], 0)
    #     elif task.target_part == "drawer":
    #         part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0], 1)
    # else:
    #     part_bbox_tensor = task.init_part_bbox_tensor
    #     handle_bbox_tensor = task.init_handle_bbox_tensor
    


    if task.target_part in ["door", "drawer", "handle"]:
        if task.cfg["obs"]["state"]["use_bbox_type"] == "gt":
            if task.target_part in ["door", "handle"]:
                part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0], 0)
            elif task.target_part == "drawer":
                part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0], 1)
        else:
            part_bbox_tensor , handle_bbox_tensor= get_bbox_pt(task)
    elif task.target_part == "button":
        if task.cfg["obs"]["state"]["use_bbox_type"] == "gt":
            part_bbox_tensor = get_bbox_isaac_tensor_nohandle(task, task.cabinet_dof_tensor[:,0], 1)
        else:
            part_bbox_tensor = get_bbox_pt_nohandle(task)
        handle_bbox_tensor = part_bbox_tensor.clone()
    
    # _draw_bbox_tensor(task, handle_bbox_tensor)
    # _draw_bbox_tensor(task, part_bbox_tensor)

    hand_pose = relative_pose(task.franka_root_tensor, task.hand_rigid_body_tensor).view(task.env_num, -1)
    
    if task.cfg["task"]["canonicalize"]:
        hand_pose[:, :3] += task.franka_root_tensor[:, :3]  
        hand_pose[:, :3] -= task.canon_center
        hand_pose[:, :3] = quaternion_apply(task.canon_quaternion_rot, hand_pose[:, :3])
        hand_pose[:, 3:7] = quaternion_multiply(task.canon_quaternion_rot, hand_pose[:, 3:7])

        handle_bbox_tensor = handle_bbox_tensor - task.canon_center.view(-1, 1, 3)
        part_bbox_tensor = part_bbox_tensor - task.canon_center.view(-1, 1, 3)

        handle_bbox_tensor = quaternion_apply(task.canon_quaternion_rot.view(-1, 1, 4), handle_bbox_tensor) #@task.rot_matrix
        part_bbox_tensor = quaternion_apply(task.canon_quaternion_rot.view(-1, 1, 4), part_bbox_tensor) #@task.rot_matrix
        root_tensor = quaternion_apply(task.canon_quaternion_rot.view(-1, 4), task.franka_root_tensor[:, :3] - task.canon_center)
    else:
        root_tensor = task.franka_root_tensor[:, :3]

        
    handle_out = handle_bbox_tensor[:, 0] - handle_bbox_tensor[:, 4]
    handle_long = handle_bbox_tensor[:, 1] - handle_bbox_tensor[:, 0]
    handle_short = handle_bbox_tensor[:, 3] - handle_bbox_tensor[:, 0]
    handle_center = (handle_bbox_tensor[:, 0] + handle_bbox_tensor[:, 6]) / 2
    part_out = part_bbox_tensor[:, 0] - part_bbox_tensor[:, 4]
    part_long = part_bbox_tensor[:, 1] - part_bbox_tensor[:, 0]
    part_short = part_bbox_tensor[:, 3] - part_bbox_tensor[:, 0]
    part_center = (part_bbox_tensor[:, 0] + part_bbox_tensor[:, 6]) / 2

    if task.target_part in ["door", "handle"]:
        part_center_input = (task.init_part_bbox_tensor[:,1,:] + task.init_part_bbox_tensor[:,3,:])/2 
    elif task.target_part =="drawer":
        part_center_input = (task.init_part_bbox_tensor[:,2,:] + task.init_part_bbox_tensor[:,5,:])/2
        
    elif task.target_part =="button":
        part_center_input = (
            task.init_part_bbox_tensor[:,0,:] + task.init_part_bbox_tensor[:,1,:] + task.init_part_bbox_tensor[:,2,:] + task.init_part_bbox_tensor[:,3,:] + 
            task.init_part_bbox_tensor[:,4,:] + task.init_part_bbox_tensor[:,5,:] + task.init_part_bbox_tensor[:,6,:] + task.init_part_bbox_tensor[:,7,:]
            )/8
    else:
        raise NotImplementedError
    
    # part_center_input = (part_bbox_tensor[:,1,:] + part_bbox_tensor[:,3,:])/2
    
    state = torch.zeros((task.num_envs, 0), device = task.device, dtype=torch.float32)
    obs = torch.zeros((task.num_envs, 0), device = task.device, dtype=torch.float32)
    robot_qpose = (2 * (task.franka_dof_tensor[:, :, 0]-task.franka_dof_lower_limits_tensor[:])/(task.franka_dof_upper_limits_tensor[:] - task.franka_dof_lower_limits_tensor[:])) - 1
    robot_qvel = task.franka_dof_tensor[:, :, 1]
    cabinet_qpose = task.cabinet_dof_tensor
    hand_root_pose = torch.cat((root_tensor, hand_pose), dim=1)
    # bbox

    ##################### state ########################
    # state = torch.cat((state, robot_qpose, robot_qvel, cabinet_qpose, hand_root_pose, part_center_input), dim = 1)
    state = torch.cat((state, robot_qpose, robot_qvel, cabinet_qpose, hand_root_pose), dim = 1)
    
    if task.cfg["obs"]["pose_baseline"] is not None:
        if task.cfg["obs"]["pose_baseline"] == "yes":
            handle_out = torch.zeros_like(handle_out, device=task.device)
            handle_long = torch.zeros_like(handle_long, device=task.device)
            handle_short = torch.zeros_like(handle_short, device=task.device)
            part_out = torch.zeros_like(part_out, device=task.device)
            part_long = torch.zeros_like(part_long, device=task.device)
            part_short = torch.zeros_like(part_short, device=task.device)

    #####################  obs  ########################
    if task.cfg["obs"]["state"]["use_bbox_form"] == "vertex": # 48
        handle_bbox = handle_bbox_tensor.view(-1, 24)
        part_bbox = part_bbox_tensor.view(-1, 24)
        state = torch.cat((state, handle_bbox, part_bbox), dim = 1)
    elif task.cfg["obs"]["state"]["use_bbox_form"] == "edges": # 24
        state = torch.cat((state, handle_out, handle_long, handle_short, handle_center, part_out, part_long, part_short, part_center), dim = 1)

    # import pdb
    # pdb.set_trace()

    if task.cfg["obs"]["state"]["use_robot_qpose"]: # numActions * 2 # 22 for slider
        obs = torch.cat((obs, robot_qpose, robot_qvel), dim = 1)
    if task.cfg["obs"]["state"]["use_cabinet_qpose"]: # 2
        obs = torch.cat((obs, cabinet_qpose), dim = 1)
    if task.cfg["obs"]["state"]["use_hand_root_pose"]: # 16
        obs = torch.cat((obs, hand_root_pose), dim = 1)
    
    if task.cfg["obs"]["state"]["use_cabinet_bbox"]:
        if task.cfg["obs"]["state"]["use_bbox_form"] == "vertex": # 48
            handle_bbox = handle_bbox_tensor.view(-1, 24)
            part_bbox = part_bbox_tensor.view(-1, 24)
            obs = torch.cat((obs, handle_bbox, part_bbox), dim = 1)
        elif task.cfg["obs"]["state"]["use_bbox_form"] == "edges": # 24
            obs = torch.cat((obs, handle_out, handle_long, handle_short, handle_center, part_out, part_long, part_short, part_center), dim = 1)
    if task.cfg["obs"]["state"]["use_part_center"]: # 16
        obs = torch.cat((obs, part_center_input), dim = 1)
    # import pdb
    # pdb.set_trace()
    if task.cfg["obs"]["state"]["use_noise_feature"]:
        obs = torch.cat((obs, torch.rand((task.num_envs,112), device = task.device)), dim = 1)
        state = torch.cat((state, torch.rand((task.num_envs,112), device = task.device)), dim = 1)
    return Observations(state = state, obs = obs)

def _refresh_observation(task) :
    task.gym.refresh_actor_root_state_tensor(task.sim)
    task.gym.refresh_dof_state_tensor(task.sim)
    task.gym.refresh_rigid_body_state_tensor(task.sim)
    if "ik" in task.cfg["env"]["driveMode"]:
        task.gym.refresh_jacobian_tensors(task.sim)

    task.obs_buf = task._get_base_observation()
    