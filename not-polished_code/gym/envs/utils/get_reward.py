
from typing import List
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
     quat_mul, tensor_clamp
import numpy as np
import time
import torch
from .misc import _draw_line
from scipy.spatial.transform import Rotation as R
from .get_running_bbox import get_bbox_for_now, get_bbox_from_world_to_isaac, _draw_bbox, \
    get_bbox_isaac, get_bbox_for_now_tensor, get_bbox_from_world_to_isaac_tensor, \
    get_bbox_isaac_tensor, _draw_bbox_tensor

# Y apply rotation in basis vector
def quat_axis(q, axis=0):
    '''
    :func apply rotation represented by quanternion `q`
    on basis vector(along axis)
    :return vector after rotation
    '''
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

def normalize_and_clip_in_interval(x, min_x, max_x=None):
    if max_x is None:
        min_x = -abs(min_x)
        max_x = abs(min_x)
    len_x = max_x - min_x
    # print(torch.max(x, torch.ones_like(x) * min_x))
    # print(torch.min(torch.max(x, torch.ones_like(x) * min_x), torch.ones_like(x) * max_x))
    # print(torch.min(torch.max(x, torch.ones_like(x) * min_x), torch.ones_like(x) * max_x) - min_x)
    # print(len_x)
    # print((torch.min(torch.max(x, torch.ones_like(x) * min_x), torch.ones_like(x) * max_x) - min_x) / len_x)
    return (torch.min(torch.max(x, torch.ones_like(x) * min_x), torch.ones_like(x) * max_x) - min_x) / len_x

def get_reward_done(task):
    # what we use:
    # handle_pos: handle position array: the middle
    # task.hand_tip_pos: hand tip position array
    # franka_lfinger_pos: left finger position array 
    # franka_rfinger_pos: right finger position array
    # hand_down_dir: hand down rotation matrix
    part_pos = task.cabinet_part_rigid_body_tensor[:, :3]
    part_rot = task.cabinet_part_rigid_body_tensor[:, 3:7]
    hand_rot = task.hand_rigid_body_tensor[..., 3:7]
    hand_grip_dir = quat_axis(hand_rot, 2)
    hand_sep_dir = quat_axis(hand_rot, 1)
    hand_down_dir = quat_axis(hand_rot, 0)
    #handle_pos = quat_apply(
    #    part_rot, task.cabinet_handle_pos_tensor) + part_pos

    #handle_pos = get_bbox_isaac()
    #print(quat_axis(part_rot, 0).shape)
    #print(task.cabinet_open_dir_tensor.view(-1, 1).shape)
    #handle_x = quat_axis(part_rot, 0)\
    #    * task.cabinet_open_dir_tensor.view(-1, 1)
    #handle_z = quat_axis(part_rot, 1)
    
    #cabinet_part_relative_o = part_pos + quat_apply(
    #    part_rot, task.cabinet_part_edge_min)
    #cabinet_part_relative_x = -handle_x
    #cabinet_part_relative_y = -quat_axis(part_rot, 2)
    #cabinet_part_relative_z = quat_axis(part_rot, 1)
    #time.sleep(1)
    franka_lfinger_pos = task.rigid_body_tensor_used[:, task.hand_lfinger_rigid_body_index][:, 0:3]\
        + hand_grip_dir*0.1
    franka_rfinger_pos = task.rigid_body_tensor_used[:, task.hand_rfinger_rigid_body_index][:, 0:3]\
        + hand_grip_dir*0.1
    # task.part_bbox_tensor, task.handle_bbox_tensor = get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0])
    if task.target_part in ["door", "handle"]:
        task.part_bbox_tensor, task.handle_bbox_tensor = get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0], 0)
    elif task.target_part == "drawer":
        task.part_bbox_tensor, task.handle_bbox_tensor = get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0], 1)
    #task.part_bbox_tensor, task.handle_bbox_tensor = get_bbox_isaac_tensor(task, torch.zeros(task.env_num, device = task.device))
    #part_bbox_tensor, handle_bbox_tensor= get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0])
    #_draw_bbox_tensor(task, task.handle_bbox_tensor)
    #_draw_bbox_tensor(task, task.part_bbox_tensor)
    rot_coefficient = task.cfg["task"]["rot_coef"]
    handle_coefficient = task.cfg["task"]["handle_coef"]
    vel_coefficient = task.cfg["task"]["vel_coef"]
    dist_coefficient = task.cfg["task"]["dist_coef"]
    tip_coefficient = task.cfg["task"]["tip_coef"]
    close_coefficient = task.cfg["task"]["close_coef"]
    if task.target_part in ["door", "drawer"]:
        handle_out = task.handle_bbox_tensor[:, 0] - task.handle_bbox_tensor[:, 4]
        handle_long = task.handle_bbox_tensor[:, 1] - task.handle_bbox_tensor[:, 0]
        handle_short = task.handle_bbox_tensor[:, 3] - task.handle_bbox_tensor[:, 0]
        handle_mid = (task.handle_bbox_tensor[:, 0] + task.handle_bbox_tensor[:, 6]) / 2
        handle_out_length = torch.norm(handle_out, dim = -1)
        handle_long_length = torch.norm(handle_long, dim = -1)
        handle_short_length = torch.norm(handle_short, dim = -1)
        handle_shortest = torch.min(torch.min(handle_out_length, handle_long_length), handle_short_length)
    
        # basic reward
        #reward = 0*torch.ones_like(task.rew_buf, device=task.device)
        #scale = 1*torch.ones_like(task.rew_buf, device=task.device)
        stage_reward = 0*torch.ones_like(task.rew_buf, device=task.device)

        ## rotation reward old
        #dot1 = (hand_grip_dir * handle_z).sum(dim=-1)
        #dot2 = (-hand_sep_dir * handle_x).sum(dim=-1)
        #rot_reward = 0.5 * (torch.sign(dot1)*dot1**2 + torch.sign(dot2)*dot2**2)
        handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
        handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
        handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
        ##rotation reward new
        if task.cfg['task']['target_part_category'] == 'drawer':
            dot1 = (-hand_grip_dir * handle_out).sum(dim=-1)
            #dot2 = (hand_down_dir * handle_long).sum(dim=-1)
            dot2 = torch.max((hand_sep_dir * handle_short).sum(dim=-1), (-hand_sep_dir * handle_short).sum(dim=-1)) 
            dot3 = torch.max((hand_down_dir * handle_long).sum(dim=-1), (-hand_down_dir * handle_long).sum(dim=-1))
        else:
            dot1 = (-hand_grip_dir * handle_out).sum(dim=-1)
            #dot2 = (hand_down_dir * handle_long).sum(dim=-1)
            dot2 = torch.max((hand_sep_dir * handle_short).sum(dim=-1), (-hand_sep_dir * handle_short).sum(dim=-1)) 
            dot3 = torch.max((hand_down_dir * handle_long).sum(dim=-1), (-hand_down_dir * handle_long).sum(dim=-1))
        #rot_reward = torch.minimum(torch.sign(dot1)*dot1**2 , torch.sign(dot2)*dot2**2) 
        #rot_reward = 0.5 * (torch.sign(dot1)*dot1**2 + torch.sign(dot2)*dot2**2) 
        #print(handle_shortest[0])
        ## distance reward
        #dist_tip = torch.norm(task.hand_tip_pos - handle_mid,  dim = -1)
        #dist_lfinger = torch.norm(franka_lfinger_pos - handle_mid, dim = -1)
        #dist_rfinger = torch.norm(franka_rfinger_pos - handle_mid, dim = -1)
        part_mid = (
            task.part_bbox_tensor[:, 0] + task.part_bbox_tensor[:, 1] + task.part_bbox_tensor[:, 2] + task.part_bbox_tensor[:, 3] +
            task.part_bbox_tensor[:, 4] + task.part_bbox_tensor[:, 5] + task.part_bbox_tensor[:, 6] + task.part_bbox_tensor[:, 7] 
            ) / 8
        finger_mid = (franka_lfinger_pos + franka_rfinger_pos) / 2
        dist_mid = torch.norm(finger_mid - handle_mid, dim = -1)
        #rew_ee_mid_handle = (normalize_and_clip_in_interval(dist_mid, -0.01, 4E-3) - 1)
        #_draw_line(task, franka_lfinger_pos[0].cpu(), franka_rfinger_pos[0].cpu())
        #_draw_line(task, handle_mid[0].cpu(), (task.hand_rigid_body_tensor[0, :3]).cpu())

        if task.cfg["task"]["use_soft_dist"]:
            rew_ee_handle = (1.0 / (1.0 + dist_mid ** 2)) ** 2
            rew_ee_handle = torch.where(dist_mid <= 0.1, rew_ee_handle ** 2, rew_ee_handle) - 0.75
        else: 
            rew_ee_handle = -dist_mid

        #print(task.cabinet_dof_tensor[:, 0])
        # next_stage_reward
        ###close
        if task.cfg["task"]["target_state"] == "close" :
            task.rew_buf = - task.cabinet_dof_tensor[:, 0] * handle_coefficient - dist_mid * dist_coefficient
        ###open
        else: 
            rot_reward = (torch.sign(dot1)*dot1**2 + 0.5 * (torch.sign(dot2)*dot2**2 + torch.sign(dot3)*dot3**2)) / 2
            #rot_reward = torch.sign(dot1)*dot1**2 
            #rot_reward = 0.5 * (torch.sign(dot1)*dot1**2 + torch.sign(dot2)*dot2**2) 
            stage_reward += (dist_mid <= handle_shortest) * torch.clamp(1 - close_coefficient * torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim = -1), 0, 1) #tip distance 

            #vel_reward = normalize_and_clip_in_interval(task.cabinet_dof_tensor[:, 1], -0.1, 0.5) * vel_coefficient  # Push vel to positive
            dist_reward = task.cabinet_dof_tensor[:, 0] * (dist_coefficient + tip_coefficient * (dist_mid <= handle_shortest / 2))
            #夹住handle然后打开门获得更高reward
            #stage_reward += (vel_coefficient) * (task.cabinet_dof_tensor[:, 0] > task.cabinet_dof_target - 0.01) #if open enough
            #stage_reward += 1*(task.cabinet_dof_tensor[:, 0] > task.cabinet_dof_target - 0.01)*(task.cabinet_dof_tensor[:, 1] < 0.01) #if open enough and stable 
            if task.cfg["task"]["task"] =='open' and task.cfg["task"]["two_stage"]:
                rew_ee_handle *= (dist_mid >= handle_shortest / 2) * torch.clip((task.cabinet_dof_target - task.cabinet_dof_tensor[:, 0] * task.cfg["task"]["dicouple_coef"]) / task.cabinet_dof_target, min = 0)
            if task.cfg["task"]["grasp_first"]:
                dist_reward *= (task.progress_buf >= task.max_episode_length / 2)
            task.rew_buf = rot_reward * rot_coefficient + rew_ee_handle * handle_coefficient \
                                + stage_reward * vel_coefficient + dist_reward * (task.cfg["task"]["task"] =='open')
                                #el_reward + (- vel_reward) * (task.cabinet_dof_tensor[:, 0] > task.cabinet_dof_target - 0.01)

            #print(rot_reward.mean(), rew_ee_handle.mean(), stage_reward.mean(), dist_reward.mean())
            #reward_info = {
            #    "hand rotation reward": rot_reward,xw
            #    "tip position reward": rew_ee_handle,
            #    "hand mid position reward": rew_ee_mid_handle,
            #    "stage reward": stage_reward,
            #    "cabinet position reward": dist_reward,
            #    "cabinet velocity reward": vel_reward,
            #    "total reward": reward,
            #}
    elif task.cfg["task"]["task"] == 'push_button':
        task.rew_buf = task.cabinet_dof_tensor[:, 0] * dist_coefficient
        part_mid = (
            task.part_bbox_tensor[:, 0] + task.part_bbox_tensor[:, 1] + task.part_bbox_tensor[:, 2] + task.part_bbox_tensor[:, 3] +
            task.part_bbox_tensor[:, 4] + task.part_bbox_tensor[:, 5] + task.part_bbox_tensor[:, 6] + task.part_bbox_tensor[:, 7] 
            ) / 8
        finger_mid = (franka_lfinger_pos + franka_rfinger_pos) / 2
        dist_mid = torch.norm(finger_mid - part_mid, dim = -1)
        rot_reward = rot_coefficient * (-hand_grip_dir * torch.tensor([[1, 0, 0]], device=task.device).repeat(hand_grip_dir.shape[0], 1)).sum(dim=-1)
        dist_reward = -dist_mid * handle_coefficient 
        finger_reward = - close_coefficient * torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim = -1)
        task.rew_buf += dist_reward + rot_reward + finger_reward

    elif task.cfg["task"]["task"] == 'grasp':
        handle_out = task.handle_bbox_tensor[:, 0] - task.handle_bbox_tensor[:, 4]
        handle_long = task.handle_bbox_tensor[:, 1] - task.handle_bbox_tensor[:, 0]
        handle_short = task.handle_bbox_tensor[:, 3] - task.handle_bbox_tensor[:, 0]
        handle_mid = (task.handle_bbox_tensor[:, 0] + task.handle_bbox_tensor[:, 6]) / 2
        handle_out_length = torch.norm(handle_out, dim = -1)
        handle_long_length = torch.norm(handle_long, dim = -1)
        handle_short_length = torch.norm(handle_short, dim = -1)
        handle_shortest = torch.min(torch.min(handle_out_length, handle_long_length), handle_short_length)
    
        dot1 = (-hand_grip_dir * handle_out).sum(dim=-1)
        #dot2 = (hand_down_dir * handle_long).sum(dim=-1)
        dot2 = torch.max((hand_sep_dir * handle_short).sum(dim=-1), (-hand_sep_dir * handle_short).sum(dim=-1)) 
        dot3 = torch.max((hand_down_dir * handle_long).sum(dim=-1), (-hand_down_dir * handle_long).sum(dim=-1))

        handle_mid = (
            task.handle_bbox_tensor[:, 0] + task.handle_bbox_tensor[:, 1] + task.handle_bbox_tensor[:, 2] + task.handle_bbox_tensor[:, 3] +
            task.handle_bbox_tensor[:, 4] + task.handle_bbox_tensor[:, 5] + task.handle_bbox_tensor[:, 6] + task.handle_bbox_tensor[:, 7] 
            ) / 8

        init_handle_mid = (
            task.init_handle_bbox_tensor[:, 0] + task.init_handle_bbox_tensor[:, 1] + task.init_handle_bbox_tensor[:, 2] + task.init_handle_bbox_tensor[:, 3] +
            task.init_handle_bbox_tensor[:, 4] + task.init_handle_bbox_tensor[:, 5] + task.init_handle_bbox_tensor[:, 6] + task.init_handle_bbox_tensor[:, 7] 
         ) / 8

        # import pdb
        # pdb.set_trace()

        finger_mid = (franka_lfinger_pos + franka_rfinger_pos) / 2
        dist_mid = torch.norm(finger_mid - handle_mid, dim = -1)
        rot_reward = rot_coefficient * (torch.sign(dot1)*dot1**2 + 0.5 * (torch.sign(dot2)*dot2**2 + torch.sign(dot3)*dot3**2)) / 2
        tip_reward = tip_coefficient * (dist_mid < handle_shortest)  * torch.clamp(1 - close_coefficient * torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim = -1), 0, 1)

        dist_reward = - dist_mid * handle_coefficient #+ dist_coefficient * torch.mul((init_handle_mid - handle_mid), handle_out / torch.norm(handle_out, dim=-1, keepdim=True)).sum(-1)
        task.rew_buf = dist_reward + tip_reward + rot_reward
    else:
        assert(0)
    # done info
    # import pdb
    # pdb.set_trace()
    if task.cfg["task"]["task"] == 'grasp':
        #grasp
        success = torch.logical_and((dist_mid < handle_short_length * 2), 
                                torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim = -1) < handle_short_length * 2+ 0.01)
        # task.success_grasp_buf = task.success_grasp_buf| grasp_success
        #rotation
        # success = torch.logical_and(success, 
        #                 (torch.sign(dot1)*dot1**2 + 0.5 * (torch.sign(dot2)*dot2**2 + torch.sign(dot3)*dot3**2)) / 2 > 0.8)
        # print((torch.sign(dot1)*dot1**2 + 0.5 * (torch.sign(dot2)*dot2**2 + torch.sign(dot3)*dot3**2)) / 2)
        #move
        # success = task.cabinet_dof_tensor_spec[:, :, 0].view(-1) > task.cfg["task"]["openProportion"]
    elif task.cfg["task"]["task"] == 'close':
        diff_from_success = (task.cabinet_dof_tensor_spec[:, :, 0] - task.success_dof_states.view(task.cabinet_num, -1)).view(-1)
        success = (diff_from_success < 0.01)
    elif task.cfg["task"]["task"] == 'push_button':
        diff_from_success = (task.cabinet_dof_tensor_spec[:, :, 0] - task.success_dof_states.view(task.cabinet_num, -1)).view(-1)
        success = (diff_from_success > 0)
    else:
        diff_from_success = (task.success_dof_states.view(task.cabinet_num, -1) - task.cabinet_dof_tensor_spec[:, :, 0]).view(-1)
        success = (diff_from_success < 0.01)
    # _draw_line(task, )
    if task.target_part in ["door", "drawer", "handle"]:
        grasp_success = torch.logical_and((dist_mid < handle_shortest), torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim = -1) < handle_short_length + 0.01).float()
    else:
        grasp_success = None
    #diff_from_success = (task.success_dof_states.view(task.cabinet_num, -1) - task.cabinet_dof_tensor_spec[:, :, 0]).view(-1)
    #success = (diff_from_success < 0.01)
    #success = (dist_mid < handle_shortest)#, torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim = -1) < handle_short_length)


    # print(task.progress_buf)
    time_out = (task.progress_buf >= task.max_episode_length-1)
    task.reset_buf = (task.reset_buf | time_out)
    # if task.cfg["task"]["task"] == 'grasp':
    #     task.success_buf = task.success_buf & (task.success_grasp_buf) & #& task.cabinet_dof_tensor_spec[:, :, 0].view(-1) > task.cfg["task"]["openProportion"]
    # else:
    task.success_buf = task.success_buf | success
    # if task.cfg["task"]["task"] == 'grasp' and time_out.float().mean() > 0:
    #      task.success_buf = task.success_buf & task.success_grasp_buf

    task.success = task.success_buf # & time_out
    # ipdb.set_trace()
    task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] *= 1 - time_out.long()
    task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] += task.success

    #task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] = task.success.float()
    task.success_rate = task.success_queue.mean(dim=-1)
    #task.current_success_rate = task.success_queue[:, task.success_idx]
    task.total_success_rate = task.success_rate.sum(dim=-1)
    task.success_entropy = - task.success_rate/(task.total_success_rate+1e-8) * torch.log(task.success_rate/(task.total_success_rate+1e-8) + 1e-8) * task.env_num
    #print(task.reset_buf)
    
    return task.rew_buf, task.reset_buf, dist_mid, grasp_success

def get_reward_done_all(task):
        # what we use:
    # handle_pos: handle position array: the middle
    # task.hand_tip_pos: hand tip position array
    # franka_lfinger_pos: left finger position array 
    # franka_rfinger_pos: right finger position array
    # hand_down_dir: hand down rotation matrix
    part_pos = task.cabinet_part_rigid_body_tensor[:, :3]
    part_rot = task.cabinet_part_rigid_body_tensor[:, 3:7]
    hand_rot = task.hand_rigid_body_tensor[..., 3:7]
    hand_grip_dir = quat_axis(hand_rot, 2)
    hand_sep_dir = quat_axis(hand_rot, 1)
    hand_down_dir = quat_axis(hand_rot, 0)
    #handle_pos = quat_apply(
    #    part_rot, task.cabinet_handle_pos_tensor) + part_pos

    #handle_pos = get_bbox_isaac()
    #print(quat_axis(part_rot, 0).shape)
    #print(task.cabinet_open_dir_tensor.view(-1, 1).shape)
    #handle_x = quat_axis(part_rot, 0)\
    #    * task.cabinet_open_dir_tensor.view(-1, 1)
    #handle_z = quat_axis(part_rot, 1)
    
    #cabinet_part_relative_o = part_pos + quat_apply(
    #    part_rot, task.cabinet_part_edge_min)
    #cabinet_part_relative_x = -handle_x
    #cabinet_part_relative_y = -quat_axis(part_rot, 2)
    #cabinet_part_relative_z = quat_axis(part_rot, 1)
    franka_lfinger_pos = task.rigid_body_tensor_used[:, task.hand_lfinger_rigid_body_index][:, 0:3]\
        + hand_down_dir*0.075
    franka_rfinger_pos = task.rigid_body_tensor_used[:, task.hand_rfinger_rigid_body_index][:, 0:3]\
        + hand_down_dir*0.075
    # print(task.cabinet_dof_tensor.shape)
    # task.part_bbox_tensor, task.handle_bbox_tensor = get_bbox_isaac_tensor(task, torch.zeros(task.env_num, device = task.device))

    #print(task.cabinet_dof_tensor[:,0])
    # part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0])
    if task.target_part in ["door", "handle"]:
        task.part_bbox_tensor, task.handle_bbox_tensor = get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0], 0)
    elif task.target_part == "drawer":
        task.part_bbox_tensor, task.handle_bbox_tensor = get_bbox_isaac_tensor(task, task.cabinet_dof_tensor[:,0], 1)

    #_draw_bbox_tensor(task, part_bbox_tensor)
    #_draw_bbox_tensor(task, part_bbox_tensor)
    #_draw_bbox_tensor(task, handle_bbox_tensor)
    #_draw_line(task, task.hand_tip_pos[0].cpu(), (task.hand_tip_pos[0] + hand_sep_dir[0]).cpu(), clear = False)
    #_draw_line(task, task.hand_tip_pos[0].cpu(), (task.hand_tip_pos[0] + hand_down_dir[0]).cpu(), clear = False)
    #print(task.handle_bbox_tensor.shape)
    handle_out = task.handle_bbox_tensor[:, 0] - task.handle_bbox_tensor[:, 4]
    handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
    handle_long = task.handle_bbox_tensor[:, 0] - task.handle_bbox_tensor[:, 1]
    handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
    handle_short = task.handle_bbox_tensor[:, 0] - task.handle_bbox_tensor[:, 3]
    handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
    handle_mid = (task.handle_bbox_tensor[:, 0] + task.handle_bbox_tensor[:, 6]) / 2
    
    #handle_out_length = torch.norm(handle_out, dim = 1)
    #handle_long_length = torch.norm(handle_long, dim = 1)
    #handle_short_length = torch.norm(handle_short, dim = 1)
    #handle_shortest = torch.min(torch.min(handle_out_length, handle_long_length), handle_short_length)
    # basic reward
    reward = 0*torch.ones_like(task.rew_buf, device=task.device)
    scale = 1*torch.ones_like(task.rew_buf, device=task.device)
    stage_reward = 0*torch.ones_like(task.rew_buf, device=task.device)

    #_draw_line(task, handle_mid[0].cpu(), (handle_mid[0]+handle_short[0]).cpu(), clear=False)
    #_draw_line(task, handle_mid[0].cpu(), (handle_mid[0]+handle_long[0]).cpu(), clear=False)
    ## rotation reward old
    #dot1 = (hand_grip_dir * handle_z).sum(dim=-1)
    #dot2 = (-hand_sep_dir * handle_x).sum(dim=-1)
    #rot_reward = 0.5 * (torch.sign(dot1)*dot1**2 + torch.sign(dot2)*dot2**2)
    
    ##rotation reward new
    if task.cfg['task']['target_part_category'] == 'drawer':
        dot1 = (-hand_sep_dir * handle_z).sum(dim=-1)
        dot2 = (hand_down_dir * handle_x).sum(dim=-1)
    else:
        dot1 = (hand_sep_dir * handle_short).sum(dim=-1)
        dot2 = (hand_down_dir * handle_long).sum(dim=-1)
        #dot2 = torch.max((hand_down_dir * handle_long).sum(dim=-1), (-hand_down_dir * handle_long).sum(dim=-1))
    #rot_reward = torch.minimum(torch.sign(dot1)*dot1**2 , torch.sign(dot2)*dot2**2) 
    #rot_reward = 0.5 * (torch.sign(dot1)*dot1**2 + torch.sign(dot2)*dot2**2) 
    
    ## distance reward

    dist_tip = torch.norm(task.hand_tip_pos - handle_mid,  dim = -1)
    dist_lfinger = torch.norm(franka_lfinger_pos - handle_mid, dim = -1)
    dist_rfinger = torch.norm(franka_rfinger_pos - handle_mid, dim = -1)
    #dist_mid = torch.norm(task.hand_mid_pos - handle_pos, dim = -1)
    #rew_ee_mid_handle = (normalize_and_clip_in_interval(dist_mid, -0.01, 4E-3) - 1)
    rew_ee_handle = -dist_tip
    if task.cfg["task"]["use_soft_dist"]:
        rew_ee_handle = (1.0 / (1.0 + rew_ee_handle ** 2)) - 1
            

    rot_coefficient = task.cfg["task"]["rot_coef"]
    handle_coefficient = task.cfg["task"]["handle_coef"]
    vel_coefficient = task.cfg["task"]["vel_coef"]
    dist_coefficient = task.cfg["task"]["dist_coef"]
    tip_coefficient = task.cfg["task"]["tip_coef"]
    close_coefficient = 1

    # next_stage_reward
    ###close
    if task.cfg["task"]["target_state"] == "close" :
        task.rew_buf = -task.cabinet_dof_tensor[:, 0]
    ###open
    else: 
        #rot_reward = torch.sign(dot1)*dot1**2 + rot_coefficient * torch.sign(dot2)*dot2**2 
        rot_reward = 0.5 * (torch.sign(dot1)*dot1**2 + torch.sign(dot2)*dot2**2) 
        stage_reward += 1.5 * (dist_tip <= 0.01) #* torch.clamp(1 - close_coefficient * torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim = -1), 0, 1) #tip distance 
        #vel_reward = normalize_and_clip_in_interval(task.cabinet_dof_tensor[:, 1], -0.1, 0.5) * vel_coefficient  # Push vel to positive
        dist_reward = task.cabinet_dof_tensor[:, 0] #夹住handle然后打开门获得更高reward
        dist_reward *= ((dist_tip <= 0.01) * tip_coefficient + dist_coefficient)
        #reward += dist_reward + vel_reward
        stage_reward += (vel_coefficient) * (task.cabinet_dof_tensor[:, 0] > task.cabinet_dof_target - 0.01) #if open enough
        stage_reward += 1*(task.cabinet_dof_tensor[:, 0] > task.cabinet_dof_target - 0.01)*(task.cabinet_dof_tensor[:, 1] < 0.01) #if open enough and stable 
        task.rew_buf = rot_reward * rot_coefficient + rew_ee_handle * handle_coefficient \
                            + stage_reward + dist_reward 
                            #vel_reward + (- vel_reward) * (task.cabinet_dof_tensor[:, 0] > task.cabinet_dof_target - 0.01)

        #print(rot_reward.mean(), rew_ee_handle.mean(), rew_ee_mid_handle.mean(), stage_reward.mean(), dist_reward.mean())#, vel_reward.mean())
        #reward_info = {
        #    "hand rotation reward": rot_reward,
        #    "tip position reward": rew_ee_handle,
        #    "hand mid position reward": rew_ee_mid_handle,
        #    "stage reward": stage_reward,
        #    "cabinet position reward": dist_reward,
        #    "cabinet velocity reward": vel_reward,
        #    "total reward": reward,
        #}

    # done info
    diff_from_success = (task.success_dof_states.view(task.cabinet_num, -1) - task.cabinet_dof_tensor_spec[:, :, 0]).view(-1)
    success = (diff_from_success < 0.01)
    #success = torch.logical_and((dist_tip < handle_shortest), torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim = -1) < handle_short_length)
    time_out = (task.progress_buf >= task.max_episode_length)
    #print(task.progress_buf, task.max_episode_length)
    task.reset_buf = (task.reset_buf | time_out)
    task.success_buf = task.success_buf | success
    task.success = task.success_buf & time_out
    # ipdb.set_trace()
    task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] *= 1 - time_out.long()
    task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] += task.success
    #print(task.progress_buf, success, task.success_buf, time_out, task.success_queue.view(-1), task.success)
    #print(task.success_queue.shape)
    #task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] = task.success.float()
    task.success_rate = task.success_queue.mean(dim=-1)
    #task.current_success_rate = task.success_queue[:, task.success_idx]
    task.total_success_rate = task.success_rate.sum(dim=-1)
    task.success_entropy = - task.success_rate/(task.total_success_rate+1e-8) * torch.log(task.success_rate/(task.total_success_rate+1e-8) + 1e-8) * task.env_num
    #print(rot_reward[0])
    return task.rew_buf, task.reset_buf, dist_tip

def get_reward_done_(task):
    # what we use:
    # handle_pos: handle position array: the middle
    # task.hand_tip_pos: hand tip position array
    # franka_lfinger_pos: left finger position array 
    # franka_rfinger_pos: right finger position array
    # hand_down_dir: hand down rotation matrix
    part_pos = task.cabinet_part_rigid_body_tensor[:, :3]
    part_rot = task.cabinet_part_rigid_body_tensor[:, 3:7]
    hand_rot = task.hand_rigid_body_tensor[..., 3:7]
    hand_down_dir = quat_axis(hand_rot, 2)
    hand_grip_dir = quat_axis(hand_rot, 1)
    hand_sep_dir = quat_axis(hand_rot, 0)
    # print()
    handle_pos = quat_apply(
        part_rot, task.cabinet_handle_pos_tensor) + part_pos

    # print(quat_axis(part_rot, 0).shape)
    # print(task.cabinet_open_dir_tensor.view(-1, 1).shape)
    handle_x = quat_axis(part_rot, 0)\
         * task.cabinet_open_dir_tensor.view(-1, 1)
    handle_z = quat_axis(part_rot, 1)
    # cabinet_part_relative_o = part_pos + quat_apply(
    #     part_rot, task.cabinet_part_edge_min)
    # cabinet_part_relative_x = -handle_x
    # cabinet_part_relative_y = -quat_axis(part_rot, 2)
    # cabinet_part_relative_z = quat_axis(part_rot, 1)
    franka_lfinger_pos = task.rigid_body_tensor_used[:, task.hand_lfinger_rigid_body_index][:, 0:3]\
        + hand_down_dir*0.075
    franka_rfinger_pos = task.rigid_body_tensor_used[:, task.hand_rfinger_rigid_body_index][:, 0:3]\
        + hand_down_dir*0.075

        
    # basic reward
    reward = 0*torch.ones_like(task.rew_buf, device=task.device)
    scale = 1*torch.ones_like(task.rew_buf, device=task.device)
    stage_reward = 0*torch.ones_like(task.rew_buf, device=task.device)

    ## rotation reward old
    #dot1 = (hand_grip_dir * handle_z).sum(dim=-1)
    #dot2 = (-hand_sep_dir * handle_x).sum(dim=-1)
    #rot_reward = 0.5 * (torch.sign(dot1)*dot1**2 + torch.sign(dot2)*dot2**2)
    
    ##rotation reward new
    if task.cfg['task']['target_part_category'] == 'drawer':
        dot1 = (-hand_grip_dir * handle_z).sum(dim=-1)
        dot2 = (hand_sep_dir * handle_x).sum(dim=-1)
    else:
        dot1 = (hand_sep_dir * handle_z).sum(dim=-1)
        dot2 = (hand_grip_dir * handle_x).sum(dim=-1)
    #rot_reward = torch.minimum(torch.sign(dot1)*dot1**2 , torch.sign(dot2)*dot2**2) 
    rot_reward = 0.5 * (torch.sign(dot1)*dot1**2 + torch.sign(dot2)*dot2**2) 

    ## distance reward
    if not task.cfg["task"]["use_new_dist_rwd"]:
        d = torch.norm(task.hand_tip_pos - handle_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.1, dist_reward*2, dist_reward)
        dist_reward *= task.cabinet_have_handle_tensor
        dist_tip = torch.norm(task.hand_tip_pos - handle_pos, dim = -1)  ##
    else:
        # print(handle_pos.shape)
        # print(task.hand_tip_pos.shape)
        # exit(123)
        _draw_line(task, part_pos[0].cpu().numpy(), task.hand_tip_pos[0].cpu().numpy(), clear = True, env_id=0)
        dist_tip = torch.norm(task.hand_tip_pos - handle_pos, dim = -1)
        dist_lfinger = torch.norm(franka_lfinger_pos - handle_pos, dim = -1)
        dist_rfinger = torch.norm(franka_rfinger_pos - handle_pos, dim = -1)
        dist_mid = torch.norm(task.hand_mid_pos - handle_pos, dim = -1)
        
        rew_ee_mid_handle = (normalize_and_clip_in_interval(dist_mid, -0.01, 4E-3) - 1)
        
        if task.cfg["task"]["use_finger_dist"]:
            rew_ee_handle = -dist_tip
        else :
            rew_ee_handle = -((dist_lfinger+dist_rfinger)/2.) * 2
        if task.cfg["task"]["use_soft_dist"]:
            rew_ee_handle = (1.0 / (1.0 + rew_ee_handle ** 2)) - 1

    #reward for now 
    #reward = rot_reward + rew_ee_handle + rew_ee_mid_handle #- (dist_coefficient+vel_coefficient)
    #stage_reward = -(5 + vel_coefficient + dist_coefficient)

    rot_coefficient = task.cfg["task"]["rot_coef"]
    handle_coefficient = task.cfg["task"]["handle_coef"]
    vel_coefficient = task.cfg["task"]["vel_coef"]
    dist_coefficient = task.cfg["task"]["dist_coef"]
    tip_coefficient = task.cfg["task"]["tip_coef"]
    #print(rot_coefficient, vel_coefficient, dist_coefficient, tip_coefficient)
    #exit()
    # next_stage_reward
    ###close
    if task.cfg["task"]["target_state"] == "close" :
        task.rew_buf = -task.cabinet_dof_tensor[:, 0]
    ###open
    else: 
        stage_reward += 1.5 * (dist_tip <= 0.01) #tip distance 

        vel_reward = normalize_and_clip_in_interval(task.cabinet_dof_tensor[:, 1], -0.1, 0.5) * vel_coefficient  # Push vel to positive
        dist_reward = task.cabinet_dof_tensor[:, 0] * dist_coefficient #夹住handle然后打开门获得更高reward
        dist_reward *= ((dist_tip <= 0.01) * tip_coefficient + 1)
        #reward += dist_reward + vel_reward

        stage_reward += (vel_coefficient) * (task.cabinet_dof_tensor[:, 0] > task.cabinet_dof_target - 0.01) #if open enough
        #reward = reward + (- vel_reward) * (task.cabinet_dof_tensor[:, 0] >wo task.cabinet_dof_target - 0.01)

        stage_reward += 1*(task.cabinet_dof_tensor[:, 0] > task.cabinet_dof_target - 0.01)*(task.cabinet_dof_tensor[:, 1] < 0.01) #if open enough and stable 

        #task.rew_buf = (reward + stage_reward) * scale
        #task.rew_buf = rew_ee_handle + 10 * task.cabinet_dof_tensor[:, 0]  #
        task.rew_buf = rot_reward * rot_coefficient + rew_ee_handle * handle_coefficient \
                            + stage_reward + dist_reward
                            #vel_reward + (- vel_reward) * (task.cabinet_dof_tensor[:, 0] > task.cabinet_dof_target - 0.01)
        #print(task.cabinet_dof_tensor[:, 1])
        #print(rot_reward.mean(), rew_ee_handle.mean(), rew_ee_mid_handle.mean(), stage_reward.mean(), dist_reward.mean())#, vel_reward.mean())
        #print(task.cabinet_dof_tensor[:, 0])
        reward_info = {
            "hand rotation reward": rot_reward,
            "tip position reward": rew_ee_handle,
            "hand mid position reward": rew_ee_mid_handle,
            "stage reward": stage_reward,
            "cabinet position reward": dist_reward,
            "cabinet velocity reward": vel_reward,
            "total reward": reward,
        }

    # done info
    diff_from_success = (task.success_dof_states.view(task.cabinet_num, -1) - task.cabinet_dof_tensor_spec[:, :, 0]).view(-1)
    #print(task.cabinet_dof_tensor_spec[:, :, 0].shape)
    #print(task.cabinet_dof_tensor[:, 0].shape)
    #print(task.cabinet_dof_tensor_spec[0, :, 0] - task.cabinet_dof_tensor[:, 0])
    #exit()
    success = (diff_from_success < 0.01)
    time_out = (task.progress_buf >= task.max_episode_length)
    #print(task.progress_buf, task.max_episode_length)
    task.reset_buf = (task.reset_buf | time_out)
    task.success_buf = task.success_buf | success
    task.success = task.success_buf # & time_out
    #print(task.success.shape)
    #exit()
    # ipdb.set_trace()
    task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] *= 1 - time_out.long()
    task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] += task.success
    #print(task.progress_buf, success, task.success_buf, time_out, task.success_queue.view(-1), task.success)
    #print(task.success_queue.shape)
    #task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] = task.success.float()
    task.success_rate = task.success_queue.mean(dim=-1)
    #task.current_success_rate = task.success_queue[:, task.success_idx]
    task.total_success_rate = task.success_rate.sum(dim=-1)
    task.success_entropy = - task.success_rate/(task.total_success_rate+1e-8) * torch.log(task.success_rate/(task.total_success_rate+1e-8) + 1e-8) * task.env_num

    #print(task.reset_buf)

    return task.rew_buf, task.reset_buf, dist_tip  
