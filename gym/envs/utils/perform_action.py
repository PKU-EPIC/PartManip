
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
    quat_mul, tensor_clamp
from .misc import _draw_line
from .compute import *
import pytorch3d.transforms as pt
from utils.gym_info import clip_actions, clip_observations

def _perform_actions(task, actions):
    # print("..." ,actions)
    # actions
    task.actions = actions
    task.actions = 0
    task.pos_act[:] = 0
    task.eff_act[:] = 0
    task.vel_act[:] = 0

    if task.cfg["env"]["action_normalization"] == "clip":
        actions = torch.clamp(actions, -clip_actions, clip_actions)

    elif task.cfg["env"]["action_normalization"] == "tanh":
        actions = torch.tanh(actions)
        if "abs" in task.cfg["env"]["driveMode"]: #(-spacing , spacing)
            actions[:, :3] = actions[:, :3] * task.space_range + task.space_middle
        # elif task.cfg["env"]["driveMode"] == "pos":
        #     actions *= (task.franka_dof_upper_limits_tensor[:]- task.franka_dof_lower_limits_tensor[:]) 
        #     actions += task.franka_dof_lower_limits_tensor[:]
            
        if "axis_angle" in task.cfg["env"]["driveMode"]:
            actions[:, 3:6] = (actions[:, 3:6]) * torch.pi
        if "pos" not in task.cfg["env"]["driveMode"]:
            actions[:, -2:] = (actions[:, -2:]) * 0.05 
    
    # else if task.cfg["env"]["driveMode"] in ["ik_abs", "ik_rela"]:
    #     actions[:, :7] = 2 * (actions[:, 3:6] - 0.5) * torch.pi

    if task.cfg["env"]["driveMode"] == "pos_rela":
        task.pos_act[:, :-1] = task.franka_dof_tensor[:, :, 0] + actions[:, :] * task.dt * 20
        # task.pos_act[:, :-3] = task.pos_act[:, :-3] + actions[:, 0:9] * task.dt * 20
        task.pos_act[:, :-1] = tensor_clamp(
            task.pos_act[:, :-1], task.franka_dof_lower_limits_tensor[:], task.franka_dof_upper_limits_tensor[:])
    elif task.cfg["env"]["driveMode"] == "pos":
        actions *= (task.franka_dof_upper_limits_tensor[:]- task.franka_dof_lower_limits_tensor[:]) 
        actions += task.franka_dof_lower_limits_tensor[:]
        task.pos_act[:, :-1] = actions[:, :]

    elif task.cfg["env"]["driveMode"] == "effort":
        task.eff_act[:, :-3] = actions[:, :-1] * task.franka_dof_max_torque_tensor[:-2]
        task.eff_act[:, -3] = actions[:, 9] * task.franka_dof_max_torque_tensor[-2]
        task.eff_act[:, -2] = actions[:, 10] * task.franka_dof_max_torque_tensor[-1]
    elif task.cfg["env"]["driveMode"] == "maniskill":
        task.vel_act[:, :-3] = actions[:, :-2] * task.franka_dof_vel_upper_limits_tensor[:-2]
        task.pos_act[:, :-3] = task.franka_dof_tensor[:, :-2, 0] + actions[:, 0:9] * task.dt * 20
        # task.pos_act[:, -3:-1] = task.pos_act[:, -3:-1] + actions[:, -2:] * task.dt * 20
        task.pos_act[:, -3] = tensor_clamp(
            task.pos_act[:, -3], task.franka_dof_lower_limits_tensor[-2], task.franka_dof_upper_limits_tensor[-2])
        task.pos_act[:, -2] = tensor_clamp(
            task.pos_act[:, -2], task.franka_dof_lower_limits_tensor[-1], task.franka_dof_upper_limits_tensor[-1])
    elif task.cfg["env"]["driveMode"] == "ik":
        raise NotImplementedError
        dof_pos = task.franka_dof_tensor[:, :, 0]
        target_pos = actions[:, :3]*task.space_range + task.space_middle
        debug = False
        if debug:
            target_pos  =torch.ones((task.env_num, 3), device=task.device)
            target_pos[:,1] = 0.
            target_pos[:,0] = 0.
        pos_err = target_pos - task.hand_rigid_body_tensor[:, :3]
        target_rot = actions[:, 3:7] / torch.sqrt((actions[:, 3:7]**2).sum(dim=-1)+1e-8).view(-1, 1)
        rot_err = orientation_error(target_rot, task.hand_rigid_body_tensor[:, 3:7])
        
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        
        delta = control_ik(task.jacobian_tensor[:, task.hand_rigid_body_index - 1, :, :-2], task.device, dpose, task.num_envs)
        # t = time.time()
        # print(t - s)
        task.pos_act[:, :-1] = dof_pos.squeeze(-1)[:, :] + delta
        task.pos_act[:, -3:-1] = actions[:, -2:]/3.0*0.05

        # task.eff_act[:, -3] = actions[:, -2] * task.franka_dof_max_torque_tensor[-2]
        # task.eff_act[:, -2] = actions[:, -1] * task.franka_dof_max_torque_tensor[-1]
    elif task.cfg["env"]["driveMode"] == "ik_abs":
        dof_pos = task.franka_dof_tensor[:, :, 0]
        target_pos = actions[:, :3] #+ task.space_middle#/task.space_range[0][0]# + task.space_middle
        pos_err = target_pos - task.hand_rigid_body_tensor[:, :3]
        # print("pos_err",pos_err)
        target_rot = actions[:, 3:7] / torch.sqrt((actions[:, 3:7]**2).sum(dim=-1)+1e-8).view(-1, 1)
        rot_err = orientation_error(target_rot, task.hand_rigid_body_tensor[:, 3:7])
        
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(task.jacobian_tensor[:, task.hand_rigid_body_index - 1, :, :-2], task.device, dpose, task.num_envs)
        task.pos_act[:, :-3] = dof_pos.squeeze(-1)[:, :-2] + delta
        task.pos_act[:, -3:-1] = actions[:, -2:]

        # task.eff_act[:, -3] = actions[:, -2] * task.franka_dof_max_torque_tensor[-2]
        # task.eff_act[:, -2] = actions[:, -1] * task.franka_dof_max_torque_tensor[-1]
    elif task.cfg["env"]["driveMode"] == "ik_rela":
        dof_pos = task.franka_dof_tensor[:, :, 0]
        #target_pos = actions[:, :3] #+ task.space_middle#/task.space_range[0][0]# + task.space_middle
        #pos_err = target_pos - task.hand_rigid_body_tensor[:, :3]
        #print(task.cfg["seed"])
        pos_err = actions[:, :3]
        #pos_err = torch.clamp(actions[:, :3], - task.cfg["seed"]["clip_relative_actions"], task.cfg["seed"]["clip_relative_actions"])
        #pos_err[0] = torch.tensor([-0.05, 0.05, 0.05])
        #_draw_line(task, task.hand_rigid_body_tensor[0, :3].cpu(), (task.hand_rigid_body_tensor[0, :3] + torch.tensor([-0.5, 0.5, 0.5]).cuda()).cpu())
        target_rot = actions[:, 3:7] / torch.sqrt((actions[:, 3:7]**2).sum(dim=-1)+1e-8).view(-1, 1)
        rot_err = orientation_error(target_rot, task.hand_rigid_body_tensor[:, 3:7])
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(task.jacobian_tensor[:, task.hand_rigid_body_index - 1, :, :-2], task.device, dpose, task.num_envs)
        task.pos_act[:, :-3] = dof_pos.squeeze(-1)[:, :-2] + delta
        task.pos_act[:, -3:-1] = actions[:, -2:]
    elif task.cfg["env"]["driveMode"] == "ik_rela_axis_angle":

        dof_pos = task.franka_dof_tensor[:, :, 0]
        pos_err = actions[:, :3]
        #pos_err = torch.clamp(actions[:, :3], - task.cfg["seed"]["clip_relative_actions"], task.cfg["seed"]["clip_relative_actions"])
        target_rot = pt.matrix_to_quaternion(pt.axis_angle_to_matrix(actions[:, 3:6])) 
        #target_rot = actions[:, 3:7] / torch.sqrt((actions[:, 3:7]**2).sum(dim=-1)+1e-8).view(-1, 1)
        rot_err = orientation_error(target_rot, task.hand_rigid_body_tensor[:, 3:7])
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(task.jacobian_tensor[:, task.hand_rigid_body_index - 1, :, :-2], task.device, dpose, task.num_envs)
        task.pos_act[:, :-3] = dof_pos.squeeze(-1)[:, :-2] + delta
        task.pos_act[:, -3:-1] = actions[:, -2:]

    elif task.cfg["env"]["driveMode"] == "ik_abs_axis_angle":

        dof_pos = task.franka_dof_tensor[:, :, 0]
        target_pos = actions[:, :3]#+ task.space_middle#/task.space_range[0][0]# + task.space_middle
        pos_err = target_pos - task.hand_rigid_body_tensor[:, :3]
        target_rot = pt.matrix_to_quaternion(pt.axis_angle_to_matrix(actions[:, 3:6]))
        #target_rot = actions[:, 3:7] / torch.sqrt((actions[:, 3:7]**2).sum(dim=-1)+1e-8).view(-1, 1)
        rot_err = orientation_error(target_rot, task.hand_rigid_body_tensor[:, 3:7])
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(task.jacobian_tensor[:, task.hand_rigid_body_index - 1, :, :-2], task.device, dpose, task.num_envs)
        task.pos_act[:, :-3] = dof_pos.squeeze(-1)[:, :-2] + delta
        task.pos_act[:, -3:-1] = actions[:, -2:]

    else:
        print("No such DriveMode")
        raise NotImplementedError


    task.pos_act_all[task.dof_state_mask] = task.pos_act
    task.vel_act_all[task.dof_state_mask] = task.vel_act
    task.eff_act_all[task.dof_state_mask] = task.eff_act
    a = task.gym.set_dof_position_target_tensor(
       task.sim, gymtorch.unwrap_tensor(task.pos_act_all)
    )
    b = task.gym.set_dof_velocity_target_tensor(
       task.sim, gymtorch.unwrap_tensor(task.vel_act_all)
    )
    c = task.gym.set_dof_actuation_force_tensor(
       task.sim, gymtorch.unwrap_tensor(task.eff_act_all)
    )
   
def _get_qpos_from_hand_pose(task, hand_pose):
    #print("hand_pose", hand_pose)
    dof_pos = task.franka_dof_tensor[:, :, 0]
    #print("task.franka_dof_tensor[:, :, 0]")
    #print("dof_pos", task.franka_dof_tensor[:, :, 0])
    target_pos = hand_pose[:, :3] #+ task.space_middle#/task.space_range[0][0]# + task.space_middle
    pos_err = target_pos - task.hand_rigid_body_tensor[:, :3]
    # print("pos_err",pos_err)
    target_rot = hand_pose[:, 3:7] / torch.sqrt((hand_pose[:, 3:7]**2).sum(dim=-1)+1e-8).view(-1, 1)
    rot_err = orientation_error(target_rot, task.hand_rigid_body_tensor[:, 3:7])
    #print("task.hand_rigid_body_tensor", task.hand_rigid_body_tensor[:, :7])
    #print("task.jacobian_tensor", task.jacobian_tensor) #全0
    dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
    delta = control_ik(task.jacobian_tensor[:, task.hand_rigid_body_index - 1, :, :-2], task.device, dpose, task.num_envs)
    qpos = torch.zeros_like(task.franka_dof_tensor[:,:, 0])
    qpos[:,:-2] = dof_pos.squeeze(-1)[:, :-2] + delta
    # qpos[:,-2:] = hand_pose[:,-2:]

    return qpos
    # task.pos_act[:, :-3] = 
    # task.pos_act[:, -3:-1] = hand_pose[:, -2:]