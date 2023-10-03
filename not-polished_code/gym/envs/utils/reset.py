from lib2to3.pgen2.literals import simple_escapes
import numpy as np
import torch
from isaacgym import gymtorch
from .compute import *
from .perform_action import _get_qpos_from_hand_pose
from pytorch3d.transforms import quaternion_invert, matrix_to_quaternion
from .get_running_bbox import _draw_bbox_tensor, _draw_line

def _partial_reset(task, to_reset = "all") :
    
    """
    reset those need to be reseted
    """

    if to_reset == "all" :
        to_reset = np.ones((task.env_num,))
    reseted = False
    task.dof_state_tensor = None
    for env_id, reset in enumerate(to_reset) :
        # is reset:
        if reset.item() :
            # need randomization
            # reset_dof_states = task.initial_dof_states[env_id].clone()
            # reset_root_states = task.initial_root_states[env_id].clone()

            #franka_reset_pos_tensor = reset_root_states[0, :3]
            #franka_reset_rot_tensor = reset_root_states[0, 3:7]
            #franka_reset_dof_pos_tensor = reset_dof_states[:task.franka_num_dofs, 0]
            #franka_reset_dof_vel_tensor = reset_dof_states[:task.franka_num_dofs, 1]
            #cabinet_reset_pos_tensor = reset_root_states[1, :3]
            #cabinet_reset_rot_tensor = reset_root_states[1, 3:7]
            #cabinet_reset_dof_pos_tensor = reset_dof_states[task.franka_num_dofs:, 0]
            #cabinet_reset_dof_vel_tensor = reset_dof_states[task.franka_num_dofs:, 1]

            #cabinet_type = env_id // task.env_per_asset
            
            
            # task.intervaledRandom_(franka_reset_pos_tensor, task.franka_reset_position_noise * task.success_rate[env_id])
            # task.intervaledRandom_(franka_reset_rot_tensor, task.franka_reset_rotation_noise * task.success_rate[env_id])
            # task.intervaledRandom_(franka_reset_dof_pos_tensor, task.franka_reset_dof_pos_interval * task.success_rate[env_id], task.franka_dof_lower_limits_tensor, task.franka_dof_upper_limits_tensor)
            # task.intervaledRandom_(franka_reset_dof_vel_tensor, task.franka_reset_dof_vel_interval * task.success_rate[env_id])
            # task.intervaledRandom_(cabinet_reset_pos_tensor, task.cabinet_reset_position_noise * task.success_rate[env_id])
            # task.intervaledRandom_(cabinet_reset_rot_tensor, task.cabinet_reset_rotation_noise * task.success_rate[env_id])
            # task.intervaledRandom_(cabinet_reset_dof_pos_tensor, task.cabinet_reset_dof_pos_interval * task.success_rate[env_id], task.cabinet_dof_lower_limits_tensor[cabinet_type], task.cabinet_dof_upper_limits_tensor[cabinet_type])
            # task.intervaledRandom_(cabinet_reset_dof_vel_tensor, task.cabinet_reset_dof_vel_interval * task.success_rate[env_id])

            # #used
            #task.intervaledRandom_(franka_reset_pos_tensor, task.franka_reset_position_noise)
            #task.intervaledRandom_(franka_reset_rot_tensor, task.franka_reset_rotation_noise)
            #task.intervaledRandom_(franka_reset_dof_pos_tensor, task.franka_reset_dof_pos_interval, task.franka_dof_lower_limits_tensor, task.franka_dof_upper_limits_tensor)
            #task.intervaledRandom_(franka_reset_dof_vel_tensor, task.franka_reset_dof_vel_interval)
            #task.intervaledRandom_(cabinet_reset_pos_tensor, task.cabinet_reset_position_noise)
            #task.intervaledRandom_(cabinet_reset_rot_tensor, task.cabinet_reset_rotation_noise)
            #task.intervaledRandom_(cabinet_reset_dof_pos_tensor, task.cabinet_reset_dof_pos_interval, task.cabinet_dof_lower_limits_tensor[cabinet_type], task.cabinet_dof_upper_limits_tensor[cabinet_type])
            #task.intervaledRandom_(cabinet_reset_dof_vel_tensor, task.cabinet_reset_dof_vel_interval)
            # print(reset_dof_states.shape)
            # if task.dof_state_tensor == None:
            #     task.dof_state_tensor_ = ([i_ for i_ in j_ for j_ in ]).view(-1)
            #     task.dof_state_tensor_used, task.dof_state_mask, initial_env_dof_sate_list = deal_with_dof_state(task.dof_state_tensor_, task.part_dof_ids, task.cabinet_dof_num, task.franka_num_dofs, task.env_num, task.env_per_asset)
            # else: 
            #     task.dof_state_tensor_ = torch.cat((task.dof_state_tensor,reset_dof_states), dim = 0)
            #     task.dof_state_tensor_used, task.dof_state_mask, initial_env_dof_sate_list = deal_with_dof_state(task.dof_state_tensor_, task.part_dof_ids, task.cabinet_dof_num, task.franka_num_dofs, task.env_num, task.env_per_asset)
            # task.root_tensor[env_id].copy_(reset_root_states)

            reseted = True
            task.progress_buf[env_id] = 0
            task.reset_buf[env_id] = 0
            task.success_buf[env_id] = 0
            task.success_grasp_buf[env_id] = 0
            task.success_idx[env_id] = (task.success_idx[env_id] + 1) % task.success_queue.shape[1]
    
    # from isaacgym.gymtorch import wrap_tensor
    task.dof_state_tensor = task.initial_dof_states
    task.root_tensor = task.initial_root_states
    if reseted :
        # #print("before reset", task.dof_state_tensor)
        task.gym.refresh_jacobian_tensors(task.sim)
        # task.dof_state_tensor_all = wrap_tensor(task.gym.acquire_dof_state_tensor(task.sim))
        # task.rigid_body_tensor_all = wrap_tensor(task.gym.acquire_rigid_body_state_tensor(task.sim))
        # task.dof_state_tensor_used = task.dof_state_tensor_all[task.dof_state_mask]
        # task.rigid_body_tensor_used = task.rigid_body_tensor_all[task.rigid_state_mask]

        # task.hand_rigid_body_tensor = task.rigid_body_tensor_used\
        #     [:, task.hand_rigid_body_index, :] # N*13
        # task.franka_dof_tensor = task.dof_state_tensor_used\
        #     [:, :task.franka_num_dofs, :] # N*11*2
        # task.cabinet_dof_tensor = task.dof_state_tensor_used\
        #     [:, task.cabinet_dof_index, :] # N*2
        # task.cabinet_dof_tensor_spec = task.cabinet_dof_tensor.view(
        #     task.cabinet_num, task.env_per_asset, -1) # M*(N/M)*2
        # tmp_tensor = torch.arange(task.num_envs, device=task.device).long()
        # task.cabinet_base_rigid_body_tensor = task.rigid_body_tensor_used[
        #     tmp_tensor, task.env_base_rigid_id_list.long()]
        # task.cabinet_part_rigid_body_tensor = task.rigid_body_tensor_used[
        #     tmp_tensor, task.env_part_rigid_id_list.long()]
        # task.cabinet_handle_rigid_body_tensor = task.rigid_body_tensor_used[
        #     tmp_tensor, task.env_handle_rigid_id_list.long()]

        # task.cabinet_handle_pos_tensor = task.cabinet_handle_rigid_body_tensor[:,:3]
        # # #print(task.dof_state_tensor)
        # if task.cfg["env"]["usePreGraspPose"] and task.need_set_init_pose:
        #     task.need_set_init_pose = False
        #     if  task.cfg["env"]["useGtBbox"]:
        #         part_bbox_tensor = task.part_bbox_tensor
        #         handle_bbox_tensor = task.handle_bbox_tensor
        #         #_draw_bbox_tensor(task, handle_bbox_tensor)
        #     else:
        #         raise NotImplementedError
            
        #     part_pos = task.cabinet_part_rigid_body_tensor[:, :3]
        #     part_rot = task.cabinet_part_rigid_body_tensor[:, 3:7]
        #     hand_rot = task.hand_rigid_body_tensor[..., 3:7]
        #     handle_out = task.handle_bbox_tensor[:, 0] - task.handle_bbox_tensor[:, 4]
        #     handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
        #     handle_long = task.handle_bbox_tensor[:, 1] - task.handle_bbox_tensor[:, 0]
        #     handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
        #     handle_short = task.handle_bbox_tensor[:, 3] - task.handle_bbox_tensor[:, 0]
        #     handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
        #     hand_pose = torch.zeros((handle_out.shape[0], 7), device = task.device)
        #     rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
        #     #print("qpose_before")
        #     #print(_get_qpos_from_hand_pose(task, hand_pose))
        #     hand_pose[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.5, 0, 0], device = task.device)
        #     hand_pose[:,3:7] = rotations
        #     # print("rotations", rotations)
        #     # hand_pose[:, :3] = torch.tensor([1, 0, 1], device = task.device)
        #     # hand_pose[:, 3:7] = torch.tensor([1, 0, 0, 0], device=task.device)
        #     #print("qpose_after")
            
        #     initial_tensor = torch.zeros((task.pos_act_all.shape[0],2), device = task.device)
        #     a = initial_tensor[task.dof_state_mask]
            
        #     a.view(task.env_num, -1, 2)[...,:-1,0] = _get_qpos_from_hand_pose(task, hand_pose)#[:,:-2]# = torch.zeros((hand_pose.shape[0], task.franka_num_dofs))
        #     #print(a.view(task.env_num, -1, 2)[...,:-1,0])
        #     for i in range(a.shape[0]):
        #         initial_tensor[task.dof_state_mask[i]] = a[i]
        #     # print(a)
        #     # print(initial_tensor[task.dof_state_mask])
        #     # exit()
        #     #print(_get_qpos_from_hand_pose(task, hand_pose))
        #     # print(task.franka_dof_tensor.shape)
        #     # print(task.cabinet_dof_tensor.shape)
        #     # initial_tensor[:,:-2] = _get_qpos_from_hand_pose(task, hand_pose)
        #     # task.env_num = 1
        #     #print(initial_tensor)
        #     pre = task.gym.set_dof_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(initial_tensor), gymtorch.unwrap_tensor(torch.arange(task.env_num, device = task.device, dtype = torch.int32)*2), task.env_num,)
        #     #print("pre", pre)

        #     #env_ptr = task.env_ptr_list[0]
        #     #task.gym.set_actor_scale(env_ptr, task.franka_actor, 1)
        #     task.initial_dof_states = initial_tensor
        #     task.dof_state_tensor = task.initial_dof_states
    
        #     #task.gym.set_actor_scale(env_ptr, task.franka_actor, task.cfg["env"]["franka_scale"])

        #print(task.initial_dof_states)
        #print("before reset", wrap_tensor(task.gym.acquire_dof_state_tensor(task.sim)))
        
        #print(task.initial_dof_states_plan)
        #task.gym.set_dof_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task.initial_dof_states_plan), gymtorch.unwrap_tensor(torch.arange(task.env_num, device = task.device, dtype = torch.int32)*2), task.env_num,)
        # task.gym.simulate(task.sim)
        # task.gym.fetch_results(task.sim, True)
        
        # if not task.headless :
        #     task.render()
        # if task.cfg["env"]["enableCameraSensors"] == True:
        #     task.gym.step_graphics(task.sim)
        # task._refresh_observation()
        # if reseted :
        task.gym.set_dof_state_tensor(
            task.sim,
            gymtorch.unwrap_tensor(task.dof_state_tensor)
        )
        task.dof_state_tensor = task.initial_dof_states
        task.root_tensor = task.initial_root_states
        task.gym.set_actor_root_state_tensor(
            task.sim,
            gymtorch.unwrap_tensor(task.root_tensor)
        )

    #exit()

def reset(task, to_reset = "all") :
    #print("reset")
    
    task._partial_reset(to_reset)

    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    if not task.headless :
        task.render()
    if task.cfg["env"]["enableCameraSensors"] == True:
        task.gym.step_graphics(task.sim)
    
    task._refresh_observation()
    success = task.success.clone()
    reward, done ,dist_tip, grasp_success_rate = task._get_reward_done()

    ###在这里记录log的信息, 不会因为reset丢失
    
    task.extras["successes"] = success
    task.extras["success_rate"] = task.success_rate
    task.extras["success_entropy"] = task.success_entropy
    if task.cfg["task"]["task"] == 'open':
        task.extras["grasp_success_rate"] = grasp_success_rate
    #task.extras["current_success_rate"] = task.current_success_rate
    task.extras["dist_tip"] = dist_tip
    #task.extras["part_position"] = task.cabinet_dof_tensor[:, 0]
    return task.obs_buf, task.rew_buf, task.reset_buf, None