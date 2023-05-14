from random import shuffle
from utils.gym_info import *
from pathlib import Path
import torch
from os.path import join as pjoin
import numpy as np
from isaacgym import gymtorch
import json
from scipy.spatial.transform import Rotation as R
from .perform_action import _get_qpos_from_hand_pose
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert
from utils.gym_info import cam1_pos, cam1_rot, cam2_pos,cam2_rot,cam3_pos, cam3_rot
import os, cv2

def _place_agents(task, env_num, spacing, use_cam):
    print("Simulation: creating agents")

    lower = gymapi.Vec3(-spacing, -spacing, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    task.space_middle = torch.zeros((env_num, 3), device=task.device)
    task.space_range = torch.zeros((env_num, 3), device=task.device)
    task.space_middle[:, 0] = task.space_middle[:, 1]= task.space_middle[:, 2] = 0
    # task.space_middle[:, 2] = spacing/2
    task.space_range[:, 0] = task.space_range[:, 1] = task.space_range[:, 2] = spacing
    # task.space_middle[:, 2] = spacing/2
    num_per_row = int(np.sqrt(env_num))

    from rich.progress import Progress
    with Progress() as progress:
        task._load_obj_asset(progress)
        print("finish loading objects")
        task1 = progress.add_task('[red]Creating envs:', total=env_num)
        for i, env_id in enumerate(range(env_num)):
            env_ptr = task.gym.create_env(task.sim, lower, upper, num_per_row)
            print(i)
            task._load_franka(env_ptr, env_id)
            task._load_obj(env_ptr, env_id)
            task.env_ptr_list.append(env_ptr)
            if use_cam or task.cfg["save_video"]:
                task._load_cam(env_ptr, env_id)
            progress.update(task1, advance=1)
        task.cabinet_open_dir_tensor = task.cabinet_open_dir_tensor\
        .repeat_interleave(task.env_per_asset, dim=0)

def _create_ground_plane(task):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.,0.,1.)
    plane_params.static_friction = plane_params_static_friction
    plane_params.dynamic_friction = plane_params_dynamic_friction
    task.gym.add_ground(task.sim, plane_params)

def _load_obj(task, env_ptr, env_id):
    cabinet_type = env_id // task.env_per_asset

    subenv_id = env_id % task.env_per_asset

    object_init_pose = gymapi.Transform()
    object_init_pose.p = object_init_pose_p
    object_init_pose.r = object_init_pose_r
    obj_actor = task.gym.create_actor(
        env_ptr,
        task.cabinet_asset_list[cabinet_type],
        object_init_pose,
        "cabinet{}-{}".format(cabinet_type, subenv_id),
        env_id,
        1, 0
    )

    cabinet_dof_props = task.gym.get_asset_dof_properties(task.cabinet_asset_list[cabinet_type])
    #keys:
    cabinet_dof_props['stiffness'][0] = 20.0 ##刚性系数
    cabinet_dof_props['damping'][0] = 200    ##阻尼系数
    cabinet_dof_props['friction'][0] = 5  ##摩擦系数
    if task.cfg["task"]["task"] == "open":
        cabinet_dof_props['effort'][0] = 0.1 # TODO raw 4.0 ??
    elif task.cfg["task"]["task"] == "close": 
        cabinet_dof_props['effort'][0] = -0.1

    if not task.cfg["env"]["usePartRecoverForce"]:
        cabinet_dof_props["driveMode"][0] = gymapi.DOF_MODE_NONE
    else:
        cabinet_dof_props["driveMode"][0] = gymapi.DOF_MODE_POS

    task.gym.set_actor_dof_properties(env_ptr, obj_actor, cabinet_dof_props)
    task.cabinet_actor_list.append(obj_actor)

def _load_obj_asset(task, progress):
    # task.cabinet_asset_name_list = []
    # task.cabinet_asset_list = []
    # task.cabinet_pose_list = []
    # task.cabinet_target_link_id = []

    train_len = min(task.cabinet_num_train, task.total_train_asset)
    valIntra_len = min(task.cabinet_num_valIntra, task.total_valIntra_asset)
    valInter_len = min(task.cabinet_num_valInter, task.total_valInter_asset)
    used_len = train_len + valIntra_len + valInter_len


    
    select_train_asset = [i for i in range(task.total_train_asset)]
    select_valIntra_asset = [i for i in range(task.total_valIntra_asset)]
    select_valInter_asset = [i for i in range(task.total_valInter_asset)]


    if task.cfg["env"]["asset"]["randomAsset"]:
        shuffle(select_train_asset)
        shuffle(select_valIntra_asset)
        shuffle(select_valInter_asset)

    select_train_asset = select_train_asset[:train_len]
    select_valIntra_asset = select_valIntra_asset[:valIntra_len]
    select_valInter_asset = select_valInter_asset[:valInter_len]


    task.select_train_asset_paths = [
        task.total_train_asset_paths[i] for i in select_train_asset]
    task.select_train_asset_linkname = [
        task.total_cabinet_train_linkname_list[i] for i in select_train_asset]
    if task.target_part in ["door", "drawer", "handle"]:
        task.select_train_asset_handlename = [
            task.total_cabinet_train_handlename_list[i] for i in select_train_asset]
    task.select_train_asset_jointname = [
        task.total_cabinet_train_jointname_list[i] for i in select_train_asset]
    if task.target_part in ["door", "drawer", "handle"]:
        task.select_train_asset_handlejoint = [
            task.total_cabinet_train_handlejoint_list[i] for i in select_train_asset]

    task.select_valIntra_asset_paths = [
        task.total_valIntra_asset_paths[i] for i in select_valIntra_asset]
    task.select_valIntra_asset_linkname = [
        task.total_cabinet_valIntra_linkname_list[i] for i in select_valIntra_asset]
    if task.target_part in ["door", "drawer", "handle"]:
        task.select_valIntra_asset_handlename = [
            task.total_cabinet_valIntra_handlename_list[i] for i in select_valIntra_asset]
    task.select_valIntra_asset_jointname = [
    task.total_cabinet_valIntra_jointname_list[i] for i in select_valIntra_asset]
    if task.target_part in ["door", "drawer", "handle"]:
        task.select_valIntra_asset_handlejoint = [
            task.total_cabinet_valIntra_handlejoint_list[i] for i in select_valIntra_asset]

    task.select_valInter_asset_paths = [
        task.total_valInter_asset_paths[i] for i in select_valInter_asset]
    task.select_valInter_asset_linkname = [
        task.total_cabinet_valInter_linkname_list[i] for i in select_valInter_asset]
    if task.target_part in ["door", "drawer", "handle"]:
        task.select_valInter_asset_handlename = [
            task.total_cabinet_valInter_handlename_list[i] for i in select_valInter_asset]
    task.select_valInter_asset_jointname = [
        task.total_cabinet_valInter_jointname_list[i] for i in select_valInter_asset]
    if task.target_part in ["door", "drawer", "handle"]:
        task.select_valInter_asset_handlejoint = [
            task.total_cabinet_valInter_handlejoint_list[i] for i in select_valInter_asset]
    
    task.selected_asset_path_list = task.select_train_asset_paths + task.select_valIntra_asset_paths\
        + task.select_valInter_asset_paths
    task.selected_asset_linkname_list = task.select_train_asset_linkname  + task.select_valIntra_asset_linkname\
        + task.select_valInter_asset_linkname
    if task.target_part in ["door", "drawer", "handle"]:
        task.selected_asset_handlename_list = task.select_train_asset_handlename  + task.select_valIntra_asset_handlename\
            + task.select_valInter_asset_handlename
    task.selected_asset_jointname_list = task.select_train_asset_jointname  + task.select_valIntra_asset_jointname\
        + task.select_valInter_asset_jointname
    if task.target_part in ["door", "drawer", "handle"]:
        task.selected_asset_handlejoint_list = task.select_train_asset_handlejoint  + task.select_valIntra_asset_handlejoint\
            + task.select_valInter_asset_handlejoint

    task.cabinet_asset_list = []
    task.cabinet_dof_num = []

    task.part_rigid_ids = []
    if task.target_part in ["door", "drawer", "handle"]:
        task.handle_rigid_ids = []
    task.part_dof_ids = []
    task.base_rigid_ids = []
    task.rigid_num_list = []
    task.cabinet_dof_lower_limits_tensor = []
    task.cabinet_dof_upper_limits_tensor = []
    task.part_bbox = []
    if task.target_part in ["door", "drawer", "handle"]:
        task.handle_bbox = []
    task.part_axis_xyz = []
    task.part_axis_dir = []
    part_pt_bbox_list = []
    if task.target_part in ["door", "drawer", "handle"]:
        handle_pt_bbox_list = []
    task.pseudo_pc = []
    task.pseudo_pc_part_mask = []
    task.pseudo_pc_handle_mask = []
    task.pseudo_pc_base_mask = []


    asset_root_parent = str(Path(".").parent.absolute())
    
    # print(task.selected_asset_path_list)

    task2 = progress.add_task('[green]loading object...', total=used_len)
    for i_, path in enumerate(task.selected_asset_path_list):
        print(path)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = asset_options_fix_base_link_object
        asset_options.disable_gravity = asset_options_disable_gravity_object
        asset_options.collapse_fixed_joints = asset_options_collapse_fixed_joints_object
        cabinet_asset = task.gym.load_asset(task.sim, asset_root_parent, \
            pjoin(str(path), "mobility_new.urdf"), asset_options)
        asset_options.convex_decomposition_from_submeshes = True

        path_bbox = str(path) + "/bbox_info.json"
        with open(path_bbox, "r") as f:
            bbox_info = json.load(f)
        
        progress.update(task2, advance=1)
        task.cabinet_asset_list.append(cabinet_asset)
        dof_dict = task.gym.get_asset_dof_dict(cabinet_asset)
        # max_torque, lower_limits, upper_limits = task._get_dof_property(cabinet_asset)
        dof_props = task.gym.get_asset_dof_properties(cabinet_asset)
        dof_num = task.gym.get_asset_dof_count(cabinet_asset)
        rig_names = task.gym.get_asset_rigid_body_dict(cabinet_asset)

        joint_dict = task.gym.get_asset_joint_dict(cabinet_asset)
        
        # import pdb
        # pdb.set_trace()

        task.cabinet_dof_lower_limits_tensor.append(dof_props["lower"][dof_dict[task.selected_asset_jointname_list[i_]]])
        task.cabinet_dof_upper_limits_tensor.append(dof_props["upper"][dof_dict[task.selected_asset_jointname_list[i_]]])

        task.part_rigid_ids.append(rig_names[task.selected_asset_linkname_list[i_]])
        if task.target_part in ["door", "drawer", "handle"]:
            task.handle_rigid_ids.append(rig_names[task.selected_asset_handlename_list[i_]])
        task.part_dof_ids.append(dof_dict[task.selected_asset_jointname_list[i_]])
        task.base_rigid_ids.append(rig_names["base"])

        task.part_bbox.append(np.array(bbox_info['bbox_world'][bbox_info['link_name'].index(task.selected_asset_linkname_list[i_])]).astype(np.float32))
        if task.target_part in ["door", "drawer", "handle"]:
            task.handle_bbox.append(np.array(bbox_info['bbox_world'][bbox_info['link_name'].index(task.selected_asset_handlename_list[i_])]).astype(np.float32))
        task.part_axis_xyz.append(np.array(bbox_info['axis_xyz_world'][bbox_info['link_name'].index(task.selected_asset_linkname_list[i_])]).astype(np.float32))
        task.part_axis_dir.append(np.array(bbox_info['axis_dir_world'][bbox_info['link_name'].index(task.selected_asset_linkname_list[i_])]).astype(np.float32))
        
        task.cabinet_target_joint_lower_limits_tensor[i_,:] = torch.tensor(dof_props["lower"][task.part_dof_ids[i_]])
        task.cabinet_target_joint_upper_limits_tensor[i_,:] = torch.tensor(dof_props["upper"][task.part_dof_ids[i_]])

        task.cabinet_open_dir_tensor[i_] = 1 # TODO wrong, need to modify!!!!!!!!!!!!

        task.cabinet_dof_num.append(dof_num)
        task.rigid_num_list.append(len(rig_names))

        ####load pretrained part bbox and handle bbox
        if task.cfg["obs"]["state"]["use_bbox_type"] == "pt":
            name = str(path) +"/perception_info.json"
            with open(name, "r") as read_content:
                data = json.load(read_content)
                #data["part_pt_bbox"], data["handle_pt_bbox"]   ###list
                for _ in range(task.env_per_asset):
                    part_pt_bbox_list.append(data["part_pt_bbox"])
                    if task.target_part in ["door", "drawer", "handle"]:
                        handle_pt_bbox_list.append(data["handle_pt_bbox"])
                
            task.part_pt_bbox_tensor = torch.tensor(part_pt_bbox_list, device=task.device, dtype=torch.float32)
            if task.target_part in ["door", "drawer", "handle"]:
                task.handle_pt_bbox_tensor = torch.tensor(handle_pt_bbox_list, device=task.device, dtype=torch.float32)
            #####change the permute of bbox point 
            
            handle_permute_list = [3, 5, 7, 6, 0, 1, 4, 2]
            task.part_pt_bbox_tensor = task.part_pt_bbox_tensor[:, handle_permute_list]
            if task.target_part in ["door", "drawer", "handle"]:
                task.handle_pt_bbox_tensor = task.handle_pt_bbox_tensor[:, handle_permute_list]

        # pdb.set_trace()
        if task.cfg["obs"]["pc"]["use_pseudo_pc"]:
            obj_name = str(path).split("/")[-1]
            pc_path = task.cfg["obs"]["pc"]["pseudo_pc_root"]+"/"+obj_name + "_pc_data.pth"

            # raw_data = torch.load(pc_path)

            # pc_data = raw_data.cpu().numpy()
            # del raw_data
            pc_data = torch.load(pc_path)
            pc = pc_data[0,:,:6]
            pc_part_ids = (pc_data[0,:,6] == 105)
            if task.target_part in ["door", "drawer", "handle"]:
                pc_handle_ids = (pc_data[0,:,6] == 106)
            # pc_part = pc[pc_part_ids]
            # pc_handle = pc[pc_handle_ids]
            # pc_base = pc[~(pc_part_ids|pc_handle_ids)]
            task.pseudo_pc.append(pc)
            task.pseudo_pc_part_mask.append(pc_part_ids)
            if task.target_part in ["door", "drawer", "handle"]:
                task.pseudo_pc_handle_mask.append(pc_handle_ids)
            task.pseudo_pc_base_mask.append(~(pc_part_ids|pc_handle_ids))
            
    if task.cfg["obs"]["pc"]["use_pseudo_pc"]:
        task.pseudo_pc_init = torch.tensor(np.array(task.pseudo_pc), device=task.device).repeat_interleave(task.env_per_asset, dim=0)

        task.pseudo_pc_part_mask = torch.tensor(np.array(task.pseudo_pc_part_mask), device=task.device).repeat_interleave(task.env_per_asset, dim=0)
        if task.target_part in ["door", "drawer", "handle"]:
            task.pseudo_pc_handle_mask = torch.tensor(np.array(task.pseudo_pc_handle_mask), device=task.device).repeat_interleave(task.env_per_asset, dim=0)
        task.pseudo_pc_base_mask = torch.tensor(np.array(task.pseudo_pc_base_mask), device=task.device).repeat_interleave(task.env_per_asset, dim=0)

    task.part_bbox_tensor_init = torch.tensor(np.array(task.part_bbox).astype(np.float32), device=task.device, dtype=torch.float32).repeat_interleave(task.env_per_asset, dim=0)
    if task.target_part in ["door", "drawer", "handle"]:
        task.handle_bbox_tensor_init = torch.tensor(np.array(task.handle_bbox).astype(np.float32), device=task.device, dtype=torch.float32).repeat_interleave(task.env_per_asset, dim=0)
    task.part_axis_xyz_tensor_init = torch.tensor(np.array(task.part_axis_xyz).astype(np.float32), device = task.device).repeat_interleave(task.env_per_asset, dim=0)
    task.part_axis_dir_tensor_init = torch.tensor(np.array(task.part_axis_dir).astype(np.float32), device = task.device).repeat_interleave(task.env_per_asset, dim=0)
    task.object_init_pose_p_tensor = torch.tensor(object_init_pose_p_np.astype(np.float32), device = task.device)
    task.object_init_pose_r_matrix_tensor = torch.tensor(R.from_quat(object_init_pose_r_np).as_matrix().astype(np.float32),device = task.device)
    task.env_base_rigid_id_list = torch.tensor(task.base_rigid_ids, device = task.device).repeat_interleave(task.env_per_asset, dim=0)
    task.env_part_rigid_id_list = torch.tensor(task.part_rigid_ids, device = task.device).repeat_interleave(task.env_per_asset, dim=0)
    if task.target_part in ["door", "drawer", "handle"]:
        task.env_handle_rigid_id_list = torch.tensor(task.handle_rigid_ids, device = task.device).repeat_interleave(task.env_per_asset, dim=0)
    task.env_part_dof_ids_list = torch.tensor(task.part_dof_ids, device = task.device).repeat_interleave(task.env_per_asset, dim=0)
    task.cabinet_dof_num_tensor = torch.tensor(task.cabinet_dof_num, device = task.device)

def _load_franka(task, env_ptr, env_id):
    if task.franka_loaded == False:
        asset_root = task.asset_root
        asset_file = task.franka_file
        asset_options = gymapi.AssetOptions()
        if task.cfg["env"]["asset"]["new_setting"]:
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.thickness = 0.001
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
            asset_options.use_mesh_materials = True
            
            # asset_options.flip_visual_attachments = asset_options_flip_visual_attachments
            # asset_options.fix_base_link = asset_options_fix_base_link
            # asset_options.disable_gravity = asset_options_disable_gravity
            # # Switch Meshes from Z-up left-handed system to Y-up -\
            # # Right-handed coordinate system.
            # # import pdb
            # # pdb.set_trace()
            # asset_options.armature = asset_options_armature
        else:
            
            asset_options.flip_visual_attachments = asset_options_flip_visual_attachments
            asset_options.fix_base_link = asset_options_fix_base_link
            asset_options.disable_gravity = asset_options_disable_gravity
            # Switch Meshes from Z-up left-handed system to Y-up -\
            # Right-handed coordinate system.
            # import pdb
            # pdb.set_trace()
            asset_options.armature = asset_options_armature
            # asset_options.convex_decomposition_from_submeshes = True
            # asset_options.vhacd_enabled = True
            # asset_options.vhacd_params = gymapi.VhacdParams()
            # asset_options.vhacd_params.resolution = 50000
            # asset_options.vhacd_params.max_convex_hulls = 10
            # asset_options.vhacd_params.max_num_vertices_per_ch = 64
        task.franka_asset = task.gym.load_asset(task.sim, asset_root, 
            asset_file, asset_options)

        task.franka_loaded = True
    
    ######### The problem happened here! ##############
    franka_dof_max_torque, franka_dof_lower_limits, franka_dof_upper_limits, franka_dof_vel_upper_limits_tensor\
            = _get_dof_property(task, task.franka_asset)
    task.franka_dof_max_torque_tensor = torch.tensor(franka_dof_max_torque, device=task.device)
    task.franka_dof_upper_limits_tensor = torch.tensor(franka_dof_upper_limits, device=task.device)
    task.franka_dof_lower_limits_tensor = torch.tensor(franka_dof_lower_limits, device=task.device)
    task.franka_dof_lower_limits_tensor = torch.tensor(franka_dof_lower_limits, device=task.device)
    task.franka_dof_upper_limits_tensor = torch.tensor(franka_dof_upper_limits, device=task.device)
    task.franka_dof_vel_upper_limits_tensor = torch.tensor(franka_dof_vel_upper_limits_tensor, device = task.device)

    task.franka_dof_mean_limits_tensor = (franka_dof_lower_limits + franka_dof_upper_limits)/2
    task.franka_dof_limits_range_tensor = (franka_dof_upper_limits - franka_dof_lower_limits)/2
    ######### The problem happened here! ##############
    
    ######### The problem happened here! raw version ##############
    # franka_dof_max_torque, franka_dof_lower_limits, franka_dof_upper_limits, franka_dof_vel_upper_limits_tensor\
    #         = _get_dof_property(task, task.franka_asset)
    # task.franka_dof_max_torque_tensor = torch.tensor(
    #     franka_dof_max_torque, device=task.device)
    # task.franka_dof_mean_limits_tensor = torch.tensor(
    #     (franka_dof_lower_limits + franka_dof_upper_limits)/2, device=task.device)
    # task.franka_dof_limits_range_tensor = torch.tensor(
    #     (franka_dof_upper_limits - franka_dof_lower_limits)/2, device=task.device)
    # task.franka_dof_lower_limits_tensor = torch.tensor(
    #     franka_dof_lower_limits, device=task.device)

    # task.franka_dof_upper_limits_tensor = torch.tensor(
    #     franka_dof_upper_limits, device=task.device)
    # task.franka_dof_vel_upper_limits_tensor = torch.tensor(
    #     franka_dof_vel_upper_limits_tensor, device = task.device)
    ######### The problem happened here! raw version##############
    
    
    dof_props = task.gym.get_asset_dof_properties(task.franka_asset)
    # use position drive for all dofs
    if task.cfg["env"]["asset"]["new_setting"]:
        dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"][:].fill(400.0)
        dof_props["damping"][:].fill(80.0)
        # dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"][-2:].fill(1.0e6)
        dof_props["damping"][-2:].fill(1.0e2)
    elif task.cfg["env"]["driveMode"] in ["pos", "ik"] or "ik" in task.cfg["env"]["driveMode"]:
        dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"][:].fill(400.0)
        dof_props["damping"][:].fill(40.0)
        # dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"][-2:].fill(0)
        dof_props["damping"][-2:].fill(0)
    elif task.cfg["env"]["driveMode"] == "maniskill":
        dof_props["driveMode"][:-2].fill(gymapi.DOF_MODE_VEL)
        dof_props["stiffness"][:-2].fill(400.0)
        dof_props["damping"][:-2].fill(40.0)
        dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"][-2:].fill(400.0)
        dof_props["damping"][-2:].fill(40.0)
    else:       # osc
        dof_props["driveMode"][:-2].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"][:-2].fill(400.0)
        dof_props["damping"][:-2].fill(40.0)
        dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"][-2:].fill(400.0)
        dof_props["damping"][-2:].fill(40.0)

    # root pose
    initial_franka_pose = gymapi.Transform()
    initial_franka_pose.r = initial_franka_pose_r
    if task.cfg["task"]["target_state"] == "close" or task.cfg["task"]["task"] == "push_button":
        task.cabinet_dof_coef = -1.0
        task.success_dof_states = task.cabinet_target_joint_lower_limits_tensor[:, 0].clone()
        if task.target_part == "drawer":
            initial_franka_pose.p = initial_franka_pose_p_close_drawer
        else: #task.target_part == "door":
            initial_franka_pose.p = initial_franka_pose_p_close_door 
    else :  # open
        if task.target_part == "drawer":
            initial_franka_pose.p = initial_franka_pose_p_open_drawer
        else: #task.target_part == "door":
            initial_franka_pose.p = initial_franka_pose_p_open_door
    # set start dof
    task.franka_num_dofs = task.gym.get_asset_dof_count(task.franka_asset)
    task.franka_rigid_num = len(task.gym.get_asset_rigid_body_dict(task.franka_asset))
    default_dof_pos = np.zeros(task.franka_num_dofs, dtype=np.float32)

    default_dof_pos[:-2] = (franka_dof_lower_limits + franka_dof_upper_limits)[:-2] * 0.3
    # grippers open
    default_dof_pos[-2:] = franka_dof_upper_limits[-2:]
    franka_dof_state = np.zeros_like(franka_dof_max_torque, gymapi.DofState.dtype)
    franka_dof_state["pos"] = default_dof_pos
    
    if task.cfg["env"]["usePreGraspPose"]:
        cabinet_type = env_id // task.env_per_asset

        # if "Slider" in task.cfg["env"]["robotName"]:
        # import pdb
        # pdb.set_trace()
        if task.cfg["task"]["use_part_pregrasp"]:
            path = str(task.selected_asset_path_list[cabinet_type]) + f"/part_pregrasp_dof_state.npy"
            state = np.load(path,allow_pickle=True)
            franka_dof_state["pos"] = (state[:-1, 0])
        else:
            path = str(task.selected_asset_path_list[cabinet_type]) + f"/handle_pregrasp_dof_state.npy"
            state = np.load(path,allow_pickle=True)
            franka_dof_state["pos"] = state[:-1, 0]
        # elif "fixed" in task.cfg["env"]["robotName"]:
        #     pass


    franka_actor = task.gym.create_actor(
        env_ptr,task.franka_asset, initial_franka_pose,"franka",env_id,2,0)
    
    shape_props = task.gym.get_actor_rigid_shape_properties(env_ptr, franka_actor)

    # exit(123)
    # rigid props
    # franka_shape_props = task.gym.get_actor_rigid_shape_properties(env_ptr, franka_actor)
    # for shape in franka_shape_props :
    #     shape.friction = 100
    
    # task.gym.set_actor_rigid_shape_properties(env_ptr, franka_actor, franka_shape_props)
    task.gym.set_actor_dof_properties(env_ptr, franka_actor, dof_props)
    task.gym.set_actor_dof_states(env_ptr, franka_actor, franka_dof_state, gymapi.STATE_ALL)
    #franka_scaling = True
    #if franka_scaling: 
    franka_scale = task.cfg["env"]["franka_scale"]
    task.gym.set_actor_scale(env_ptr, franka_actor, franka_scale)  #################################
    task.franka_actor = franka_actor

def _get_dof_property(task, asset):
    dof_props = task.gym.get_asset_dof_properties(asset)

    dof_num = task.gym.get_asset_dof_count(asset)
    dof_lower_limits = []
    dof_upper_limits = []
    dof_max_torque = []
    dof_max_vel = []
    for i in range(dof_num):
        dof_max_torque.append(dof_props["effort"][i])
        dof_lower_limits.append(dof_props["lower"][i])
        dof_upper_limits.append(dof_props["upper"][i])
        dof_max_vel.append(dof_props["velocity"][i])
    dof_max_torque = np.array(dof_max_torque)
    dof_lower_limits = np.array(dof_lower_limits)
    dof_upper_limits = np.array(dof_upper_limits)
    dof_max_vel = np.array(dof_max_vel)
    return dof_max_torque, dof_lower_limits, dof_upper_limits, dof_max_vel

def _load_cam(task, env_ptr, env_id):
    
    if task.cfg["save_video"]:
        video_save_path = task.video_save_root + "/" + "{}/{}.mp4".format(task.cfg["task"]["target_part_category"], env_id)
        video_file = os.path.join(video_save_path)
        video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 25,
                                                (task.image_shape[1], task.image_shape[0]))
        task.video_file_list.append(video_file)
        task.video_writer_list.append(video_writer)

        video_camera_handle = task.gym.create_camera_sensor(env_ptr, task.video_camera_props)
        # set on the front and look towards bottom
        task.gym.set_camera_location(video_camera_handle, env_ptr, video_cam_pos, video_cam_rot)
        video_cam_depth_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, video_camera_handle, gymapi.IMAGE_DEPTH)
    
        video_torch_cam_depth_tensor = gymtorch.wrap_tensor(video_cam_depth_tensor)
    

        video_cam_rgb_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, video_camera_handle, gymapi.IMAGE_COLOR)
        video_torch_cam_rgb_tensor = gymtorch.wrap_tensor(video_cam_rgb_tensor)
    
        vidoe_cam_seg_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, video_camera_handle, gymapi.IMAGE_SEGMENTATION)
        video_torch_cam_seg_tensor = gymtorch.wrap_tensor(vidoe_cam_seg_tensor)
    
        video_cam_vinv = torch.inverse((torch.tensor(task.gym.get_camera_view_matrix(task.sim, env_ptr, video_camera_handle)))).to(task.device)
        video_cam_proj = torch.tensor(task.gym.get_camera_proj_matrix(task.sim, env_ptr, video_camera_handle), device=task.device)

        per_env_camera_depth_tensor_list = [video_torch_cam_depth_tensor]
        per_env_camera_rgb_tensor_list =   [video_torch_cam_rgb_tensor]
        per_env_camera_seg_tensor_list =   [video_torch_cam_seg_tensor]

        per_env_camera_view_matrix_inv_list = [video_cam_vinv]
        per_env_camera_proj_matrix_list = [video_cam_proj]

        task.video_cameras.append([video_camera_handle])
            
        task.video_camera_depth_tensor_list.append(per_env_camera_depth_tensor_list)
        task.video_camera_rgb_tensor_list.append(per_env_camera_rgb_tensor_list)
        task.video_camera_seg_tensor_list.append(per_env_camera_seg_tensor_list)
        
        task.video_camera_view_matrix_inv_list.append(per_env_camera_view_matrix_inv_list)
        task.video_camera_proj_matrix_list.append(per_env_camera_proj_matrix_list)

        if task.cfg["use_general_camera"] and env_id == task.num_envs // 2:
            video_save_path = task.video_save_root + "/" + "{}/general.mp4".format(task.cfg["task"]["target_part_category"])
            video_file = os.path.join(video_save_path)
            video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 25,
                                                    (task.image_shape[1], task.image_shape[0]))
            task.video_file_list.append(video_file)
            task.video_writer_list.append(video_writer)
            video_camera_handle = task.gym.create_camera_sensor(env_ptr, task.video_camera_props)
            # set on the front and look towards bottom
            task.gym.set_camera_location(video_camera_handle, env_ptr, general_cam_pos, general_cam_rot)
            video_cam_depth_tensor = task.gym.get_camera_image_gpu_tensor(
                task.sim, env_ptr, video_camera_handle, gymapi.IMAGE_DEPTH)
            
            video_torch_cam_depth_tensor = gymtorch.wrap_tensor(video_cam_depth_tensor)
            video_cam_rgb_tensor = task.gym.get_camera_image_gpu_tensor(
                task.sim, env_ptr, video_camera_handle, gymapi.IMAGE_COLOR)
            video_torch_cam_rgb_tensor = gymtorch.wrap_tensor(video_cam_rgb_tensor)
            vidoe_cam_seg_tensor = task.gym.get_camera_image_gpu_tensor(
                task.sim, env_ptr, video_camera_handle, gymapi.IMAGE_SEGMENTATION)
            video_torch_cam_seg_tensor = gymtorch.wrap_tensor(vidoe_cam_seg_tensor)
            video_cam_vinv = torch.inverse((torch.tensor(task.gym.get_camera_view_matrix(task.sim, env_ptr, video_camera_handle)))).to(task.device)
            video_cam_proj = torch.tensor(task.gym.get_camera_proj_matrix(task.sim, env_ptr, video_camera_handle), device=task.device)
           
            per_env_camera_depth_tensor_list = [video_torch_cam_depth_tensor]
            per_env_camera_rgb_tensor_list =   [video_torch_cam_rgb_tensor]
            per_env_camera_seg_tensor_list =   [video_torch_cam_seg_tensor]
            per_env_camera_view_matrix_inv_list = [video_cam_vinv]
            per_env_camera_proj_matrix_list = [video_cam_proj]
            task.video_cameras.append([video_camera_handle])
                
            task.video_camera_depth_tensor_list.append(per_env_camera_depth_tensor_list)
            task.video_camera_rgb_tensor_list.append(per_env_camera_rgb_tensor_list)
            task.video_camera_seg_tensor_list.append(per_env_camera_seg_tensor_list)
            
            task.video_camera_view_matrix_inv_list.append(per_env_camera_view_matrix_inv_list)
            task.video_camera_proj_matrix_list.append(per_env_camera_proj_matrix_list)

    if task.cfg["env"]["single_camera"]:
        camera_handle = task.gym.create_camera_sensor(env_ptr, task.camera_props)
        # set on the front and look towards bottom
        task.gym.set_camera_location(camera_handle, env_ptr, video_cam_pos, video_cam_rot)
        cam_depth_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
    
        torch_cam_depth_tensor = gymtorch.wrap_tensor(cam_depth_tensor)
    
        cam_rgb_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
        torch_cam_rgb_tensor = gymtorch.wrap_tensor(cam_rgb_tensor)
    
        cam_seg_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
        torch_cam_seg_tensor = gymtorch.wrap_tensor(cam_seg_tensor)
    
        cam_vinv = torch.inverse((torch.tensor(task.gym.get_camera_view_matrix(task.sim, env_ptr, camera_handle)))).to(task.device)
        cam_proj = torch.tensor(task.gym.get_camera_proj_matrix(task.sim, env_ptr, camera_handle), device=task.device)

        per_env_camera_depth_tensor_list = [torch_cam_depth_tensor]
        per_env_camera_rgb_tensor_list =   [torch_cam_rgb_tensor]
        per_env_camera_seg_tensor_list =   [torch_cam_seg_tensor]

        per_env_camera_view_matrix_inv_list = [cam_vinv]
        per_env_camera_proj_matrix_list =     [cam_proj]

        task.cameras.append([camera_handle])
            
        task.camera_depth_tensor_list.append(per_env_camera_depth_tensor_list)
        task.camera_rgb_tensor_list.append(per_env_camera_rgb_tensor_list)
        task.camera_seg_tensor_list.append(per_env_camera_seg_tensor_list)
        
        task.camera_view_matrix_inv_list.append(per_env_camera_view_matrix_inv_list)
        task.camera_proj_matrix_list.append(per_env_camera_proj_matrix_list)

    else:
        camera_handle1 = task.gym.create_camera_sensor(env_ptr, task.camera_props)
        camera_handle2 = task.gym.create_camera_sensor(env_ptr, task.camera_props)
        camera_handle3 = task.gym.create_camera_sensor(env_ptr, task.camera_props)
        # set on the front and look towards bottom
        task.gym.set_camera_location(camera_handle1, env_ptr, cam1_pos, cam1_rot)
        # left 
        task.gym.set_camera_location(camera_handle2, env_ptr, cam2_pos, cam2_rot)
        # right
        task.gym.set_camera_location(camera_handle3, env_ptr, cam3_pos, cam3_rot)
        cam1_depth_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle1, gymapi.IMAGE_DEPTH)
        cam2_depth_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle2, gymapi.IMAGE_DEPTH)
        cam3_depth_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle3, gymapi.IMAGE_DEPTH)
        torch_cam1_depth_tensor = gymtorch.wrap_tensor(cam1_depth_tensor)
        torch_cam2_depth_tensor = gymtorch.wrap_tensor(cam2_depth_tensor)
        torch_cam3_depth_tensor = gymtorch.wrap_tensor(cam3_depth_tensor)

        cam1_rgb_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle1, gymapi.IMAGE_COLOR)
        cam2_rgb_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle2, gymapi.IMAGE_COLOR)
        cam3_rgb_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle3, gymapi.IMAGE_COLOR)
        torch_cam1_rgb_tensor = gymtorch.wrap_tensor(cam1_rgb_tensor)
        torch_cam2_rgb_tensor = gymtorch.wrap_tensor(cam2_rgb_tensor)
        torch_cam3_rgb_tensor = gymtorch.wrap_tensor(cam3_rgb_tensor)

        cam1_seg_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle1, gymapi.IMAGE_SEGMENTATION)
        cam2_seg_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle2, gymapi.IMAGE_SEGMENTATION)
        cam3_seg_tensor = task.gym.get_camera_image_gpu_tensor(
            task.sim, env_ptr, camera_handle3, gymapi.IMAGE_SEGMENTATION)
        torch_cam1_seg_tensor = gymtorch.wrap_tensor(cam1_seg_tensor)
        torch_cam2_seg_tensor = gymtorch.wrap_tensor(cam2_seg_tensor)
        torch_cam3_seg_tensor = gymtorch.wrap_tensor(cam3_seg_tensor)

        cam1_vinv = torch.inverse((torch.tensor(task.gym.get_camera_view_matrix(task.sim, env_ptr, camera_handle1)))).to(task.device)
        cam2_vinv = torch.inverse((torch.tensor(task.gym.get_camera_view_matrix(task.sim, env_ptr, camera_handle2)))).to(task.device)
        cam3_vinv = torch.inverse((torch.tensor(task.gym.get_camera_view_matrix(task.sim, env_ptr, camera_handle3)))).to(task.device)
        cam1_proj = torch.tensor(task.gym.get_camera_proj_matrix(task.sim, env_ptr, camera_handle1), device=task.device)
        cam2_proj = torch.tensor(task.gym.get_camera_proj_matrix(task.sim, env_ptr, camera_handle2), device=task.device)
        cam3_proj = torch.tensor(task.gym.get_camera_proj_matrix(task.sim, env_ptr, camera_handle3), device=task.device)

        per_env_camera_depth_tensor_list = [torch_cam1_depth_tensor, torch_cam2_depth_tensor, torch_cam3_depth_tensor]
        per_env_camera_rgb_tensor_list = [torch_cam1_rgb_tensor, torch_cam2_rgb_tensor, torch_cam3_rgb_tensor]
        per_env_camera_seg_tensor_list = [torch_cam1_seg_tensor, torch_cam2_seg_tensor, torch_cam3_seg_tensor]

        per_env_camera_view_matrix_inv_list = [cam1_vinv, cam2_vinv, cam3_vinv]
        per_env_camera_proj_matrix_list = [cam1_proj, cam2_proj, cam3_proj]

        task.cameras.append([camera_handle1, camera_handle2, camera_handle3])
            
        task.camera_depth_tensor_list.append(per_env_camera_depth_tensor_list)
        task.camera_rgb_tensor_list.append(per_env_camera_rgb_tensor_list)
        task.camera_seg_tensor_list.append(per_env_camera_seg_tensor_list)
        
        task.camera_view_matrix_inv_list.append(per_env_camera_view_matrix_inv_list)
        task.camera_proj_matrix_list.append(per_env_camera_proj_matrix_list)
    # import pdb
    # pdb.set_trace()

    origin = task.gym.get_env_origin(env_ptr)
    task.env_origin[env_id][0] = origin.x
    task.env_origin[env_id][1] = origin.y
    task.env_origin[env_id][2] = origin.z 
