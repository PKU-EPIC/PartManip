from tasks.load_robot import *

from isaacgym import gymapi, gymutil
from os.path import join as pjoin
import os
import numpy as np
import yaml 
from copy import deepcopy

device = 'cuda:4'
num_envs = 10
robot_cfg = {
    'driveMode': 'ik',
    'assetFile': 'franka_panda_sdf_mobile',
    'dof': [0, 0, 0, -0.2724, -0.1511,  0.2898, -2.3792, -2.8973,  2.4690,  2.3973, 0.0400, 0.0400],
    'root': [0., 0., 0., 0., 0., 0., 1.]
}

gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="Asset and Environment Information")

# set physx parameters
with open('cfg/base_cfg.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
sim_params = gymapi.SimParams()
cfg_physx = cfg['sim']['physx']
for k in cfg_physx.keys():
    setattr(sim_params.physx, k, cfg_physx[k])
sim_params.use_gpu_pipeline = True

args.graphics_device_id = int(device.split(':')[-1])
args.compute_device_id = int(device.split(':')[-1])
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

dt = 1 / 60
robot = franka(gym, robot_cfg, dt, num_envs, device)

# create scene
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.,0.,1.)
plane_params.static_friction = 0.1
plane_params.dynamic_friction = 0.1
gym.add_ground(sim, plane_params)

spacing = 2
lower = gymapi.Vec3(-spacing, -spacing, 0.0)
upper = gymapi.Vec3(spacing, spacing, spacing)
num_per_row = int(np.sqrt(num_envs))
asset_root = 'assets'
robot.preload(sim, asset_root)
env_ptr_list = []
for env_id in range(num_envs):    
    env_ptr = gym.create_env(sim, lower, upper, num_per_row)
    env_ptr_list.append(env_ptr)
    robot.load_to_env(env_ptr, env_id)

gym.prepare_sim(sim)
robot.init_jacobian()

# create camera
camera_props = gymapi.CameraProperties()
camera_props.horizontal_fov = 75.0
camera_props.width = 1920
camera_props.height = 1080
env_camera = env_ptr_list[0]
x,y,z = 0,-2,2
tx,ty,tz = 0,0,0
save_cam_handle = gym.create_camera_sensor(env_camera, camera_props)
gym.set_camera_location(save_cam_handle, env_camera, gymapi.Vec3(x,y,z), gymapi.Vec3(tx,ty,tz))
save_image_path = './debug.png'
           
# initialize
rigid_body_tensor = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
dof_state_tensor = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
dof_state_tensor = dof_state_tensor.view(num_envs, -1, 2)    # (env, 9, 2)   hand
rigid_body_tensor = rigid_body_tensor.view(num_envs, -1, 13) # (env, 13, 13) 0-11:hand, 12: obj
reset_dof_states = torch.zeros((num_envs * robot.num_dofs, 2), device=device)
reset_dof_states[:, 0] = torch.tensor(robot_cfg['dof'], device=device).unsqueeze(0).repeat(num_envs, 1).reshape(-1)
reset_root_states =torch.zeros((num_envs, 13), device=device)
reset_root_states[:, :7] = torch.tensor(robot_cfg['root'], device=device).unsqueeze(0).repeat(num_envs, 1)
gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(reset_dof_states))
gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(reset_root_states))     
gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(reset_dof_states[...,0].contiguous()))
gym.simulate(sim)
gym.fetch_results(sim, True)

def refresh_gym_tensor(gym, sim):
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    return 

refresh_gym_tensor(gym, sim)   
robot.update_state(rigid_body_tensor, dof_state_tensor)

for i in range(20):
    action = torch.rand((num_envs, 10), device=device)
    # action[:, :3] = -0.99
    # action[:, 3:] = 0.99

    curr_tip_pos = deepcopy(robot.tip_pos)
    curr_base_pos = deepcopy(robot.dof_qpos_raw[:, :3])
    curr_tip_length = deepcopy(robot.gripper_length)

    target_tip_pos = curr_tip_pos + action[:, 3:6] * 0.005
    target_base_pos = curr_base_pos + action[:, :3] * 0.005
    target_tip_length = curr_tip_length + action[:, -1] * dt * 2 / 5
    
    pos_act = robot.control(action)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_act))

    for j in range(2):
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

        refresh_gym_tensor(gym, sim)   

        gym.write_camera_image_to_file(sim, env_camera, save_cam_handle, gymapi.IMAGE_COLOR, save_image_path)
        robot.update_state(rigid_body_tensor, dof_state_tensor)

    real_tip_pos = robot.tip_pos
    real_base_pos = robot.dof_qpos_raw[:, :3]
    real_tip_length = robot.gripper_length

    print(real_tip_pos[0], target_tip_pos[0], curr_tip_pos[0])
    # print(real_tip_length[0], target_tip_length[0], curr_tip_length[0])
    # print(real_base_pos[0], target_base_pos[0], curr_base_pos[0])
    err_tip_pos = ((real_tip_pos - target_tip_pos).norm(dim=-1) / (target_tip_pos-curr_tip_pos).norm(dim=-1)).mean()
    err_tip_length = ((real_tip_length-target_tip_length) / (target_tip_length-curr_tip_length)).abs().mean()
    err_base_pos = ((real_base_pos - target_base_pos).norm(dim=-1) / (target_base_pos-curr_base_pos).norm(dim=-1)).mean()

    print('Error', err_tip_pos, err_tip_length, err_base_pos)