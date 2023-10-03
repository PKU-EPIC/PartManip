from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from utils import TSDFVolume, gen_camera_pose #TSDFfromMesh
from utils.torch_jit_utils import *

import numpy as np
import torch
import os 
import math 
import sys
from copy import deepcopy
import random

# Y Environments' Base class for RL task
class BaseTask():
    def __init__(self, cfg, base_cfg, sim_params=None):
        ### self
        self.cfg = cfg
        self.base_cfg = base_cfg
        ### gym
        self.gym = gymapi.acquire_gym()
        self.up_axis = gymapi.UP_AXIS_Z
        self.physics_engine = cfg['physics_engine']
        self.device = cfg['device']
        self.device_id = cfg['device_id']
        self.save_video = cfg['save_video']
        self.learn_input_mode = cfg['learn_input_mode']   # compute in utils/config.py
        self.add_proprio_obs = cfg['add_proprio_obs']
        print('Learning mode: ', self.learn_input_mode)
        print('Add proprio obs: ', self.add_proprio_obs)
        self.headless = cfg['headless']
        self.graphics_device_id = cfg['graphics_device_id']
        if not self.save_video and self.headless and 'pc' not in self.learn_input_mode:
            self.graphics_device_id = -1
        self.num_envs = cfg["num_envs"] # need modify
        self.max_episode_length = cfg["maxEpisodeLength"]
        self.control_freq_inv = cfg["controlFrequencyInv"]
        self.clip_actions = cfg['clipActions']
        self.clip_obs = cfg['clipObservations']

        if cfg['driveMode'] == 'ik' or cfg['driveMode'] == 'heuristic':
            self.num_actions = 7
        elif cfg['driveMode'] == 'pos':
            self.num_actions = 8
        else:
            raise NotImplementedError
        
        self.num_obs = {}
        for obs_mode in cfg['obs_mode'].keys():
            self.num_obs[obs_mode] = cfg['obs_mode'][obs_mode]
        if self.add_proprio_obs:
            self.num_obs[self.learn_input_mode] += self.num_obs['proprio_state']
            
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = {}
        self.extras = {}
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # create envs, sim
        self.create_sim(sim_params)
        self.gym.prepare_sim(self.sim)

        # create viewer and load camera
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            
            # set the camera position based on up axis
            if self.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(1.4, -1.3, 1.7)
                cam_target = gymapi.Vec3(0, 0, 0.7)
            else:
                cam_pos = gymapi.Vec3(10, 10, 3.0)
                cam_target = gymapi.Vec3(0, 0, 0.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        self.load_camera()

        # from simulator acquire tensors
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.rigid_body_tensor = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "franka"))
        # NOTE use net_contact_foce instead of force_sensor! force_sensor is unstable and can >0 even without touching object
        self.contact_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        
        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)         # (env, 2, 13) 0:hand, 1:object, 
        self.dof_state_tensor = self.dof_state_tensor.view(self.num_envs, -1, 2)    # (env, 9, 2)   hand
        self.rigid_body_tensor = self.rigid_body_tensor.view(self.num_envs, -1, 13) # (env, 12, 13) 0-11:hand, 12: obj
        self.contact_force_tensor = self.contact_force_tensor.view(self.num_envs, -1, 3) # (env, 12, 3) 0-11:hand, 12: obj
        
        # precise slices of tensors
        env_ptr = self.env_ptr_list[0]
        franka1_actor = 0
        self.lfinger_rb_index = self.gym.find_actor_rigid_body_index(env_ptr, franka1_actor, "panda_leftfinger", gymapi.DOMAIN_ENV)
        self.rfinger_rb_index = self.gym.find_actor_rigid_body_index(env_ptr,franka1_actor, "panda_rightfinger", gymapi.DOMAIN_ENV)
        self.ltip_rb_index = self.gym.find_actor_rigid_body_index(env_ptr,franka1_actor, "panda_lefttip", gymapi.DOMAIN_ENV)
        self.rtip_rb_index = self.gym.find_actor_rigid_body_index(env_ptr,franka1_actor, "panda_righttip", gymapi.DOMAIN_ENV)
        
        self.ltip_rb_tensor = self.rigid_body_tensor[:, self.ltip_rb_index, :]
        self.rtip_rb_tensor = self.rigid_body_tensor[:, self.rtip_rb_index, :]
        self.franka_dof_tensor = self.dof_state_tensor[:, :self.num_franka_dofs, :]
        self.obj_root_tensor = self.root_tensor[:, self.obj_actor, :]       # (num_env, 13)

        return 

    ###########################################################
    ######       create scenes
    ###########################################################

    def create_sim(self, sim_params):
        self.dt = sim_params.dt

        if self.up_axis == gymapi.UP_AXIS_Z:
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
        else:
            raise NotImplementedError

        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
        
        self._load_scene()

        return
        
    def _load_franka(self, env_ptr, env_id):
        if self.franka_loaded == False:
            asset_root = self.asset_root
            asset_file = "franka_description/robots/franka_panda.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
            asset_options.flip_visual_attachments = True
            asset_options.collapse_fixed_joints = False 
            asset_options.armature = 0.01       #TODO: armature??
            self.franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            self.franka_loaded = True

            # set initial dof
            self.num_franka_dofs = self.gym.get_asset_dof_count(self.franka_asset)  # 9
            self.franka_default_dof_pos = torch.tensor([0.6806, -0.2319, -0.6754, -2.4905, -0.1961,  2.2925, -0.6390, 
                                    0.0400, 0.0400], device=self.device, dtype=torch.float)
            self.franka_default_root = torch.tensor([0.0, -0.45, 0.0, 0, 0, np.sin(np.pi/4), np.cos(np.pi/4)],
                                 device=self.device, dtype=torch.float)

            # for saving 6dof pose of franka joints
            self.coordinate_transform_matrix = torch.zeros((self.num_envs,11,3,3), device=self.device)
            self.coordinate_transform_matrix[:,:,0,0] = 1
            self.coordinate_transform_matrix[:,:,1,2] = -1
            self.coordinate_transform_matrix[:,:,2,1] = 1
            self.coordinate_transform_matrix[:,-1,1,2] = 1   # I don't know why but it seems necessary

            franka_dof_max_torque, franka_dof_lower_limits, franka_dof_upper_limits = self._get_dof_property(self.franka_asset) # (9,) (9,) (9,)
            self.franka_dof_max_torque_tensor = torch.tensor(franka_dof_max_torque, device=self.device) #(9,)
            self.franka_dof_mean_limits_tensor = torch.tensor((franka_dof_lower_limits + franka_dof_upper_limits)/2, device=self.device)#(9,)
            self.franka_dof_limits_range_tensor = torch.tensor((franka_dof_upper_limits - franka_dof_lower_limits)/2, device=self.device)#(9,)
            self.franka_dof_lower_limits_tensor = torch.tensor(franka_dof_lower_limits, device=self.device)
            self.franka_dof_upper_limits_tensor = torch.tensor(franka_dof_upper_limits, device=self.device)

        # set initial props
        franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        arm_stf = 400
        arm_dmp = 40
        gripper_stf = 800
        gripper_dmp = 40
        franka_dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
        franka_dof_props['stiffness'][:-2] = arm_stf
        franka_dof_props['damping'][:-2] = arm_dmp
        franka_dof_props['stiffness'][-2:] = gripper_stf
        franka_dof_props['damping'][-2:] = gripper_dmp

        franka_actor = self.gym.create_actor(env_ptr, self.franka_asset, gymapi.Transform(), "franka", env_id, 2, 0)
        self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
        
        return 

    def _get_dof_property(self, asset):
        dof_props = self.gym.get_asset_dof_properties(asset)
        dof_num = self.gym.get_asset_dof_count(asset)
        dof_lower_limits = []
        dof_upper_limits = []
        dof_max_torque = []
        for i in range(dof_num):
            dof_max_torque.append(dof_props["effort"][i])
            dof_lower_limits.append(dof_props["lower"][i])
            dof_upper_limits.append(dof_props["upper"][i])
        dof_max_torque = np.array(dof_max_torque)
        dof_lower_limits = np.array(dof_lower_limits)
        dof_upper_limits = np.array(dof_upper_limits)
        return dof_max_torque, dof_lower_limits, dof_upper_limits

    def _load_scene(self):
        # create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.,0.,1.)
        plane_params.static_friction = 0.1
        plane_params.dynamic_friction = 0.1
        self.gym.add_ground(self.sim, plane_params)

        # arrange envs
        spacing = self.cfg['envSpacing']
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.space_middle = torch.zeros((self.num_envs, 3), device=self.device)
        self.space_range = torch.zeros((self.num_envs, 3), device=self.device)
        self.space_middle[:, 0] = self.space_middle[:, 1] = 0
        self.space_middle[:, 2] = spacing/2
        self.space_range[:, 0] = self.space_range[:, 1] = spacing
        self.space_middle[:, 2] = spacing/2
        num_per_row = int(np.sqrt(self.num_envs))

        # create envs
        self.env_ptr_list = []
        for env_id in range(self.num_envs):    
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.env_ptr_list.append(env_ptr)
            self._load_franka(env_ptr, env_id)
            self._load_obj(env_ptr, env_id)
        return 

    def _load_obj(self, env_id):
        raise NotImplementedError

    def load_camera(self):
        ### q
        if self.base_cfg['pipeline_mode'] in ["fixed_nerf-rl", "nerf-rl"]:
            cam_RT_list = []
            for env_id in range(len(self.env_ptr_list)):
                look_at = np.array(self.shm_cfg['look_at'])
                hemisphere_num_point_ver_list = self.shm_cfg['hemisphere_num_point_ver_list']  # 水平方向整圆的点数
                hemisphere_num_point_hor = self.shm_cfg['hemisphere_num_point_hor']  # 竖直方向一条圆弧（1/4圆以内）的点数
                self.view_num = sum(hemisphere_num_point_ver_list) * hemisphere_num_point_hor
                if self.shm_cfg['is_camera_rand']:
                    hemisphere_alpha_range_list = [(random.uniform(-55, -30) * math.pi / 180, (
                                random.uniform(210, 235) * math.pi / 180))]  # [(-55*math.pi/180, 235*math.pi/180)]
                    hemisphere_beta_range = (random.uniform(30, 50) * math.pi / 180,
                                             random.uniform(60, 70) * math.pi / 180)  # (50*math.pi/180, 70*math.pi/180)
                    hemisphere_radius = random.uniform(0.75, 0.85)  # 0.85
                else:
                    hemisphere_alpha_range_list = [(-45 * math.pi / 180, 225 * math.pi / 180)]
                    hemisphere_beta_range = (40 * math.pi / 180, 70 * math.pi / 180)  # 竖直方向一条圆弧的角度范围
                    hemisphere_radius = self.shm_cfg['hemisphere_radius']  # 球半径（米）

                # generate camera poses
                cam_quat_src_list, cam_rot_src_list, cam_trans_src_list, cam_quat_list, cam_rot_list, cam_trans_list, cam_posemat_list = gen_camera_pose(
                    look_at,
                    hemisphere_alpha_range_list,
                    hemisphere_num_point_ver_list,
                    hemisphere_num_point_hor,
                    hemisphere_beta_range,
                    hemisphere_radius,
                    is_camera_rand=True)
                cam_rot_array = np.array(cam_rot_src_list)  # (num_view, 3, 3)
                cam_trans_array = np.array(cam_trans_src_list)  # (num_view, 3, 1)
                cam_RT = np.concatenate([cam_rot_array, cam_trans_array], 2)  # (num_view, 3, 4)
                cam_RT_list.append(cam_RT)

            cam_RT_array = np.array(cam_RT_list, np.float32).reshape(-1).tobytes()  # (num_envs, num_view, 3, 4)
            #print(np.array(cam_RT_list, np.float32).shape, self.shm_dict["shm_campose_size"])
            self.shm_dict["campose"][0:self.shm_dict["shm_campose_size"]] = cam_RT_array
        ### q

        if 'depth' in self.learn_input_mode:
            # parameters
            look_at = np.array([0,0,0])         
            hemisphere_radius = 0.8   
            hemisphere_num_point_ver_list = [6]                        
            hemisphere_num_point_hor = 1                                
            hemisphere_alpha_range_list = [(-45*math.pi/180, 225*math.pi/180)]
            hemisphere_beta_range = (40*math.pi/180, 70*math.pi/180) 
            if 'tsdf' in self.learn_input_mode:
                self.image_width = 640
                self.image_height = 360
            else:   # to speed up fps
                self.image_width = 320
                self.image_height = 180
            horizontal_fov = 69.75
            fx = self.image_width / 2.0 / math.tan((horizontal_fov / 180.0 * math.pi) / 2.0)
            intrinsic = np.array([[fx, 0, self.image_width//2],
                                [0, fx, self.image_height//2],
                                [0, 0, 1]])
            self.num_view = sum(hemisphere_num_point_ver_list) * hemisphere_num_point_hor     

            camera_props = gymapi.CameraProperties()
            camera_props.width = self.image_width
            camera_props.height = self.image_height
            camera_props.horizontal_fov = horizontal_fov
            camera_props.enable_tensors = True

            # generate camera poses
            ### q
            _, _, _, cam_quat_list, _, cam_trans_list, cam_posemat_list = gen_camera_pose(look_at, hemisphere_alpha_range_list, hemisphere_num_point_ver_list, hemisphere_num_point_hor, hemisphere_beta_range, hemisphere_radius)
            ### q

            self.camera_handle_list = {}
            self.camera_tensor_list = []

            # register camera per-view
            for env_id in range(len(self.env_ptr_list)):
                env_ptr = self.env_ptr_list[env_id]
                pre_env_camera_tensor_list = []
                for view_id in range(self.num_view):
                    quat_new = cam_quat_list[view_id]
                    trans = cam_trans_list[view_id]  
                    
                    transform = gymapi.Transform()
                    transform.p = gymapi.Vec3(trans[0][0], trans[1][0], trans[2][0])
                    transform.r = gymapi.Quat(quat_new[0], quat_new[1], quat_new[2], quat_new[3])
                    
                    camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
                    self.gym.set_camera_transform(camera_handle, env_ptr, transform)       
                    self.camera_handle_list[f"{env_id}_{view_id}"] = camera_handle
                
                    one_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
                    torch_camera_tensor = gymtorch.wrap_tensor(one_camera_tensor)
                    pre_env_camera_tensor_list.append(torch_camera_tensor)
                self.camera_tensor_list.append(pre_env_camera_tensor_list)
            
            self.TSDF = TSDFVolume(device=self.device, size=self.tsdf_size, resolution=self.tsdf_resolution)
            self.TSDF.register_camera(cam_posemat_list, intrinsic, self.image_height, self.image_width, self.num_envs)

        ### q
        if self.learn_input_mode == 'mesh_tsdf':
            self.mesh2TSDF = TSDFfromMesh(self.num_envs, self.tsdf_size, self.tsdf_resolution, self.device, parallel=True)
        ### q

        if self.save_video:
            self.env_for_video = self.env_ptr_list[0]
            x, y, z = 0.8, -0.8, 0.8
            tx, ty, tz = 0, 0, 0
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1024
            camera_props.height = 728
            self.save_cam_handle = self.gym.create_camera_sensor(self.env_for_video, camera_props)
            assert self.save_cam_handle != -1
            self.gym.set_camera_location(self.save_cam_handle, self.env_for_video, gymapi.Vec3(x,y,z), gymapi.Vec3(tx,ty,tz))

        return 

    ###########################################################
    ######      take one step
    ###########################################################

    def step(self, actions, curr_iter=0, step_id=0, save_image_path=None):
        ### q
        self.curr_iter = curr_iter
        self.step_id = step_id
        ### q

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.render(save_image_path=save_image_path)

        # compute observations, rewards, resets, ...
        self.post_physics_step(actions)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def render(self, save_image_path=None):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

        if (self.save_video and save_image_path is not None) or 'depth' in self.learn_input_mode:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            if 'depth' in self.learn_input_mode:
                all_camera_tensor = []
                for env_id in range(self.num_envs):
                    tmp = torch.stack(self.camera_tensor_list[env_id], dim=0)
                    all_camera_tensor.append(tmp)
                all_camera_tensor = torch.stack(all_camera_tensor, dim=0) # [num_envs, num_views, 360, 640]
                all_camera_tensor = -all_camera_tensor
                all_camera_tensor = torch.where(torch.isinf(all_camera_tensor), torch.full_like(all_camera_tensor, 100), all_camera_tensor)
                if self.learn_input_mode == 'depth_tsdf':
                    tsdf_vol = self.TSDF.integrate(all_camera_tensor)
                    self.obs_buf['depth_tsdf'] = tsdf_vol.reshape(self.num_envs, -1)
                    # pc_list = self.TSDF.extract_point_cloud()
                    # for i in range(2):
                    #     np.savetxt(f'debug{i}.txt', pc_list[i])
                    # exit(1)
                elif self.learn_input_mode == 'depth_pc':
                    pc_list = self.TSDF.depth2pc(all_camera_tensor)
                    self.obs_buf['depth_pc'] = pc_list.reshape(self.num_envs,-1)
                else:
                    raise NotImplementedError

            if self.save_video and save_image_path is not None:
                os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
                self.gym.write_camera_image_to_file(self.sim, self.env_for_video, self.save_cam_handle, gymapi.IMAGE_COLOR, save_image_path)
            
            self.gym.end_access_image_tensors(self.sim)
        
        return 

    def post_physics_step(self, actions):
        self.progress_buf += 1
        self._refresh()
        self.compute_observations()
        self.compute_reward(actions)
        return 

    def reset(self, curr_iter=0, step_id=0):
        ### q
        self.curr_iter = curr_iter
        self.step_id = step_id
        ### q
        to_reset = torch.ones(self.num_envs)
        self.reset_idx(to_reset)
        self.gym.simulate(self.sim)     # Necessary!!
        self.gym.fetch_results(self.sim, True)
        self.render()
        self._refresh()  
        self.compute_observations()
        return self.obs_buf

    def _refresh(self) :
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return 
    
    def pre_physics_step(self, actions):
        raise NotImplementedError

    def reset_idx(self, to_reset):
        raise NotImplementedError

    def compute_reward(self, action):
        raise NotImplementedError
    
    def compute_observations(self):
        raise NotImplementedError

    ###########################################################
    ######       useful functions
    ###########################################################

    def compute_force_grasp(self, max_angle=None):
        lforce = self.contact_force_tensor[:, self.lfinger_rb_index]
        rforce = self.contact_force_tensor[:, self.rfinger_rb_index]

        # NOTE don't use angle! this is very unstable and inaccurate!!!
        lflag = (lforce.norm(dim=-1) > 0.0001) 
        rflag = (rforce.norm(dim=-1) > 0.0001) 
        return lflag & rflag, lflag | rflag
    
    def control_ik(self, dpose):
        damping = 0.05
        j_eef = (self.jacobian_tensor[:, self.ltip_rb_index - 1, :, :7] + self.jacobian_tensor[:, self.rtip_rb_index - 1, :, :7]) / 2

        # solve damped least squares
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def compute_scene_pose(self):
        remove_tip_rb_tensor = deepcopy(self.rigid_body_tensor[:, :12, :7])
        remove_tip_rb_tensor[:,-2] = self.rigid_body_tensor[:, -3, :7]
        remove_tip_rb_tensor[:,-1] = self.rigid_body_tensor[:, -1, :7]
        pos = remove_tip_rb_tensor[:, :, :3]
        q = remove_tip_rb_tensor[:, :, 3:]

        rot = quat_to_mat(q)
        rot[:,:-1,:] = torch.matmul(rot[:,:-1,:], self.coordinate_transform_matrix)
        return rot, pos 

    def save_scene_pose(self, save_path=None):
        rot, pos = self.compute_scene_pose()
        save_dict = {'rot': rot.cpu().numpy(), 'pos': pos.cpu().numpy()}
        ### q
        if save_path != None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ### q
        # np.save(save_path, save_dict)
        return save_dict

 