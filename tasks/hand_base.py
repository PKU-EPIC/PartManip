from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from utils import TSDFVolume, gen_camera_pose, TSDFfromMesh
from utils.torch_jit_utils import *
from tasks.load_robot import *

import numpy as np
import torch
import os 
import math 
import sys
from copy import deepcopy

class BaseTask():
    def __init__(self, cfg, sim_params=None):
        self.cfg = cfg
        self.gym = gymapi.acquire_gym()
        self.up_axis = gymapi.UP_AXIS_Z
        self.dt = sim_params.dt
        self.physics_engine = cfg['physics_engine']
        self.asset_root = cfg["asset"]["assetRoot"]
        self.device = cfg['device']
        self.device_id = cfg['device_id']
        self.save_video = cfg['save_video']
        self.add_mask = cfg['add_mask']
        self.learn_input_mode = cfg['learn_input_mode']   # compute in utils/config.py
        self.add_proprio_obs = cfg['add_proprio_obs']
        print('Learning mode: ', self.learn_input_mode)
        print('Add proprio obs: ', self.add_proprio_obs)
        self.headless = cfg['headless']
        self.graphics_device_id = cfg['graphics_device_id']
        if not self.save_video and self.headless and 'depth' not in self.learn_input_mode:
            self.graphics_device_id = -1
        print('graphics card:', self.graphics_device_id)
        self.num_envs = cfg["num_envs"]
        self.max_episode_length = cfg["maxEpisodeLength"]
        self.control_freq_inv = cfg["controlFrequencyInv"]
        self.clip_actions = cfg['clipActions']
        self.clip_obs = cfg['clipObservations']
        # self.reset_time = cfg['reset_time']
        self.robot = eval(cfg['robot']['name'])(self.gym, cfg['robot'], self.dt, self.num_envs, self.device)
        self.num_actions = self.robot.num_actions

        self.num_obs = {}
        for obs_mode in cfg['obs_mode'].keys():
            self.num_obs[obs_mode] = cfg['obs_mode'][obs_mode]
        if 'tsdf' in self.learn_input_mode:
            self.num_obs[self.learn_input_mode] = cfg['obs_mode']['tsdf']['resolution'] ** 3
        self.tsdf_size = cfg['obs_mode']['tsdf']['size']
        self.tsdf_resolution = cfg['obs_mode']['tsdf']['resolution']
        self.tsdf_origin = cfg['obs_mode']['tsdf']['origin']
        if self.add_proprio_obs:
            self.num_obs[self.learn_input_mode] += self.num_obs['proprio_state']
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = {}
        self.extras = {}
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.train_test_flag = 'train'
        self.explore_step = cfg['explore_step']
        self.epis_max_rew = -100 * torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.epis_max_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # create envs, sim
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
        self.load_scene()
        self.gym.prepare_sim(self.sim)

        self.robot.init_jacobian()
        # create viewer
        self.enable_viewer_sync = True
        self.viewer = None
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

        # init camera
        # camera settings
        if self.graphics_device_id != -1:
            self.cam_look_at = cfg['cam']['look_at']
            self.cam_radius = cfg['cam']['radius']
            self.load_camera()

        if self.learn_input_mode == 'mesh_tsdf':
            self.mesh2TSDF = TSDFfromMesh(self.num_envs, self.tsdf_size, self.tsdf_resolution, self.device, parallel=True, vox_origin=self.tsdf_origin)
        
        return 

    ###########################################################
    ######       create scenes
    ###########################################################

    def load_scene(self):
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
        self.preload_all_obj()
        self.robot.preload(self.sim, self.asset_root)
        self.env_ptr_list = []
        for env_id in range(self.num_envs):    
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.env_ptr_list.append(env_ptr)
            self.robot.load_to_env(env_ptr, env_id)
            self.load_obj(env_ptr, env_id)
        self.obj_actor = 1
        return 
        
    def preload_all_obj(self,):
        pass 

    def load_obj(self, env_ptr, env_id):
        raise NotImplementedError
    
    def load_camera(self):
        hemisphere_num_point_ver_list = [3]                        
        hemisphere_num_point_hor = 1                                
        hemisphere_alpha_range_list = [(-45*math.pi/180, 225*math.pi/180)]
        hemisphere_beta_range = (40*math.pi/180, 70*math.pi/180) 
        horizontal_fov = 69.75
        
        if 'img' in self.learn_input_mode:
            self.num_view = 1
            view_list = [0]
            self.image_width = 512 // 4
            self.image_height = 288 // 4
        else:
            self.num_view = sum(hemisphere_num_point_ver_list) * hemisphere_num_point_hor     
            view_list = range(self.num_view)
            self.image_width = 512 
            self.image_height = 288

        fx = self.image_width / 2.0 / math.tan((horizontal_fov / 180.0 * math.pi) / 2.0)
        intrinsic = np.array([[fx, 0, self.image_width//2],
                            [0, fx, self.image_height//2],
                            [0, 0, 1]])
        
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.image_width
        camera_props.height = self.image_height
        camera_props.horizontal_fov = horizontal_fov
        camera_props.enable_tensors = True

        # generate camera poses
        _, _, _, cam_quat_list, _, cam_trans_list, cam_posemat_list = gen_camera_pose(self.cam_look_at, hemisphere_alpha_range_list, hemisphere_num_point_ver_list, hemisphere_num_point_hor, hemisphere_beta_range, self.cam_radius)

        if 'depth' in self.learn_input_mode or 'rgb' in self.learn_input_mode:
            self.camera_handle_list = {}
            self.camera_tensor_list = []
            self.seg_tensor_list = []

            # register camera per-view
            for env_id in range(len(self.env_ptr_list)):
                env_ptr = self.env_ptr_list[env_id]
                pre_env_camera_tensor_list = []
                pre_env_seg_tensor_list = []
                for view_id in view_list:
                    quat_new = cam_quat_list[view_id]
                    trans = cam_trans_list[view_id]  
                    
                    transform = gymapi.Transform()
                    transform.p = gymapi.Vec3(trans[0][0], trans[1][0], trans[2][0])
                    transform.r = gymapi.Quat(quat_new[0], quat_new[1], quat_new[2], quat_new[3])
                    
                    camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
                    self.gym.set_camera_transform(camera_handle, env_ptr, transform)       
                    self.camera_handle_list[f"{env_id}_{view_id}"] = camera_handle
                    if 'depth' in self.learn_input_mode:
                        one_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
                    else:
                        one_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)

                    torch_camera_tensor = gymtorch.wrap_tensor(one_camera_tensor)
                    pre_env_camera_tensor_list.append(torch_camera_tensor)

                    if self.add_mask:
                        one_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
                        torch_seg_tensor = gymtorch.wrap_tensor(one_seg_tensor)
                        pre_env_seg_tensor_list.append(torch_seg_tensor)
                self.camera_tensor_list.append(pre_env_camera_tensor_list)
                self.seg_tensor_list.append(pre_env_seg_tensor_list)
            
            self.TSDF = TSDFVolume(device=self.device, size=self.tsdf_size, resolution=self.tsdf_resolution, _vol_origin=self.tsdf_origin)
            self.TSDF.register_camera(cam_posemat_list, intrinsic, self.image_height, self.image_width, self.num_envs)

        if self.save_video:
            self.env_for_video = self.env_ptr_list[3]
            view_id = 0
            quat_new = cam_quat_list[view_id]
            trans = cam_trans_list[view_id]  
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(trans[0][0], trans[1][0], trans[2][0])
            transform.r = gymapi.Quat(quat_new[0], quat_new[1], quat_new[2], quat_new[3])
            # camera_props.use_collision_geometry = True
            camera_props.width = 1024
            camera_props.height = 512
            self.save_cam_handle = self.gym.create_camera_sensor(self.env_for_video, camera_props)
            self.gym.set_camera_transform(self.save_cam_handle, self.env_for_video, transform) 
           
        return

    ###########################################################
    ######      take one step
    ###########################################################

    def step(self, actions, save_image_path=None):
        '''
        To add a new task, overwrite 'self.reset_idx()', 'self.compute_observations()' and 'self.compute_reward()'

        Pipeline in time T:
            -- pre_physics_step()
                1. compute actions (using self.robot.control())
                2. compute 'self.reset_buf' using 'self.rew_buf' and 'self.progress_buf' in time T-1
                3. reset some envs using 'self.reset_idx()',
                    or directly using gym.set_dof_position_target_tensor to take actions.
                    NOTE: In 'self.reset_idx()', those envs who don't need to reset 
                    should take actions using gym.set_dof_position_target_tensor.

            -- post_physics_step()
                update 'self.progress_buf'.
                -- refresh_gym_tensor(): Must refresh gym tensors before compute observation.
                -- compute_observations(): compute 'self.obs_buf'
                -- compute_reward(): compute 'self.rew_buf' using updated 'self.obs_buf'

        return:
            self.obs_buf (If self.reset_buf False, obs_{T-1} + action_T -> obs_T)
            self.rew_buf 
            self.reset_buf (If True, then this transition shouldn't be used for training!)
            self.extra (Debug)
        
        '''
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
        # save_image_path = f'./debug.png'
        if (self.save_video and save_image_path is not None) or 'depth' in self.learn_input_mode or 'rgb' in self.learn_input_mode:
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
                elif self.learn_input_mode == 'depth_sparse':
                    self.obs_buf['depth_sparse'] = self.TSDF.sparse_voxel(all_camera_tensor).reshape(self.num_envs,-1)
                elif self.learn_input_mode == 'depth_img':
                    # import cv2
                    # for i in range(self.num_envs):
                    #     cv2.imwrite(f'debug{i}.jpg', -self.camera_tensor_list[i][0].cpu().numpy()*255)
                    self.obs_buf['depth_img'] = all_camera_tensor.squeeze().reshape(self.num_envs, -1)
                else:
                    raise NotImplementedError
            elif 'rgb' in self.learn_input_mode:
                all_camera_tensor = []
                for env_id in range(self.num_envs):
                    all_camera_tensor.append(self.camera_tensor_list[env_id][0][...,:3].permute(2,0,1))
                all_camera_tensor = torch.stack(all_camera_tensor, dim=0)       # [num_envs, 3, 288, 512]
                # import cv2
                # cv2.imwrite('debug.jpg', self.camera_tensor_list[0][0][...,:3].cpu().numpy())
                # cv2.imwrite('debug2.jpg', self.camera_tensor_list[1][0][...,:3].cpu().numpy())
                # exit(1)
                self.obs_buf['rgb_img'] = all_camera_tensor.reshape(self.num_envs,-1)

            if self.save_video and save_image_path is not None:
                os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
                self.gym.write_camera_image_to_file(self.sim, self.env_for_video, self.save_cam_handle, gymapi.IMAGE_COLOR, save_image_path)
            
            self.gym.end_access_image_tensors(self.sim)

        return 

    def pre_physics_step(self, actions):
        # print(actions[5,:3])
        self.pos_act = self.robot.control(actions)
        
        if self.train_test_flag == 'train':
            # self.reset_buf = (self.progress_buf >= self.max_episode_length) | self.success
            self.epis_max_step = torch.where(self.rew_buf < self.epis_max_rew, self.epis_max_step, self.progress_buf)
            self.epis_max_rew = torch.maximum(self.rew_buf, self.epis_max_rew)
            self.reset_buf = ((self.progress_buf >= self.epis_max_step + self.explore_step)) | self.success
            self.reset_succ = deepcopy(self.success)
            self.extras['succ_rate'] = self.success.int().sum(dim=-1, keepdim=True) / torch.clamp(self.reset_buf.int().sum(), min=1)
        elif self.train_test_flag == 'test':
            self.reset_buf = self.progress_buf >= self.max_episode_length
        else:
            raise NotImplementedError

        if self.reset_buf.sum() > 0:
            self.reset_idx(self.reset_buf)
        else:
            self.pos_act_all[self.dof_state_mask[:, :self.robot.num_dofs]] = self.pos_act
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_act_all))

        return

    def post_physics_step(self, actions):
        self.progress_buf += 1
        self.refresh_gym_tensor()
        self.compute_observations()
        self.compute_reward(actions)
        return 

    def reset(self,type="reset"):
        to_reset = torch.ones(self.num_envs)
        self.reset_idx(to_reset)
        self.gym.simulate(self.sim)     # Necessary for updating rigid_body_state!!
        self.gym.fetch_results(self.sim, True)
        self.render()
        self.refresh_gym_tensor()
        self.compute_observations(type=type)
        return self.obs_buf

    def refresh_gym_tensor(self) :
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return 
    
    def reset_idx(self, to_reset):
        """
        Reset envs where to_reset[env_ids] = 1.
        For to_reset[env_ids] = 1, set dof + action + root.
        For to_reset[env_ids] != 1, set action.
        """
        raise NotImplementedError

    def compute_reward(self, action):
        raise NotImplementedError
    
    def compute_observations(self):
        raise NotImplementedError

    ###########################################################
    ######       useful functions
    ###########################################################

    def compute_scene_pose(self):
        exit(1) # TODO: check this function
        remove_tip_rb_tensor = deepcopy(self.rigid_body_tensor[:, :12, :7])
        remove_tip_rb_tensor[:,-2] = self.rigid_body_tensor[:, -3, :7]
        remove_tip_rb_tensor[:,-1] = self.rigid_body_tensor[:, -1, :7]
        pos = remove_tip_rb_tensor[:, :, :3]
        q = remove_tip_rb_tensor[:, :, 3:]

        rot = quat_to_mat(q)
        rot[:,:-1,:] = torch.matmul(rot[:,:-1,:], self.coordinate_transform_matrix.unsqueeze(0))
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

 