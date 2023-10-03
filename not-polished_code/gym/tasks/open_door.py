
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from tasks.hand_base import BaseTask
from utils.torch_jit_utils import *
# from utils import TSDFfromMesh

import numpy as np
import torch
import trimesh 
from copy import deepcopy
import time
import os

class open_door(BaseTask):
    def __init__(self, cfg, base_cfg, shm_cfg=None, sim_params=None, shm_dict=None):
        self.cfg = cfg
        ### q
        self.base_cfg = base_cfg
        self.shm_cfg = shm_cfg
        self.shm_dict = shm_dict
        ### q
        self.asset_root = cfg["asset"]["assetRoot"]

        self.obj_loaded = False
        self.franka_loaded = False

        super().__init__(cfg, base_cfg, shm_cfg, sim_params, shm_dict)

        # params of randomization
        self.random_reset = cfg["random_reset"]

        # create useful variables
        self.goal_thresh=0.025
        self.success_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.pos_act = torch.zeros((self.num_envs, self.num_franka_dofs), device=self.device)
        self.eff_act = torch.zeros((self.num_envs, self.num_franka_dofs), device=self.device)
        self.global_indices = torch.arange(self.num_envs*2, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        # initialize
        self.reset()

        return 

    def _load_obj(self, env_ptr, env_id):
        if self.obj_loaded == False :
            self.obj_scale = 0.05
            cubeA_opts = gymapi.AssetOptions()
            cubeA_opts.disable_gravity = False 
            cubeA_opts.density = 1000
            self.obj_asset = self.gym.create_box(self.sim, *([self.obj_scale] * 3), cubeA_opts)
            self.obj_pc = trimesh.sample.sample_surface_even(trimesh.creation.box(), 512)[0] * self.obj_scale
            self.obj_pc = torch.FloatTensor(self.obj_pc).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
            self.obj_loaded = True

            # set initial obj pose
            self.obj_default_root = torch.tensor([0, 0, 0, 0, 0, 0, 1], device=self.device, dtype=torch.float)

        self.obj_actor = self.gym.create_actor(env_ptr, self.obj_asset, gymapi.Transform(), "obj", env_id, 0, 0)

    def compute_obj_goal_pos(self, init_pos, num_envs):
        goal_pos = init_pos + torch.tensor([0,0,0.2], device=self.device)[None,:]
        # REGION = torch.tensor([0.15, 0.15, 0.1], device=self.device).unsqueeze(0)
        # goal_pos = torch.rand((num_envs, 3),device=self.device) * REGION * 2 - REGION + torch.tensor([0,0,0.2], device=self.device)[None,:] 
        # MIN_DIST = self.goal_thresh * 2
        # init_goal_dist = (goal_pos - init_pos).norm(dim=-1)
        # tooclose = torch.where(init_goal_dist < MIN_DIST)[0]
        # if len(tooclose) > 0:
        #     init_goal_dist = init_goal_dist.unsqueeze(-1)
        #     goal_pos = goal_pos + (init_goal_dist < MIN_DIST) * (goal_pos-init_pos) / torch.clamp(init_goal_dist, min=1e-5) * (MIN_DIST-init_goal_dist)
        return goal_pos

    def compute_reward(self, action):
        '''
        update self.rew_buf, self.reset_buf
        NOTE: 
        1. reward design should be smooth even in some good situation.
        2. 
        '''
        # reaching
        obj_pos = self.obj_root_tensor[:,:3] 
        tcp_to_obj_dist = (self.tip_rb_tensor[:,:3] - obj_pos).norm(dim=-1)
        is_reached = (tcp_to_obj_dist < 0.05)
        reaching_reward = - tcp_to_obj_dist     # better than tanh reward (not sure, seems to be related to other variables) 

        visual_grasp_reward, visual_is_grasp = self.compute_obj_in_gripper(self.obj_pc, self.obj_root_tensor[:, :7], self.tip_rb_tensor[:,:7], self.gripper_length)
        open_reward = (self.gripper_length + 0.1*torch.clamp(-action[:, -1].mean(dim=-1), min=-0.5, max=0.)) * (~visual_is_grasp)

        # close gripper 
        close_reward = (1.5+torch.clamp(-action[:, -1].mean(dim=-1), min=-1, max=0.5)) * visual_is_grasp 

        # # rotation reward
        # obj_rot = quat_to_mat(self.obj_root_tensor[:, 3:7])
        # hand_rot = quat_to_mat(self.tip_rb_tensor[..., 3:7]) 
        # ho_delta_rot = torch.matmul(obj_rot.transpose(-1, -2), hand_rot)
        # rot_reward = np.pi - mat_diff_rad(ho_delta_rot, self.goal_delta_rot)

        # grasp reward
        force_is_grasped, force_one_finger_grasp = self.compute_force_grasp()
        one_finger_grasp = force_one_finger_grasp & visual_is_grasp
        is_grasped = force_is_grasped & visual_is_grasp
        action_penalty = - action[:, :-1].abs().mean(dim=-1) * visual_is_grasp * (~is_grasped)

        self.rew_buf = reaching_reward + open_reward + close_reward + action_penalty 

        # reaching-goal reward
        obj_to_goal_dist = (obj_pos[..., 2] - self.success_pos[..., 2]).abs()
        # obj_to_goal_dist = (obj_pos - self.success_pos).norm(dim=-1)
        reaching_goal_reward =  - obj_to_goal_dist
        self.rew_buf += torch.clamp(5+20*reaching_goal_reward, min=1) * is_grasped

        # success 
        is_obj_placed = obj_to_goal_dist <= self.goal_thresh
        self.rew_buf +=  5 * is_obj_placed * is_grasped
        
        # NOTE only reset when time out. Don't reset when success. 
        # If reset when success, will have BUG since reset_idx() won't update the observation.
        self.reset_buf = (self.progress_buf >= self.max_episode_length)

        self.extras['reaching_reward'] = reaching_reward
        self.extras["action_penalty"] = action_penalty
        self.extras["close_reward"] = close_reward
        self.extras["open_reward"] = open_reward
        self.extras["is_reached"] = is_reached
        self.extras['visual_is_grasp'] = visual_is_grasp
        self.extras["is_grasped"] = is_grasped
        self.extras["reaching_goal_reward"] = reaching_goal_reward
        self.extras["obj_movement"] = (self.obj_root_tensor[...,:3] - self.obj_default_root[:3]).norm(dim=-1)
        self.extras["raw_reward"] = self.rew_buf 
        self.extras["obj_height"] = self.obj_root_tensor[...,2]
        self.extras["obj_up_flag"] = self.obj_root_tensor[...,2] > 0.1
        self.extras["successes"] = is_obj_placed
        return 
    
    def compute_obj_in_gripper(self, obj_pc, obj_pose, gripper_pose,  gripper_length, width_thre=0.034, height_thre=0.017/2, gripper_origin=[0,0,0]):
        '''
        args:
            obj_pc: [env, n, 3] in obj coordinate space
            obj_pose: T = [env, :3], R = [env, 3:7]
            gripper_pose: T = [env, :3], R = [env, 3:7]
            gripper_length: [env]
        return:
            volume: [env] (about 0~0.05)
        '''
        eps = 0.002
        objR = quat_to_mat(obj_pose[:, 3:])
        gripperR = quat_to_mat(gripper_pose[:, 3:])
        obj_pc_world = torch.bmm(obj_pc, objR.transpose(-1,-2)) + obj_pose[:, :3].unsqueeze(1)
        obj_pc_gripper = torch.bmm(obj_pc_world-gripper_pose[:, :3].unsqueeze(1), gripperR)

        mask_y = ((obj_pc_gripper[:,:,1]-gripper_origin[1]).abs() < gripper_length.unsqueeze(-1)/2+eps)
        mask_z = ((obj_pc_gripper[:,:,2]-gripper_origin[2]).abs() < height_thre)

        clip_num = 5
        for i in range(clip_num):
            mid = - width_thre / 2 + (2*i+1) * width_thre / (clip_num*2)
            mask_x = (obj_pc_gripper[:,:,0]-mid).abs() < (width_thre / clip_num)
            mask = mask_x * mask_y * mask_z
            in_gripper_obj_y = obj_pc_gripper[:,:,1] * mask
            in_gripper_obj_z = obj_pc_gripper[:,:,2] * mask + (~mask) * gripper_origin[2]
            height_mask = (in_gripper_obj_z.max(dim=-1)[0] - in_gripper_obj_z.min(dim=-1)[0]) > 0.01
            if i == 0:
                in_gripper_length = deepcopy(torch.clamp(in_gripper_obj_y.max(dim=-1)[0] - in_gripper_obj_y.min(dim=-1)[0], max=0.05) * height_mask)
            else:
                in_gripper_length += deepcopy(torch.clamp(in_gripper_obj_y.max(dim=-1)[0] - in_gripper_obj_y.min(dim=-1)[0], max=0.05) * height_mask)

        in_gripper_length = in_gripper_length/clip_num
        visual_is_grasp = in_gripper_length > 0.045
        return in_gripper_length, visual_is_grasp

    def compute_observations(self):
        # NOTE: rotation of finger and hand are all the same
        # tip_rb_tensor[:, :3] is in the center of the finger tips.
        self.tip_rb_tensor = (self.ltip_rb_tensor + self.rtip_rb_tensor) / 2
        self.gripper_length = (self.ltip_rb_tensor[:, :3]-self.rtip_rb_tensor[:, :3]).norm(dim=-1)

        # 0-2: obj delta pos
        obj_curr_goal_delta_pos = (self.obj_root_tensor[:, :3]-self.success_pos) 
        # 3-5: ho delta pos
        ho_delta_pos = self.tip_rb_tensor[:,:3] - self.obj_root_tensor[:, :3]
        # 6-11: ho delta rot
        obj_rot = quat_to_mat(self.obj_root_tensor[:, 3:7])
        hand_rot = quat_to_mat(self.tip_rb_tensor[:, 3:7]) 
        ho_delta_rot = torch.matmul(obj_rot.transpose(-1, -2), hand_rot).reshape(-1,9)[:, :6] 

        # 12-15: gripper_length, visual_is_grasp, is_grasped
        _, visual_is_grasp = self.compute_obj_in_gripper(self.obj_pc, self.obj_root_tensor[:, :7], self.tip_rb_tensor[:, :7], self.gripper_length)
        force_is_grasped, _ = self.compute_force_grasp()
        is_grasped = force_is_grasped & visual_is_grasp
        input_extra = torch.stack([self.gripper_length, visual_is_grasp, is_grasped], dim=-1)
        
        # oracle infos
        oracle_obs = torch.cat([obj_curr_goal_delta_pos, ho_delta_pos, ho_delta_rot, input_extra], dim=-1)
        oracle_obs = torch.clamp(oracle_obs, -self.clip_obs, self.clip_obs)
        self.obs_buf['oracle_state'] = oracle_obs
        
        # easiest info: gripper tip pose, gripper length, obj pose
        normal_obs = torch.cat([self.tip_rb_tensor[:,:3],self.tip_rb_tensor[:,3:7]*0, self.gripper_length.unsqueeze(-1)*0, self.obj_root_tensor[:, :3], self.obj_root_tensor[:, 3:7]*0], dim=-1)
        normal_obs = torch.clamp(normal_obs, -self.clip_obs, self.clip_obs)
        self.obs_buf['normal_state'] = normal_obs

        # compute tsdf from mesh
        if self.learn_input_mode == 'mesh_tsdf':
            rot, pos = self.compute_scene_pose()
            ### q
            if self.base_cfg['pipeline_mode'] in ["fixed_nerf-rl", "nerf-rl"]:
                pos_obj = pos.cpu().numpy()  # (num_envs, 12, 3))
                rot_obj = rot.cpu().numpy()  # (num_envs, 12, 3, 3)

                self.shm_dict["sceneid"][0:self.shm_dict["shm_sceneid_size"]] = (np.array(list(range(pos_obj.shape[0])), np.float32) + np.full(pos_obj.shape[0],pos_obj.shape[0] * (self.curr_iter),np.float32)).reshape(-1).tobytes()
                self.shm_dict["lock"][0:self.shm_dict["shm_lock_size"]] = np.full(pos_obj.shape[0], self.step_id + 5,np.float32).reshape(-1).tobytes()

                print("curr_iter: ", self.curr_iter, ", step: ", self.step_id, ", scenes: ", (np.array(list(range(pos_obj.shape[0])), np.float32) + np.full(pos_obj.shape[0],pos_obj.shape[0] * (self.curr_iter),np.float32)))

                RT_obj = np.concatenate([rot_obj, pos_obj.reshape(pos_obj.shape[0], pos_obj.shape[1], 3, 1)],3)  # (num_envs, num_poses, 3, 4)
                vector_zero_one = np.repeat(np.array([[0, 0, 0, 1]])[np.newaxis, np.newaxis, ...],pos_obj.shape[0], 0)  # (num_envs, 1, 1, 4)
                vector_zero_one = np.repeat(vector_zero_one, pos_obj.shape[1], 1)  # (num_envs, num_poses, 1, 4)
                RT_obj = np.concatenate([RT_obj, vector_zero_one], 2)  # (num_envs, num_poses, 4, 4)
                # print(RT_obj.shape, np.array(RT_obj, np.float32).swapaxes(0, 1).shape)
                state_pose = np.array(RT_obj, np.float32).reshape(-1).tobytes()  # (num_envs*num_poses*4*4)
                self.shm_dict["pose"][0:self.shm_dict["shm_pose_size"]] = state_pose

                t1 = time.time()
                # 等client读完给反馈，继续执行
                while True:
                    # print(np.array(buf[0:shm_lock_size].cast('f'), np.float32))
                    # when the sum of values in buf_lock == 0, read TSDFs from buf_tsdf
                    flag_sum = np.sum(np.array(self.shm_dict["lock"][0:self.shm_dict["shm_lock_size"]].cast('f'), np.float32))
                    if flag_sum == 0:
                        if self.base_cfg['pipeline_mode'] == "fixed_nerf-rl":
                            tsdf_all = np.array(self.shm_dict["tsdf"][0:self.shm_dict["shm_tsdf_resolution"]].cast('f'),np.float32).reshape(-1, self.shm_cfg['tsdf_resolution']*self.shm_cfg['tsdf_resolution']*self.shm_cfg['tsdf_resolution'])
                            self.mesh2TSDF.initialize_sdf(tsdf_all)
                            print("OK, receive tsdf, continue", tsdf_all.shape, self.mesh2TSDF.init_tsdf.shape)
                        else:
                            rgb_all = np.array(self.shm_dict["rgb"][0:self.shm_dict["shm_rgb_size"]].cast('f'),np.float32).reshape(-1, self.shm_dict["view_num"],self.shm_cfg['img_height'],self.shm_cfg['img_width'], 3)
                            # cv2.imwrite("./robot_men/robot_men_%d_%d.png"%(epoch, step),cv2.cvtColor(img_all[0]*255,cv2.COLOR_RGB2BGR))
                            print("OK, receive rgbs, continue", rgb_all.shape)
                        break
                print("step_id: %d, time: %.4f" % (self.step_id, time.time() - t1))

                self.obs_buf['mesh_tsdf'] = self.mesh2TSDF.query_tsdf(rot[:,:-1,:,:], pos[:,:-1,:]).reshape(self.num_envs, -1)
            else:
                self.obs_buf['mesh_tsdf'] = self.mesh2TSDF.query_tsdf(rot, pos).reshape(self.num_envs, -1)

            # save as point cloud txt
            if self.base_cfg['pipeline_mode'] in ["fixed_nerf-rl", "nerf-rl"]:
                scene_id_list = np.array(list(range(pos_obj.shape[0])), np.float32) + np.full(pos_obj.shape[0],pos_obj.shape[0] * self.curr_iter,np.float32)
                for i in range(len(scene_id_list)):
                    scene_id = scene_id_list[i]
                    tsdf_grid = self.obs_buf['mesh_tsdf'][i].cpu().numpy().reshape(self.shm_cfg['tsdf_resolution'],
                                                                                   self.shm_cfg['tsdf_resolution'],
                                                                                   self.shm_cfg['tsdf_resolution'])
                    tsdf_grid_unmerge = tsdf_all[i].reshape(self.shm_cfg['tsdf_resolution'],
                                                            self.shm_cfg['tsdf_resolution'],
                                                            self.shm_cfg['tsdf_resolution'])
                    tsdf_save_dir = os.path.join("/home/qiyudai/Code/DynamicRLNeRF/RL-NeRF/Mitsuba",self.shm_cfg['save_root_path'], "scene_%05d" % scene_id,"step_%05d" % self.step_id, "tsdf")
                    if not os.path.exists(tsdf_save_dir):
                        os.makedirs(tsdf_save_dir)
                    ind = np.transpose(((tsdf_grid > -0.2).astype(np.bool) & (tsdf_grid < 0.2).astype(np.bool)).nonzero())
                    ind_unmerge = np.transpose(((tsdf_grid_unmerge > -0.2).astype(np.bool) & (tsdf_grid_unmerge < 0.2).astype(np.bool)).nonzero())
                    np.savetxt(f"{tsdf_save_dir}/surface_pred.txt", ind)
                    np.savetxt(f"{tsdf_save_dir}/surface_pred_unmerge.txt", ind_unmerge)
                    np.savez_compressed(f"{tsdf_save_dir}/surface_pred_unmerge.npz", grid=tsdf_grid_unmerge)
                    np.save(f"{tsdf_save_dir}/RT_obj.npy", RT_obj[i])
            ### q

        if self.add_proprio_obs:
            # prop info: gripper tip pose, gripper length
            proprio_obs = torch.cat([self.tip_rb_tensor[:,:7], self.gripper_length.unsqueeze(-1), is_grasped.unsqueeze(-1)], dim=-1)
            proprio_obs = torch.clamp(proprio_obs, -self.clip_obs, self.clip_obs) 
            if self.temporal_proprio is None:
                self.temporal_proprio = proprio_obs
                # self.temporal_proprio = torch.cat([proprio_obs,proprio_obs,proprio_obs,proprio_obs,proprio_obs],dim=1)
            else:
                self.temporal_proprio = torch.cat([self.temporal_proprio[:, proprio_obs.shape[-1]:],proprio_obs],dim=1)
            self.obs_buf['proprio_state'] = self.temporal_proprio 
            self.obs_buf[self.learn_input_mode] = torch.cat((self.obs_buf[self.learn_input_mode], self.temporal_proprio), dim=-1)

        return 
    
    ###########################################################
    ######       take one step
    ###########################################################

    def pre_physics_step(self, actions):
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions) 
        if self.cfg["driveMode"] == "pos":
            self.pos_act[..., :-2] = self.franka_dof_tensor[..., :-2, 0] + actions[:,:-1] * self.dt * 20
            self.pos_act[..., -2:-1] = self.franka_dof_tensor[..., -2:-1, 0] + actions[:,-1:] * self.dt
            self.pos_act[..., -1:] = self.franka_dof_tensor[..., -1:, 0] + actions[:,-1:] * self.dt 
            self.pos_act = tensor_clamp(self.pos_act, self.franka_dof_lower_limits_tensor, self.franka_dof_upper_limits_tensor)
        elif self.cfg['driveMode'] == 'ik':  
            pos_err = actions[..., :3] * self.dt *2          # 2 is very important!!!
            orn_err = actions[..., 3:6] * self.dt 
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            self.pos_act[:, :-2] = self.franka_dof_tensor[...,:-2, 0]  + self.control_ik(dpose)
            self.pos_act[:, -2:-1] = self.franka_dof_tensor[...,-2:-1, 0] + actions[..., -1:] * self.dt / 5
            self.pos_act[:, -1:] = self.franka_dof_tensor[...,-1:, 0] + actions[..., -1:] * self.dt / 5
            self.pos_act = tensor_clamp(self.pos_act, self.franka_dof_lower_limits_tensor, self.franka_dof_upper_limits_tensor)
        elif self.cfg['driveMode'] == 'heuristic':

            # NOTE based on heuristic and don't use the action
            if not self.phase1:
                grasp_pose = torch.zeros((self.num_envs, 7), device=self.device)
                grasp_pose[:, 4] = 1
                grasp_pose[:, 2] = 0.25
                pos_err = grasp_pose[:, :3] - self.tip_rb_tensor[:,:3]
                orn_err = orientation_error(grasp_pose[:, 3:], self.tip_rb_tensor[:,3:7])

                dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
                self.pos_act[:, :-2] = self.franka_dof_tensor[...,:-2, 0]  + self.control_ik(dpose)
                self.pos_act[:, -2:] = self.franka_dof_tensor[...,-2:, 0]
                if pos_err.norm(dim=-1).mean() < 0.001 and orn_err.mean() < 0.001:
                    self.phase1 = True 
                    print(self.dof_state_tensor[0,:,0])
                    exit(1)
            else:
                grasp_pose = torch.zeros((self.num_envs, 7), device=self.device)
                grasp_pose[:, 4] = 1
                grasp_pose[:, 2] = 0.12
                pos_err = grasp_pose[:, :3] - self.tip_rb_tensor[:,:3]
                orn_err = orientation_error(grasp_pose[:, 3:], self.tip_rb_tensor[:,3:7])
                dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
                self.pos_act[:, :-2] = self.franka_dof_tensor[...,:-2, 0]  + self.control_ik(dpose)
                if pos_err.norm(dim=-1).mean() < 0.005:
                    print('closing!')
                    self.pos_act[:, -2:] = self.franka_dof_tensor[...,-2:, 0] - 0.001
                else:
                    self.pos_act[:, -2:] = self.franka_dof_tensor[...,-2:, 0]
        else:
            raise NotImplementedError
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_act))

        return

    def reset_idx(self, to_reset) :
        """
        reset env_ids where to_reset[env_ids] = 1
        """
        env_ids = torch.nonzero(to_reset).squeeze(-1)
        if len(env_ids) == 0:
            return 

        reset_dof_states = self.dof_state_tensor 
        reset_dof_states[env_ids, :self.num_franka_dofs, 0] = self.franka_default_dof_pos
        reset_dof_states[env_ids, :self.num_franka_dofs, 1] = 0
        reset_root_states = self.root_tensor
        reset_root_states[env_ids, 1, :7] =  self.obj_default_root
        reset_root_states[env_ids, 0, :7] =  self.franka_default_root
        reset_root_states[env_ids, :, 7:] =  0

        obj_reset_pos_tensor = reset_root_states[:, 1, :3]
        obj_reset_rot_tensor = reset_root_states[:, 1, 3:7]
        
        # random initialize the object
        if self.random_reset:
            obj_reset_pos_tensor[...,:-1] += torch.rand(obj_reset_pos_tensor[...,:-1].shape, device=self.device)*0.3 - 0.15
            random_rot = torch.zeros_like(obj_reset_rot_tensor, device=self.device)
            random_angle = torch.rand(obj_reset_rot_tensor.shape[:-1], device=self.device) * np.pi * 2 - np.pi 
            random_rot[...,-2] = torch.sin(random_angle)
            random_rot[...,-1] = torch.cos(random_angle)
            obj_reset_rot_tensor = quat_mul(obj_reset_rot_tensor, random_rot)
           
        # only reset hand dof
        dof_ind = self.global_indices[env_ids, 0].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(reset_dof_states),
                            gymtorch.unwrap_tensor(dof_ind), len(dof_ind))
        self.pos_act[env_ids, :] = reset_dof_states[:,:,0].contiguous()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.pos_act),
                            gymtorch.unwrap_tensor(dof_ind), len(dof_ind))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.eff_act),
                            gymtorch.unwrap_tensor(dof_ind),len(dof_ind))
        # reset obj and hand root
        root_ind = self.global_indices[env_ids, :].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(reset_root_states),
                            gymtorch.unwrap_tensor(root_ind), len(root_ind))     
        
        self.success_pos[env_ids] = self.compute_obj_goal_pos(obj_reset_pos_tensor, len(env_ids))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.phase1=False   # for debug
        self.temporal_proprio = None 
        return 
    
