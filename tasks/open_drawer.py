
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from tasks.hand_base import BaseTask
from utils.torch_jit_utils import *

import random
import numpy as np
import torch
from copy import deepcopy
import os
from os.path import join as pjoin
import json 

class open_drawer(BaseTask):
    def __init__(self, cfg, sim_params=None):
       
        self.part_bbox_init = []
        self.part_axis_xyz_init = []
        self.part_axis_dir_init = []
        self.part_joint_upper_limits = []
        self.part_joint_lower_limits = []
        self.obj_dof_num_lst = []
        self.obj_rigid_num_lst = []
        self.obj_target_dof_id_lst = [] 
        self.obj_target_link_id_lst = []    
        self.obj_target_handle_id_lst = []  
        self.obj_asset_lst = [] # [num_objs]
        self.obj_lstid_lst = [] # [num_envs]
        self.obj_scale = 0.5    # TODO? randomize 
        
        # object path list. split: 'train', 'valIntra' or 'valInter'.
        self.split = cfg['asset']['splits']
        obj_folder = pjoin(cfg['asset']['assetRoot'], 'objs/drawer', self.split)
        self.obj_lst = [pjoin(obj_folder, path) for path in os.listdir(obj_folder)]
        random.shuffle(self.obj_lst)

        if cfg['save_video']:
            st = 10
            self.obj_lst = self.obj_lst[st:st+2] # debug

        # default obj and hand settings
        self.obj_default_root = torch.tensor([-0.6, 0, 0.5, 0, 0, 1, 0], device=cfg['device'], dtype=torch.float)
        self.random_reset = cfg["random_reset"]
        self.reset_t_range = 0.05
        self.reset_r_range = np.pi / 12

        super().__init__(cfg, sim_params)

        # from simulator acquire tensors
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof_state_tensor_all = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.rigid_body_tensor_all = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)         # (env, 2, 13) 0:hand, 1:object, 
       
        # choose useful dof_state and rigid_body
        self.dof_state_mask = torch.zeros((self.num_envs, self.robot.num_dofs + 1), device = self.device).long()
        self.rigid_body_mask = torch.zeros((self.num_envs, self.robot.num_rigid_body + 2), device = self.device).long()
        dof_count = 0
        rigid_count = 0
        for i in range(self.num_envs):
            obj_lstid = self.obj_lstid_lst[i] 
            self.dof_state_mask[i, :self.robot.num_dofs] = torch.arange(dof_count, dof_count+self.robot.num_dofs, device = self.device)
            self.dof_state_mask[i, -1] = dof_count+self.robot.num_dofs+self.obj_target_dof_id_lst[obj_lstid]
            self.rigid_body_mask[i, :self.robot.num_rigid_body] = torch.arange(rigid_count, rigid_count+self.robot.num_rigid_body, device = self.device)
            self.rigid_body_mask[i, -2] =  rigid_count+self.robot.num_rigid_body+self.obj_target_link_id_lst[obj_lstid]
            self.rigid_body_mask[i, -1] =  rigid_count+self.robot.num_rigid_body+self.obj_target_handle_id_lst[obj_lstid]
            dof_count += self.robot.num_dofs + self.obj_dof_num_lst[obj_lstid]
            rigid_count += self.robot.num_rigid_body + self.obj_rigid_num_lst[obj_lstid]
        
        # NOTE self.dof_state_tensor_all will updated automatically but self.dof_state_tensor won't. Only used for first reset. 
        self.dof_state_tensor = self.dof_state_tensor_all[self.dof_state_mask]    # (env, 10, 2)   0-8:hand, 9:obj  
        
        # stack 
        self.part_bbox_init = torch.stack(self.part_bbox_init, dim=0).to(self.device)[self.obj_lstid_lst]
        self.part_axis_xyz_init = torch.stack(self.part_axis_xyz_init, dim=0).to(self.device)[self.obj_lstid_lst]
        self.part_axis_dir_init = torch.stack(self.part_axis_dir_init, dim=0).to(self.device)[self.obj_lstid_lst]
        self.part_joint_upper_limits = torch.stack(self.part_joint_upper_limits, dim=0).to(self.device)[self.obj_lstid_lst] * self.obj_scale
        self.part_joint_lower_limits = torch.stack(self.part_joint_lower_limits, dim=0).to(self.device)[self.obj_lstid_lst]
        self.obj_lstid_lst = torch.tensor(self.obj_lstid_lst).to(self.device)

        # create useful variables
        self.suc_prop = 0.5
        self.success = torch.zeros(self.num_envs, device=self.device).bool()
        # self.last_succ_time = torch.zeros(self.num_envs, device=self.device).int()
        self.succ_objid_lst = torch.zeros(len(self.obj_asset_lst), device=cfg['device']).bool()
        self.pos_act = torch.zeros((self.num_envs, self.robot.num_dofs), device=self.device)
        self.pos_act_all = torch.zeros_like(self.dof_state_tensor_all[:, 0], device=self.device)
        self.global_indices = torch.arange(self.num_envs*2, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        

        # initialize
        self.reset(type="init")

        return 

    def preload_all_obj(self):
        for obj_id in range(len(self.obj_lst)):

            # choose obj path
            root_path, obj_path_name = os.path.split(self.obj_lst[obj_id]) 
            asset_id, link_name, handle_name, joint_name = obj_path_name.split('-')[-5:-1]
            
            # read annotations for bbox and axis
            with open(pjoin(root_path, obj_path_name, 'bbox_info.json'), 'rb') as f:
                bbox_info = json.load(f)
            link_id, handle_id = bbox_info['link_name'].index(link_name), bbox_info['link_name'].index(handle_name)
            bbox_obj = torch.tensor(bbox_info['bbox_world'][handle_id])*self.obj_scale
            self.part_bbox_init.append(bbox_obj)
            self.part_axis_xyz_init.append(torch.tensor(bbox_info['axis_xyz_world'][link_id]))
            self.part_axis_dir_init.append(torch.tensor(bbox_info['axis_dir_world'][link_id]))
            self.obj_target_link_id_lst.append(link_id)
            self.obj_target_handle_id_lst.append(handle_id)

            # load obj asset
            obj_opts = gymapi.AssetOptions()
            obj_opts.disable_gravity = True     # if not disabled, it will need a very initial large force to open a drawer
            obj_opts.fix_base_link = True 
            obj_opts.collapse_fixed_joints = False
            obj_opts.convex_decomposition_from_submeshes = True
            obj_opts.armature = 0.005
            obj_asset = self.gym.load_asset(self.sim, root_path, pjoin(obj_path_name, 'mobility_new.urdf'), obj_opts)
            self.obj_asset_lst.append(obj_asset)

            # save useful properties
            obj_dof_props = self.gym.get_asset_dof_properties(obj_asset)
            obj_dof_dict = self.gym.get_asset_dof_dict(obj_asset)
            obj_dof_num = self.gym.get_asset_dof_count(obj_asset)
            obj_rigid_num = len(self.gym.get_asset_rigid_body_dict(obj_asset))
            obj_target_dof_id = obj_dof_dict[joint_name]
            self.obj_dof_num_lst.append(obj_dof_num)
            self.obj_rigid_num_lst.append(obj_rigid_num)
            self.obj_target_dof_id_lst.append(obj_target_dof_id)
            self.part_joint_upper_limits.append(torch.tensor(obj_dof_props["upper"][obj_target_dof_id]))
            self.part_joint_lower_limits.append(torch.tensor(obj_dof_props["lower"][obj_target_dof_id]))
        
            print(obj_id, obj_path_name)

        return 

    def load_obj(self, env_ptr, env_id):
        # # choose obj path
        split = 'train' # TODO: add val
        obj_lstid = env_id % len(self.obj_lst)
        self.obj_lstid_lst.append(obj_lstid)

        # root_path, obj_path_name = os.path.split(self.obj_lst[obj_lstid]) 
        # asset_id, link_name, handle_name, joint_name = obj_path_name.split('-')[-5:-1]

        # set initial obj pose and scale
        obj_transform = gymapi.Transform()
        obj_transform.p = gymapi.Vec3(self.obj_default_root[0], self.obj_default_root[1], self.obj_default_root[2])
        obj_transform.r = gymapi.Quat(self.obj_default_root[3],self.obj_default_root[4],self.obj_default_root[5],self.obj_default_root[6])

        # create object actor
        obj_asset = self.obj_asset_lst[obj_lstid]
        obj_actor = self.gym.create_actor(env_ptr, obj_asset, obj_transform, "cabinet{}".format(env_id), env_id, 1, 0)
        self.gym.set_actor_scale(env_ptr, obj_actor, self.obj_scale)

        # set physical props
        obj_dof_props = self.gym.get_asset_dof_properties(obj_asset)
        obj_dof_props['stiffness'][:] = 0.0 
        obj_dof_props['damping'][:] = 20.0      # large damping can reduce interia(?)
        obj_dof_props['friction'][:] = 5.0 
        obj_dof_props["driveMode"][:] = gymapi.DOF_MODE_NONE
        self.gym.set_actor_dof_properties(env_ptr, obj_actor, obj_dof_props)
        return 
    
    def compute_reward(self, action):
        # handle information
        handle_out = self.part_bbox[:, 0] - self.part_bbox[:, 4]
        handle_long = self.part_bbox[:, 1] - self.part_bbox[:, 0]
        handle_short = self.part_bbox[:, 3] - self.part_bbox[:, 0]
        handle_mid_point = (self.part_bbox[:, 0] + self.part_bbox[:, 6]) / 2
        handle_out_length = torch.norm(handle_out, dim = -1)
        handle_long_length = torch.norm(handle_long, dim = -1)
        handle_short_length = torch.norm(handle_short, dim = -1)
        handle_shortest = torch.min(torch.min(handle_out_length, handle_long_length), handle_short_length)
        handle_out = handle_out / handle_out_length.unsqueeze(-1)
        handle_long = handle_long / handle_long_length.unsqueeze(-1)
        handle_short = handle_short / handle_short_length.unsqueeze(-1)

        # reaching
        tcp_to_obj_delta = self.robot.tip_rb_tensor[:,:3] - handle_mid_point
        tcp_to_obj_dist = tcp_to_obj_delta.norm(dim=-1)
        is_reached_out = (tcp_to_obj_delta * handle_out).sum(dim=-1).abs() < handle_out_length / 2
        short_ltip = ((self.robot.ltip_rb_tensor[:,:3] - handle_mid_point) * handle_short).sum(dim=-1) 
        short_rtip = ((self.robot.rtip_rb_tensor[:,:3] - handle_mid_point) * handle_short).sum(dim=-1)
        is_reached_short = (short_ltip * short_rtip) < 0
        is_reached_long = (tcp_to_obj_delta * handle_long).sum(dim=-1).abs() < handle_long_length / 2
        is_reached = is_reached_out & is_reached_short & is_reached_long
        reaching_reward = - tcp_to_obj_dist + 0.1 * (is_reached_out + is_reached_short + is_reached_long)

        # rotation reward
        hand_rot = self.robot.tip_rb_tensor[..., 3:7]
        hand_grip_dir = quat_axis(hand_rot, 2)
        hand_sep_dir = quat_axis(hand_rot, 1)
        hand_down_dir = quat_axis(hand_rot, 0)

        dot1 = (-hand_grip_dir * handle_out).sum(dim=-1)
        dot2 = torch.max((hand_sep_dir * handle_short).sum(dim=-1), (-hand_sep_dir * handle_short).sum(dim=-1)) 
        dot3 = torch.max((hand_down_dir * handle_long).sum(dim=-1), (-hand_down_dir * handle_long).sum(dim=-1))
        rot_reward = dot1 + dot2 + dot3 - 3

        # close reward 
        close_reward = (0.1 - self.robot.gripper_length) * is_reached + 0.1*(self.robot.gripper_length-0.1) * (~is_reached)

        grasp_success = is_reached & (self.robot.gripper_length < handle_short_length + 0.01) & (rot_reward > -0.2)

        # open drawer reward
        joint_state_reward = grasp_success * (0.1 +torch.clamp((self.dof_state_tensor[:, -1, 0]-self.part_joint_lower_limits) / self.part_joint_upper_limits,max=self.suc_prop))
        is_open = grasp_success * ((self.dof_state_tensor[:, -1, 0]-self.part_joint_lower_limits) / self.part_joint_upper_limits > 0.1)
        is_open_notgrasp = ((self.dof_state_tensor[:, -1, 0]-self.part_joint_lower_limits) / self.part_joint_upper_limits > 0.1)
        self.rew_buf = reaching_reward + 0.5*rot_reward + 5*close_reward + 5*joint_state_reward 
        self.rew_buf = self.rew_buf + self.rew_buf.abs() * rot_reward
        
        # if self.robot.mobile and self.robot.driveMode == 'ik':
        #     action_penalty = torch.clamp((action[:, :3] * action[:, 3:6]).sum(dim=-1), max=0)
        #     self.rew_buf += action_penalty       
        # else:
        #     action_penalty = action[:, 0] * 0

        # print(reaching_reward[0], rot_reward[0], close_reward[0], joint_state_reward[0])

        # success 
        self.success = grasp_success * ((self.dof_state_tensor[:, -1, 0]-self.part_joint_lower_limits) >= self.suc_prop * self.part_joint_upper_limits)
        # self.last_succ_time = self.last_succ_time + (self.last_succ_time != 0).int() + ((self.last_succ_time == 0) & self.success).int() # for reset
        succ_env = torch.nonzero(self.success==1).squeeze(-1)
        # print(succ_env[:6])
        if len(succ_env) != 0:
            self.succ_objid_lst[self.obj_lstid_lst[succ_env]] = 1

        self.rew_buf += 2 * self.success
        
        self.extras['is_open'] = is_open
        self.extras['is_open_notgrasp'] = is_open_notgrasp
        self.extras['reaching_reward'] = reaching_reward
        self.extras["close_reward"] = close_reward
        self.extras["rot_reward"] = rot_reward
        self.extras["is_reached"] = is_reached
        # self.extras["action_penalty"] = action_penalty
        self.extras["joint_state_reward"] = joint_state_reward 
        self.extras["raw_reward"] = self.rew_buf 
        self.extras["is_grasped"] = grasp_success.float()
        self.extras["success_objnum"] = self.succ_objid_lst
        self.extras["step_id"] = deepcopy(self.progress_buf).float()
        return 

    def compute_observations(self, type="step"):
        self.dof_state_tensor = self.dof_state_tensor_all[self.dof_state_mask]    # (env, 10, 2)   0-8:hand, 9:obj
        self.rigid_body_tensor = self.rigid_body_tensor_all[self.rigid_body_mask] # (env, 13, 13) 0-11:hand, 12-13: obj
        self.obj_root_tensor = self.root_tensor[:, self.obj_actor, :]       # (num_env, 13)
        
        self.robot.update_state(self.rigid_body_tensor, self.dof_state_tensor)

        # compute bbox 
        part_bbox_objcoord = self.part_bbox_init + self.dof_state_tensor[:, -1:, 0:1] * self.part_axis_dir_init.reshape(-1,1,3) # [env, 8, 3] 
        self.part_bbox = torch.matmul(part_bbox_objcoord, quat_to_mat(self.obj_root_tensor[:, 3:7]).transpose(-1,-2)) + self.obj_root_tensor[:, None, :3] # [env, 8, 3]
        handle_out = self.part_bbox[:, 0] - self.part_bbox[:, 4]
        handle_long = self.part_bbox[:, 1] - self.part_bbox[:, 0]
        handle_short = self.part_bbox[:, 3] - self.part_bbox[:, 0]
        handle_mid_point = (self.part_bbox[:, 0] + self.part_bbox[:, 6]) / 2
        handle_out_length = torch.norm(handle_out, dim = -1).unsqueeze(-1)
        handle_long_length = torch.norm(handle_long, dim = -1).unsqueeze(-1)
        handle_short_length = torch.norm(handle_short, dim = -1).unsqueeze(-1)
        handle_out = handle_out / handle_out_length
        handle_long = handle_long / handle_long_length
        handle_short = handle_short / handle_short_length

        handle_rot = torch.stack([handle_out, handle_long, -handle_short], dim=-1)
        robot_relative_pos = torch.bmm(handle_rot.transpose(-1,-2), (self.robot.tip_pos - handle_mid_point).unsqueeze(-1))
        robot_relative_rot = torch.bmm(handle_rot.transpose(-1,-2), self.robot.tip_rot_9d)
        # env_id = 0
        # print(self.dof_state_tensor[0, -1:, 0], self.grasp_success[0], robot_relative_pos[env_id].norm(dim=-1))
        # print(robot_relative_rot[env_id], handle_rot[env_id])
        normal_obs = torch.cat([self.robot.tip_rb_tensor,
        # normal_obs = torch.cat([robot_relative_pos.squeeze(-1), robot_relative_rot.reshape(-1, 9), self.robot.tip_rb_tensor[:,7:],
        # normal_obs = torch.cat([self.robot.tip_pos, self.robot.tip_rot_9d.reshape(-1, 9),
                                handle_mid_point, 
                                handle_out, handle_short, handle_long, 
                                handle_out_length, handle_long_length, handle_short_length, 
                                self.robot.dof_qpos_normalized, self.robot.dof_qvel_raw, self.dof_state_tensor[:, -1:, 0]], dim=-1)
        self.obs_buf['normal_state'] = normal_obs
        # np.savetxt('debug.txt', self.obs_buf['depth_pc'][0].cpu().numpy().reshape(-1,3))
        # np.savetxt('debug_bbox.txt', self.part_bbox[0].cpu().numpy())
        # exit(1)

        return 
    

    def reset_idx(self, to_reset) :
        """
        Reset envs where to_reset[env_ids] = 1.
        For to_reset[env_ids] = 1, set dof + action + root.
        For to_reset[env_ids] != 1, set action.
        """
        env_ids = torch.nonzero(to_reset).squeeze(-1)
        if len(env_ids) == 0:
            raise NotImplementedError 

        reset_root_states = self.root_tensor
        reset_root_states[env_ids, 1, :7] =  self.obj_default_root
        reset_root_states[env_ids, 0, :7] =  self.robot.default_root
        reset_root_states[env_ids, :, 7:] =  0

        reset_dof_states = self.dof_state_tensor
        reset_dof_states[env_ids, :self.robot.num_dofs, 0] = self.robot.default_dof_pos
        reset_dof_states[env_ids, -1, 0] = self.part_joint_lower_limits[env_ids]
        reset_dof_states[env_ids, :, 1] = 0

        if self.random_reset:
            reset_root_states[env_ids, 1, :3] += torch.rand(reset_root_states[env_ids, 1, :3].shape, device=self.device)*self.reset_t_range*2 - self.reset_t_range
            random_rot = torch.zeros_like(reset_root_states[env_ids, 1, 3:7], device=self.device)
            random_angle = torch.rand(reset_root_states[env_ids, 1, 3:7].shape[:-1], device=self.device) * self.reset_r_range * 2 - self.reset_r_range
            random_rot[..., -2] = torch.sin(random_angle)
            random_rot[..., -1] = torch.cos(random_angle)
            reset_root_states[env_ids, 1, 3:7] = quat_mul(reset_root_states[env_ids, 1, 3:7], random_rot)

        # reset hand and obj dof. Indexed!
        dof_ind = self.global_indices[env_ids, :].flatten()
        self.dof_state_tensor_all[self.dof_state_mask, :] = reset_dof_states
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state_tensor_all),
                            gymtorch.unwrap_tensor(dof_ind), len(dof_ind))

        # overlap the action. Can't use Indexed! Because those envs that don't need to reset should take actions.
        self.pos_act[env_ids, :] = reset_dof_states[env_ids,:self.robot.num_dofs,0]
        self.pos_act_all[self.dof_state_mask[:, :self.robot.num_dofs]] = self.pos_act
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_act_all))
        
        # reset obj and hand root. Indexed!
        root_ind = self.global_indices[env_ids, :].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(reset_root_states),
                            gymtorch.unwrap_tensor(root_ind), len(root_ind))     
        self.progress_buf[env_ids] = 0
        self.success[env_ids] = 0
        # self.last_succ_time[env_ids] = 0
        # self.reset_buf[env_ids] = 0 # Don't use it! Or will be wrong.
        self.epis_max_rew[env_ids] = -100
        self.epis_max_step[env_ids] = 0

        return 
    
