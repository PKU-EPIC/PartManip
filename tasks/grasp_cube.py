
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from tasks.hand_base import BaseTask
from utils.torch_jit_utils import *

import numpy as np
import torch
from copy import deepcopy
from os.path import join as pjoin 

class grasp_cube(BaseTask):
    def __init__(self, cfg, sim_params=None):
        self.obj_loaded = False
        
        self.reset_range = 0.15
        self.pose_lower_limit = torch.tensor([-self.reset_range, -self.reset_range, 0.0, -1, -1, -1, -1],
                                 device=cfg['device'], dtype=torch.float)
        self.pose_upper_limit = torch.tensor([self.reset_range, self.reset_range, 0.4, 1, 1, 1, 1],
                                 device=cfg['device'], dtype=torch.float)
        super().__init__(cfg, sim_params)

        # from simulator acquire tensors
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.rigid_body_tensor = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        # NOTE use net_contact_foce instead of force_sensor! force_sensor is unstable and can >0 even without touching object
        # self.contact_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        
        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)         # (env, 2, 13) 0:hand, 1:object, 
        self.dof_state_tensor = self.dof_state_tensor.view(self.num_envs, -1, 2)    # (env, 9, 2)   hand
        self.rigid_body_tensor = self.rigid_body_tensor.view(self.num_envs, -1, 13) # (env, 13, 13) 0-11:hand, 12: obj
        # self.contact_force_tensor = self.contact_force_tensor.view(self.num_envs, -1, 3) # (env, 13, 3) 0-11:hand, 12: obj

        # params of randomization
        self.random_reset = cfg["random_reset"]

        # create useful variables
        self.goal_thresh=0.025
        self.success_pos = torch.tensor([0, 0, 0.2], device=self.device)[None,:]
        self.success = torch.zeros(self.num_envs, device=self.device).bool()
        # self.last_succ_time = torch.zeros(self.num_envs, device=self.device).int()
        self.pos_act = torch.zeros((self.num_envs, self.robot.num_dofs), device=self.device)
        self.pos_act_all = torch.zeros(self.num_envs*self.robot.num_dofs, device=self.device)
        self.dof_state_mask = torch.arange(self.num_envs*self.robot.num_dofs, device=self.device).reshape(self.num_envs, -1)
        self.eff_act = torch.zeros((self.num_envs, self.robot.num_dofs), device=self.device)
        self.global_indices = torch.arange(self.num_envs*2, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        # initialize
        self.reset(type="init")

        return 

    def load_obj(self, env_ptr, env_id):
        if self.obj_loaded == False :
            # self.obj_scale = 0.05
            obj_opts = gymapi.AssetOptions()
            obj_opts.disable_gravity = False 
            obj_opts.density = 1000
            self.obj_asset = self.gym.load_asset(self.sim, self.asset_root, 'objs/cube/model.urdf', obj_opts)
            self.obj_loaded = True

            # set initial obj pose
            self.obj_default_root = torch.tensor([0, 0, 0.025, 0, 0, 0, 1], device=self.device, dtype=torch.float)

        self.obj_actor = self.gym.create_actor(env_ptr, self.obj_asset, gymapi.Transform(), "obj", env_id, 0, 0)

    def compute_reward(self, action):
        '''
        update self.rew_buf, self.reset_buf
        '''
        # reaching
        obj_pos = self.obj_root_tensor[:,:3] 
        tcp_to_obj_dist = (self.robot.tip_rb_tensor[:,:3] - obj_pos).norm(dim=-1)
        is_reached = (tcp_to_obj_dist < 0.02)
        reaching_reward = - tcp_to_obj_dist

        # close gripper 
        close_reward = (0.1 - self.robot.gripper_length) * is_reached  + 0.1*(self.robot.gripper_length-0.1) * (~is_reached)

        # rotation reward
        obj_rot = deambiguity_rotation(self.obj_root_tensor[:, 3:7])
        hand_rot = quat_to_mat(self.robot.tip_rb_tensor[..., 3:7]) 
        down = - hand_rot[:, -1, -1] 
        parallel1 = ((hand_rot[:,:,0]*obj_rot[:,:,0]).abs() + (hand_rot[:,:,1]*obj_rot[:,:,1]).abs()).sum(dim=-1) 
        parallel2 = ((hand_rot[:,:,0]*obj_rot[:,:,1]).abs() + (hand_rot[:,:,1]*obj_rot[:,:,0]).abs()).sum(dim=-1) 
        rot_reward = down + torch.max(parallel1, parallel2) - 3

        # reaching-goal reward
        obj_to_goal_dist = (obj_pos - self.success_pos).norm(dim=-1)
        reaching_goal_reward = torch.max(0.2 - obj_to_goal_dist, torch.zeros_like(obj_to_goal_dist).to(self.device)) * is_reached

        self.rew_buf = reaching_reward + 0.5*rot_reward + 5*close_reward  + 20*reaching_goal_reward 
        
        # success 
        is_obj_placed = obj_to_goal_dist <= self.goal_thresh
        self.success = is_obj_placed * is_reached
        # self.last_succ_time = self.last_succ_time + (self.last_succ_time != 0).int() + ((self.last_succ_time == 0) & self.success).int() # for reset
        self.rew_buf += 3*self.success

        # print(obj_to_goal_dist[0], tcp_to_obj_dist[0], self.rew_buf[0])
        # fail_env = torch.nonzero(self.success==0).squeeze(-1)
        # print(fail_env[:5])
        self.extras['reaching_reward'] = reaching_reward
        self.extras["close_reward"] = close_reward
        self.extras["rot_reward"] = rot_reward
        self.extras["is_reached"] = is_reached
        self.extras["reaching_goal_reward"] = reaching_goal_reward
        self.extras["obj_movement"] = (self.obj_root_tensor[...,:3] - self.obj_default_root[:3]).norm(dim=-1)
        self.extras["raw_reward"] = self.rew_buf 
        self.extras["obj_height"] = self.obj_root_tensor[...,2]
        self.extras["obj_up_flag"] = self.obj_root_tensor[...,2] > 0.1
        self.extras["step_id"] = deepcopy(self.progress_buf).float()
        return 


    def compute_observations(self,type="step"):
        self.robot.update_state(self.rigid_body_tensor, self.dof_state_tensor)
        self.obj_root_tensor = self.root_tensor[:, self.obj_actor, :]       # (num_env, 13)
        
        robot_tip_pose = (2 * (self.robot.tip_rb_tensor[:,:7]-self.pose_lower_limit[:])/(self.pose_upper_limit[:] - self.pose_lower_limit[:])) - 1
        obj_pos = (2 * (self.obj_root_tensor[:,:3]-self.pose_lower_limit[:3])/(self.pose_upper_limit[:3] - self.pose_lower_limit[:3])) - 1
        obj_pose = torch.cat([obj_pos, deambiguity_rotation(self.obj_root_tensor[:,3:7]).reshape(self.num_envs, -1)], dim=-1)
        normal_obs = torch.cat([robot_tip_pose, obj_pose, self.robot.dof_qpos_normalized, self.robot.dof_qvel_raw], dim=-1)
        self.obs_buf['normal_state'] = normal_obs

        # compute tsdf from mesh
        if self.learn_input_mode == 'mesh_tsdf':
            rot, pos = self.compute_scene_pose()
            self.obs_buf['mesh_tsdf'] = self.mesh2TSDF.query_tsdf(rot, pos).reshape(self.num_envs, -1)

   
        if self.add_proprio_obs and type != 'init':
            proprio_obs = torch.cat([robot_tip_pose, self.robot.dof_qpos_normalized, self.robot.dof_qvel_raw], dim=-1)
            self.obs_buf['proprio_state'] = proprio_obs
            self.obs_buf[self.learn_input_mode] = torch.cat((self.obs_buf[self.learn_input_mode], proprio_obs), dim=-1)

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
         
        reset_dof_states = self.dof_state_tensor 
        reset_dof_states[env_ids, :self.robot.num_dofs, 0] = self.robot.default_dof_pos
        reset_dof_states[env_ids, :self.robot.num_dofs, 1] = 0
        reset_root_states = self.root_tensor
        reset_root_states[env_ids, 1, :7] =  self.obj_default_root
        reset_root_states[env_ids, 0, :7] =  self.robot.default_root
        reset_root_states[env_ids, :, 7:] =  0

        # random initialize the object
        if self.random_reset:
            reset_root_states[env_ids, 1, :2] += torch.rand(reset_root_states[env_ids, 1, :2].shape, device=self.device)*self.reset_range*2 - self.reset_range
            random_rot = torch.zeros_like(reset_root_states[env_ids, 1, 3:7], device=self.device)
            random_angle = torch.rand(reset_root_states[env_ids, 1, 3:7].shape[:-1], device=self.device) * np.pi * 2 - np.pi 
            random_rot[..., -2] = torch.sin(random_angle)
            random_rot[..., -1] = torch.cos(random_angle)
            reset_root_states[env_ids, 1, 3:7] = quat_mul(reset_root_states[env_ids, 1, 3:7], random_rot)
            
        # only reset hand dof. Indexed!
        dof_ind = self.global_indices[env_ids, 0].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(reset_dof_states),
                            gymtorch.unwrap_tensor(dof_ind), len(dof_ind))

        # overlap the action. Can't use Indexed! Because those envs that don't need to reset should take actions.
        self.pos_act[env_ids, :] = reset_dof_states[env_ids,:self.robot.num_dofs,0].contiguous().clone()
        self.pos_act_all[self.dof_state_mask[:, :self.robot.num_dofs]] = self.pos_act
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_act_all))
        
        # reset obj and hand root. Indexed!
        root_ind = self.global_indices[env_ids, :].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(reset_root_states),
                            gymtorch.unwrap_tensor(root_ind), len(root_ind))     
        self.progress_buf[env_ids] = 0
        self.success[env_ids] = 0
        # self.last_succ_time[env_ids] = 0
        # self.reset_buf[env_ids] = 0 # Don't use it. Or will be wrong.
        self.epis_max_rew[env_ids] = -100
        self.epis_max_step[env_ids] = 0

        return 
    

