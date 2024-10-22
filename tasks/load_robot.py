from isaacgym import gymapi, gymtorch
from utils.torch_jit_utils import *
import torch 
from os.path import join as pjoin

class franka():
    def __init__(self, gym, robot_cfg, dt, num_envs, device):
        self.gym = gym 
        self.device = device
        self.num_envs = num_envs
        self.dt = dt 
        self.driveMode = robot_cfg['driveMode']
        self.asset_file = f"franka_description/robots/{robot_cfg['assetFile']}.urdf"

        if self.driveMode == 'ik':
            self.num_actions = 7
        elif self.driveMode == 'pos' or self.driveMode == 'ik_abs':
            self.num_actions = 8
        elif self.driveMode == 'heuristic': # only used to debug.
            self.num_actions = 1  
        else:
            raise NotImplementedError
        
        if 'mobile' in robot_cfg['assetFile']:
            self.num_actions += 3
            self.mobile = True
            print('The robot base CAN move!')
        else:
            self.mobile = False
            print('The robot base CANNOT move!')

        self.default_dof_pos = torch.tensor(robot_cfg['dof'], device=self.device, dtype=torch.float)
        self.default_root = torch.tensor(robot_cfg['root'], device=self.device, dtype=torch.float)
        return 
    
    def preload(self, sim, asset_root):
        self.sim = sim 
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = False 
        asset_options.thickness = 0.001
        self.robot_asset = self.gym.load_asset(sim, asset_root, self.asset_file, asset_options)

        # set initial dof
        self.num_dofs = self.gym.get_asset_dof_count(self.robot_asset)  # 9 if not mobile else 11
        self.num_rigid_body = len(self.gym.get_asset_rigid_body_dict(self.robot_asset))
        
        # for saving 6dof pose of robot joints
        self.coordinate_transform_matrix = torch.zeros((11,3,3), device=self.device)
        self.coordinate_transform_matrix[:,0,0] = 1
        self.coordinate_transform_matrix[:,1,2] = -1
        self.coordinate_transform_matrix[:,2,1] = 1
        self.coordinate_transform_matrix[-1,1,2] = 1   # I don't know why but it seems necessary

        dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        self.dof_max_torque_tensor = torch.tensor(dof_props['effort'], device=self.device) #(9,)
        self.dof_lower_limits_tensor = torch.tensor(dof_props['lower'], device=self.device)
        self.dof_upper_limits_tensor = torch.tensor(dof_props['upper'], device=self.device)
        self.action_tensor = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        return 
    
    def load_to_env(self, env_ptr, env_id):
        # set initial props
        robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        arm_stf = 1e6
        arm_dmp = 1e3
        gripper_stf = 1.0e4
        gripper_dmp = 1.0e2
        robot_dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
        if self.mobile:
            robot_dof_props['stiffness'][:self.mobile*3] = 1e6
            robot_dof_props['damping'][:self.mobile*3] = 1e3
        robot_dof_props['stiffness'][self.mobile*3:-2] = arm_stf
        robot_dof_props['damping'][self.mobile*3:-2] = arm_dmp
        robot_dof_props['stiffness'][-2:] = gripper_stf
        robot_dof_props['damping'][-2:] = gripper_dmp
        self.robot_actor = self.gym.create_actor(env_ptr, self.robot_asset, gymapi.Transform(), "franka", env_id, 2, 0)
        self.gym.set_actor_dof_properties(env_ptr, self.robot_actor, robot_dof_props)
        for rigid_body_index in range(12):
            self.gym.set_rigid_body_segmentation_id(env_ptr, self.robot_actor, rigid_body_index, 1)
        
        if env_id == 0:
            self.lfinger_rb_index = self.gym.find_actor_rigid_body_index(env_ptr, self.robot_actor, "panda_leftfinger", gymapi.DOMAIN_ENV)
            self.rfinger_rb_index = self.gym.find_actor_rigid_body_index(env_ptr,self.robot_actor, "panda_rightfinger", gymapi.DOMAIN_ENV)
            self.ltip_rb_index = self.gym.find_actor_rigid_body_index(env_ptr,self.robot_actor, "panda_lefttip", gymapi.DOMAIN_ENV)
            self.rtip_rb_index = self.gym.find_actor_rigid_body_index(env_ptr,self.robot_actor, "panda_righttip", gymapi.DOMAIN_ENV)
        
        return 

    def init_jacobian(self):
        self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "franka"))

    def control(self, raw_output):
        if self.mobile:
            dpose_base = raw_output[...,:3].unsqueeze(-1) * 0.005
            root_r = quat_to_mat(self.default_root[None, 3:7]).repeat(raw_output.shape[0],1,1)
            self.action_tensor[..., :3] = self.dof_qpos_raw[..., :3] + torch.bmm(root_r.transpose(-1,-2), dpose_base).squeeze(-1)
            raw_output = raw_output[..., 3:]

        if self.driveMode == "pos":
            self.action_tensor[..., self.mobile*3:-2] = self.dof_qpos_raw[..., self.mobile*3:-2] + raw_output[:,:-1] * self.dt * 20
            self.action_tensor[..., -2:-1] = self.dof_qpos_raw[..., -2:-1] + raw_output[:,-1:] * self.dt
            self.action_tensor[..., -1:] = self.dof_qpos_raw[..., -1:] + raw_output[:,-1:] * self.dt 
            self.action_tensor = tensor_clamp(self.action_tensor, self.dof_lower_limits_tensor, self.dof_upper_limits_tensor)
        elif self.driveMode == 'ik':  
            pos_err = raw_output[..., :3] * 0.005        
            orn_err = raw_output[..., 3:6] * 0.005
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            if self.mobile:
                dpose[:,:3] -= dpose_base
            self.action_tensor[:, self.mobile*3:-2] = self.dof_qpos_raw[...,self.mobile*3:-2] + self.solve_ik(dpose)
            self.action_tensor[:, -2:-1] = self.dof_qpos_raw[...,-2:-1] + raw_output[..., -1:] * self.dt / 5
            self.action_tensor[:, -1:] = self.dof_qpos_raw[...,-1:] + raw_output[..., -1:] * self.dt / 5
            self.action_tensor = tensor_clamp(self.action_tensor, self.dof_lower_limits_tensor, self.dof_upper_limits_tensor)
        elif self.driveMode == 'ik_abs':  
            pos_err = raw_output[..., :3] - self.tip_rb_tensor[:, :3]          
            target_r = raw_output[..., 3:7] / (raw_output[...,3:7].norm(dim=-1,keepdim=True) + 1e-8)
            orn_err = orientation_error(target_r, self.tip_rb_tensor[:, 3:7])
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            self.action_tensor[:, self.mobile*3:-2] = self.dof_qpos_raw[...,self.mobile*3:-2] + self.solve_ik(dpose)
            self.action_tensor[:, -2:] = (raw_output[..., -1:] * (self.dof_upper_limits_tensor[-1] - self.dof_lower_limits_tensor[-1]) / 2 + (self.dof_upper_limits_tensor[-1] + self.dof_lower_limits_tensor[-1]) / 2).unsqueeze(-1)
            self.action_tensor = tensor_clamp(self.action_tensor, self.dof_lower_limits_tensor, self.dof_upper_limits_tensor)
        elif self.driveMode == 'heuristic':
            # NOTE based on heuristic and don't use the raw_output. Only used to debug.
            grasp_pose = torch.tensor([-0.2, 0, 0.4, -0.5, 0.5, 0.5, -0.5], device=self.device).unsqueeze(0)
            pos_err = grasp_pose[:, :3] - self.tip_rb_tensor[:,:3]
            orn_err = orientation_error(grasp_pose[:, 3:].repeat(self.tip_rb_tensor.shape[0], 1), self.tip_rb_tensor[:,3:7])
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            self.action_tensor[:, self.mobile*3:-2] = self.dof_qpos_raw[...,self.mobile*3:-2]  + self.solve_ik(dpose)
            self.action_tensor[:, -2:] = self.dof_qpos_raw[...,-2:]
            if pos_err.norm(dim=-1).mean() < 0.001 and orn_err.abs().mean() < 0.001:
                print(orn_err.mean())
                print(self.dof_qpos_raw[0,:])
                exit(1)
        else:
            raise NotImplementedError
        return self.action_tensor
    
    def solve_ik(self, dpose):
        damping = 0.05
        j_eef = (self.jacobian_tensor[:, self.ltip_rb_index - 1, :, self.mobile*3:self.num_dofs-2] + self.jacobian_tensor[:, self.rtip_rb_index - 1, :, self.mobile*3:self.num_dofs-2]) / 2
        if j_eef.sum().abs() < 1e-5:
            print('Jacobian has problem!')
            exit(1)
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(-1, self.num_dofs-2-self.mobile*3)
        return u

    def update_state(self, rigid_body_tensor, dof_state_tensor):
        self.ltip_rb_tensor = rigid_body_tensor[:, self.ltip_rb_index, :]
        self.rtip_rb_tensor = rigid_body_tensor[:, self.rtip_rb_index, :]
        self.tip_rb_tensor = (self.ltip_rb_tensor + self.rtip_rb_tensor) / 2
        self.tip_pos = self.tip_rb_tensor[:, :3]
        self.tip_rot_9d = quat_to_mat(self.tip_rb_tensor[:, 3:7])
        self.gripper_length = (self.ltip_rb_tensor[:, :3]-self.rtip_rb_tensor[:, :3]).norm(dim=-1)
        raw_data = dof_state_tensor[:, :self.num_dofs, :]   
        self.dof_qpos_normalized = (2 * (raw_data[:, :, 0]-self.dof_lower_limits_tensor[:])/(self.dof_upper_limits_tensor[:] - self.dof_lower_limits_tensor[:])) - 1
        self.dof_qpos_raw = raw_data[:, :, 0]
        self.dof_qvel_raw = raw_data[:, :, 1]
        return 

# class xarm6():
#     raise NotImplementedError