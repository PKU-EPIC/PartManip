import numpy as np

from envs.utils.get_running_bbox import _draw_bbox_tensor, get_bbox_pt
from envs.utils.misc import _draw_line
# from scipy.spatial.transform import Rotation as R
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
     quat_mul, tensor_clamp
import torch    
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert, quaternion_to_matrix, quaternion_apply, quaternion_raw_multiply, quaternion_multiply, euler_angles_to_matrix

def quat_axis(q, axis=0):
    '''
    :func apply rotation represented by quanternion `q`
    on basis vector(along axis)
    :return vector after rotation
    '''
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

class HEURISTICS:
    def __init__(
        self,
        vec_env,
        algo_cfg,

    ):
        self.vec_env = vec_env
        self.stage = torch.zeros(self.vec_env.task.env_num)
        self.algo_cfg = algo_cfg
        self.use_gt_bbox = self.algo_cfg["useGtBbox"]

    def run(self, num_eval_iterations):
        self.vec_env.task.step_heu = 0
        if self.use_gt_bbox:
            part_bbox_tensor = self.vec_env.task.part_bbox_tensor
            handle_bbox_tensor = self.vec_env.task.handle_bbox_tensor
            # print(handle_bbox_tensor.shape)
            # print(part_bbox_tensor.shape)
            # exit(123)
        else:
            part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
        part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
        part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
        hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
        hand_grip_dir = quat_axis(hand_rot, 2)
        hand_sep_dir = quat_axis(hand_rot, 1)
        hand_down_dir = quat_axis(hand_rot, 0)
    
        
        actions = torch.zeros((self.vec_env.task.env_num, self.vec_env.task.num_actions), device = self.vec_env.task.device)
        for _ in range(num_eval_iterations):

            # stage 1 go to pre-grasp pose
            print("stage 1 go to pre-grasp pose")
            hand_rot_ = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
            for i in range(500):    
                if self.use_gt_bbox:
                    part_bbox_tensor = self.vec_env.task.part_bbox_tensor
                    handle_bbox_tensor = self.vec_env.task.handle_bbox_tensor
                else:
                    part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
                part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
                part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
                hand_xyz = self.vec_env.task.hand_rigid_body_tensor[..., :3]
                hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
                handle_out = self.vec_env.task.handle_bbox_tensor[:, 0] - self.vec_env.task.handle_bbox_tensor[:, 4]
                handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
                handle_long = self.vec_env.task.handle_bbox_tensor[:, 1] - self.vec_env.task.handle_bbox_tensor[:, 0]
                handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
                handle_short = self.vec_env.task.handle_bbox_tensor[:, 3] - self.vec_env.task.handle_bbox_tensor[:, 0]
                handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)

                # rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
                # print(torch.cat((-handle_long.reshape((-1,3,1)), handle_short.reshape((-1,3,1)), handle_out.reshape((-1,3,1))), dim = 2))
                # exit(123)
                rotations = (matrix_to_quaternion(torch.cat((-handle_long.reshape((-1,3,1)), handle_short.reshape((-1,3,1)), handle_out.reshape((-1,3,1))), dim = 2)))
                hand_r = quaternion_to_matrix(hand_rot)

                actions[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
                actions[:,3:7] = quaternion_multiply(rotations, torch.tensor([0,0,1,0], device = self.vec_env.task.device))
                actions[:, -2:] = (torch.norm(handle_bbox_tensor[:,0,:] - handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.04)/2

                i=0
                # _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = True, env_id=i)
                # _draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
                # _draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
                # for i in range((actions.shape[0])):
                #     _draw_line(self.vec_env.task, self.vec_env.task.handle_bbox_tensor[i, 4].cpu(), self.vec_env.task.handle_bbox_tensor[i, 0].cpu(), clear = False, env_id=i)
                #     _draw_line(self.vec_env.task, self.vec_env.task.handle_bbox_tensor[i, 1].cpu(), self.vec_env.task.handle_bbox_tensor[i, 0].cpu(), clear = False, env_id=i)
                #     _draw_line(self.vec_env.task, hand_xyz[i].cpu(),(hand_xyz[i]+0.5*hand_r[i, 0]).cpu(), clear = False, env_id=i)
                #     _draw_line(self.vec_env.task, hand_xyz[i].cpu(),(hand_xyz[i]+1*hand_r[i, 1]).cpu(), clear = False, env_id=i)
                #     _draw_line(self.vec_env.task, hand_xyz[i].cpu(),(hand_xyz[i]+2*hand_r[i, 2]).cpu(), clear = False, env_id=i)
                #     _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = False, env_id=i)

                self.vec_env.control_step(actions)
                
            # stage 2 grasping pose
            print("stage 2 grasping pose")
            total_step  = 10
            for step in range(total_step):
            
                print("step", step)
                for i in range(50):
                    if self.use_gt_bbox:
                        part_bbox_tensor = self.vec_env.task.part_bbox_tensor
                        handle_bbox_tensor = self.vec_env.task.handle_bbox_tensor
                    else:
                        part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
                    part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
                    part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
                    hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
                    hand_xyz = self.vec_env.task.hand_rigid_body_tensor[..., :3]
                    handle_out = self.vec_env.task.handle_bbox_tensor[:, 0] - self.vec_env.task.handle_bbox_tensor[:, 4]
                    handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
                    handle_long = self.vec_env.task.handle_bbox_tensor[:, 1] - self.vec_env.task.handle_bbox_tensor[:, 0]
                    handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
                    handle_short = self.vec_env.task.handle_bbox_tensor[:, 3] - self.vec_env.task.handle_bbox_tensor[:, 0]
                    handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)

                    rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
                    pos_pre = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
                    pos_tar = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)

                    actions[:,:3] = (step/total_step)*(pos_tar - pos_pre)+pos_pre
                    actions[:,3:7] = rotations
                    # actions[:,3] = 1
                    # actions[:,4:6] = 0
                    actions[:, -2:] = (torch.norm(handle_bbox_tensor[:,0,:] - handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.04)/2
                    hand_r = quaternion_to_matrix(hand_rot)
                    i=0

                    # _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = True, env_id=i)
                    #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
                    #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
                    # for i in range((actions.shape[0])):
                        # _draw_line(self.vec_env.task, actions[i,:3].cpu(), actions[i,:3].cpu() - torch.tensor([0.6, 0, 0]), clear = False, env_id=i)
                        # _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = False, env_id=i)
                        # _draw_line(self.vec_env.task, hand_xyz[i].cpu(),(hand_xyz[i]+0.5*hand_r[i, 0]).cpu(), clear = False, env_id=i)
                        # _draw_line(self.vec_env.task, hand_xyz[i].cpu(),(hand_xyz[i]+1*hand_r[i, 1]).cpu(), clear = False, env_id=i)
                        # _draw_line(self.vec_env.task, hand_xyz[i].cpu(),(hand_xyz[i]+2*hand_r[i, 2]).cpu(), clear = False, env_id=i)
                    self.vec_env.control_step(actions)

            # stage 3 grasp
            print("stage 3 grasp")
            for step in range(1):
                for i in range(100):
                    if self.use_gt_bbox:
                        part_bbox_tensor = self.vec_env.task.part_bbox_tensor
                        handle_bbox_tensor = self.vec_env.task.handle_bbox_tensor
                    else:
                        part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
                    part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
                    part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
                    hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
                    handle_out = self.vec_env.task.handle_bbox_tensor[:, 0] - self.vec_env.task.handle_bbox_tensor[:, 4]
                    handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
                    handle_long = self.vec_env.task.handle_bbox_tensor[:, 1] - self.vec_env.task.handle_bbox_tensor[:, 0]
                    handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
                    handle_short = self.vec_env.task.handle_bbox_tensor[:, 3] - self.vec_env.task.handle_bbox_tensor[:, 0]
                    handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)

                    rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))

                    actions[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device) 
                    actions[:,3:7] = rotations
                    # actions[:,3] = 1
                    # actions[:,4:6] = 0
                    actions[:, -2:] = -0.1# (torch.norm(handle_bbox_tensor[:,0,:] - handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.005)/2

                    i=0
                    # _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = True, env_id=i)
                    #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
                    #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
                    # for i in range((actions.shape[0])):
                        # _draw_line(self.vec_env.task, actions[i,:3].cpu(), actions[i,:3].cpu() - torch.tensor([0.6, 0, 0]), clear = False, env_id=i)
                        # _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = False, env_id=i)

                    self.vec_env.control_step(actions)


            # stage 4 interaction
            print("stage 4 interaction with part")
            if self.vec_env.task.target_part == "drawer":
                for i in range(1000):
                    if self.use_gt_bbox:
                        part_bbox_tensor = self.vec_env.task.part_bbox_tensor
                        handle_bbox_tensor = self.vec_env.task.handle_bbox_tensor
                    else:
                        part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
                    part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
                    part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
                    hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
                    handle_out = self.vec_env.task.handle_bbox_tensor[:, 0] - self.vec_env.task.handle_bbox_tensor[:, 4]
                    handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
                    handle_long = self.vec_env.task.handle_bbox_tensor[:, 1] - self.vec_env.task.handle_bbox_tensor[:, 0]
                    handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
                    handle_short = self.vec_env.task.handle_bbox_tensor[:, 3] - self.vec_env.task.handle_bbox_tensor[:, 0]
                    handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)

                    rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))

                    actions[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.5, 0, 0], device = self.vec_env.task.device) #- self.vec_env.task.hand_rigid_body_tensor[:, :3]
                    actions[:,3:7] = rotations
                    actions[:, -2:] = 0# (torch.norm(handle_bbox_tensor[:,0,:] - handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.005)/2

                    i=0
                    # _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = True, env_id=i)
                    #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
                    #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
                    # for i in range((actions.shape[0])):
                        # _draw_line(self.vec_env.task, actions[i,:3].cpu(), actions[i,:3].cpu() - torch.tensor([0.6, 0, 0]), clear = False, env_id=i)
                        # _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = False, env_id=i)

                    self.vec_env.control_step(actions)
                # import pdb
                # pdb.set_trace()
            else:
                if self.use_gt_bbox:
                    part_bbox_tensor = self.vec_env.task.part_bbox_tensor
                    handle_bbox_tensor = self.vec_env.task.handle_bbox_tensor
                    part_axis_xyz_tensor = self.vec_env.task.part_axis_xyz_tensor
                    part_axis_dir_tensor = self.vec_env.task.part_axis_dir_tensor

                else:
                    part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
                    part_axis_xyz_tensor_gt = self.vec_env.task.part_axis_xyz_tensor
                    part_axis_dir_tensor_gt = self.vec_env.task.part_axis_dir_tensor
                    part_axis_xyz_tensor = part_bbox_tensor[:, [2,3,6,7], :].mean(1)
                    part_axis_dir_tensor = part_bbox_tensor[:, [3,7], :].mean(1) - part_bbox_tensor[:, [2,6], :].mean(1)
                    mask = ((part_axis_dir_tensor * part_axis_dir_tensor_gt).sum(-1))<0
                    part_axis_dir_tensor[mask] = - part_axis_dir_tensor[mask]
                    part_axis_dir_tensor = part_axis_dir_tensor / torch.norm(part_axis_dir_tensor, dim = -1,keepdim=True)
                part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
                part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
                hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
                handle_out = self.vec_env.task.handle_bbox_tensor[:, 0] - self.vec_env.task.handle_bbox_tensor[:, 4]
                handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
                handle_long = self.vec_env.task.handle_bbox_tensor[:, 1] - self.vec_env.task.handle_bbox_tensor[:, 0]
                handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
                handle_short = self.vec_env.task.handle_bbox_tensor[:, 3] - self.vec_env.task.handle_bbox_tensor[:, 0]
                handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
                total_step = 50
                for global_step in range(total_step):
                    for local_step in range(30):
                        print(local_step, global_step)
                        
                        import pdb
                        # pdb.set_trace()
                        rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))

                        initial_target = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)
                        relative_target = initial_target - part_axis_xyz_tensor
                        angle = (3.14159 / (3* total_step)) * global_step
                        # pdb.set_trace()
                        rotation_ = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
                        # rotation_angle = euler_angles_to_matrix(angle*part_axis_dir_tensor, "XYZ")
                        quat_ = matrix_to_quaternion(rotation_)
                        rotated_relative_target = torch.bmm(rotation_, relative_target.reshape(-1,3,1), ).reshape(-1,3)
                        rotated_target = rotated_relative_target + part_axis_xyz_tensor
                        actions[:,:3] = rotated_target
                        actions[:,3:7] = quaternion_multiply(quat_, rotations)
                        actions[:, -2:] = 0# (torch.norm(handle_bbox_tensor[:,0,:] - handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.005)/2

                        i=0
                        # _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = True, env_id=i)
                        # _draw_line(self.vec_env.task, part_axis_xyz_tensor[i].cpu(), part_axis_xyz_tensor[i].cpu() + torch.tensor([0,0,3]), clear = False, env_id=i)
                        #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
                        #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
                        # for i in range((actions.shape[0])):
                        #     _draw_line(self.vec_env.task, actions[i,:3].cpu(), actions[i,:3].cpu() - torch.tensor([0.6, 0, 0]), clear = False, env_id=i)
                        #     _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = False, env_id=i)

                        self.vec_env.control_step(actions)
                # pdb.set_trace()
                print(self.vec_env.task.cabinet_dof_tensor[:,0]) 
                print((self.vec_env.task.cabinet_dof_tensor[:50,0] > 0.3).sum()/(self.vec_env.task.cabinet_dof_tensor[:50,0].shape[0]))
                # TODO: for more smooth:
                # Quaternion interpolation using SLERP and SQUAD provide a way to interpolate smoothly between orientations in space.

            # stage 4: interaction











            # actions = torch.zeros((self.vec_env.task.env_num, self.vec_env.task.num_actions), device = self.vec_env.task.device)
            # # actions[:,2] = 1
            # actions[:,:3] = (handle_bbox_tensor[:,2,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.5, 0, 0], device = self.vec_env.task.device)
            # print(actions[:,:3])
            # actions[:,3] = 1
            # for i in range(1000):    
            #     if self.use_gt_bbox:
            #         part_bbox_tensor = self.vec_env.task.part_bbox_tensor
            #         handle_bbox_tensor = self.vec_env.task.handle_bbox_tensor
            #         # print(handle_bbox_tensor.shape)
            #         # print(part_bbox_tensor.shape)
            #         # exit(123)
            #     else:
            #         part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
            #     part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
            #     part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
            #     hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
            #     hand_grip_dir = quat_axis(hand_rot, 2)
            #     hand_sep_dir = quat_axis(hand_rot, 1)
            #     hand_down_dir = quat_axis(hand_rot, 0)
            #     _draw_line(self.vec_env.task, self.vec_env.task.hand_rigid_body_tensor[0, 3:7].cpu(), self.vec_env.task.hand_rigid_body_tensor[0, 3:7].cpu() + hand_grip_dir[0].cpu(), clear = True,)
            #     next_obs, rews, dones, infos = self.vec_env.step(actions)
            # stage 3 interaction



        return 


if __name__ == "__main__":
    # from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp 
    rs = R.from_quat(torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,0]]))
    rs = R.from_quat(torch.tensor([[1,0,0,0],[0,1,0,0]]))
    v = [1,0,0]
    print(rs.apply(v))
    key_times = [0,1,]
    slerp = Slerp(key_times,rs)
    times = [0, 0.1, 0.5, 0.7, 1]
    rots = slerp(times)

    vec1 = [[1,0,0]]
    vec2 = [[0,1,0]]
    print(R.align_vectors(vec1,vec2)[0].apply(v))
    # print(rots.apply(v))
