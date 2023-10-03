import importlib
from typing import List

import numpy as np

from perception.structures.point_cloud import PointCloud
from perception.datasets.gapartnet_new import apply_voxelization
from perception.utils.pose_fitting import estimate_pose_from_npcs

from envs.utils.get_running_bbox import _draw_bbox_tensor, get_bbox_pt
from envs.utils.misc import _draw_line
# from scipy.spatial.transform import Rotation as R
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
     quat_mul, tensor_clamp
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert

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
        self.perception_model = self._load_perception_model(self.algo_cfg["perception_model"])

    def _load_perception_model(self, perception_model_cfg):
        class_path = perception_model_cfg["class_path"]
        ckpt_pth = perception_model_cfg["ckpt_pth"]
        device = perception_model_cfg["device"]

        module_name = ".".join(class_path.split(".")[:-1])
        class_name = class_path.split(".")[-1]

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        
        import pdb
        
        # pdb.set_trace()

        net = cls.load_from_checkpoint(ckpt_pth)

        net.cluster_proposals_start_at = 0
        net.score_net_start_at = 0
        net.npcs_net_start_at = 0
        net.freeze()
        net.eval()
        net.to(device)

        return net

    def _inference_perception_model(self, points_list: List[torch.Tensor]):
        device = self.perception_model.device

        pcs = []
        for points in points_list:
            pc = PointCloud(
                scene_id=["eval"],
                points=points,
            )
            pc = apply_voxelization(
                pc,  voxel_size=(1. / 100, 1. / 100, 1. / 100)
            )
            pc = pc.to(device=device)
            pcs.append(pc)

        with torch.no_grad():
            scene_ids, segmentations, proposals = self.perception_model(pcs)

        sem_preds = segmentations.sem_preds

        pt_xyz = proposals.pt_xyz
        batch_indices = proposals.batch_indices
        proposal_offsets = proposals.proposal_offsets
        num_points_per_proposal = proposals.num_points_per_proposal
        num_proposals = num_points_per_proposal.shape[0]
        npcs_preds = proposals.npcs_preds
        score_preds= proposals.score_preds

        # indices = torch.arange(sem_preds.shape[0], dtype=torch.int64, device=device)
        # propsoal_indices = indices[proposals.valid_mask][proposals.sorted_indices]

        bboxes = [[] for _ in range(len(points_list))]
        for i in range(num_proposals):
            offset_begin = proposal_offsets[i].item()
            offset_end = proposal_offsets[i + 1].item()

            batch_idx = batch_indices[offset_begin]
            xyz_i = pt_xyz[offset_begin:offset_end]
            npcs_i = npcs_preds[offset_begin:offset_end]

            npcs_i = npcs_i - 0.5

            bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                estimate_pose_from_npcs(xyz_i.cpu().numpy(), npcs_i.cpu().numpy())

            bboxes[batch_idx].append(bbox_xyz.tolist())

        return bboxes

    def run(self, num_eval_iterations):
        
        
        
        if self.use_gt_bbox:
            part_bbox_tensor = self.vec_env.task.part_bbox_tensor
            handle_bbox_tensor = self.vec_env.task.handle_bbox_tensor
        else:
            part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
        # print("handle_bbox_tensor", handle_bbox_tensor, handle_bbox_tensor.shape)
        part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
        part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
        hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
        hand_grip_dir = quat_axis(hand_rot, 2)
        hand_sep_dir = quat_axis(hand_rot, 1)
        hand_down_dir = quat_axis(hand_rot, 0)

        actions = torch.zeros((self.vec_env.task.env_num, self.vec_env.task.num_actions), device = self.vec_env.task.device)
        for _ in range(num_eval_iterations):
            print(_)

            # stage 1 go to pre-grasp pose
            for i in range(500):
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

                actions[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
                actions[:,3:7] = rotations
                # actions[:,3] = 1
                # actions[:,4:7] = 0
                actions[:, -2:] = (torch.norm(handle_bbox_tensor[:,0,:] - handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.04)/2

                i=2
                _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = True, env_id=i)
                # _draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
                # _draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
                for i in range((actions.shape[0])):
                    _draw_line(self.vec_env.task, self.vec_env.task.handle_bbox_tensor[i, 3].cpu(), self.vec_env.task.handle_bbox_tensor[i, 0].cpu(), clear = False, env_id=i)
                    # _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = False, env_id=i)

                next_obs, rews, dones, infos = self.vec_env.step(actions)
                
                import pdb
                pdb.set_trace()
                
                bboxes = self._inference_perception_model([next_obs.points[0][...,:6],])
            
                import pdb
                pdb.set_trace()

            # stage 2 grasping pose
            total_step  = 50
            for step in range(total_step):
                import time
                print("step", step)
                for i in range(50):
                    if self.use_gt_bbox:
                        part_bbox_tensor = self.vec_env.task.part_bbox_tensor
                        handle_bbox_tensor = self.vec_env.task.handle_bbox_tensor
                    else:
                        part_bbox_tensor, handle_bbox_tensor = get_bbox_pt(self.vec_env.task)
                    start = time.time()
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
                    pos_pre = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.6, 0, 0], device = self.vec_env.task.device)
                    pos_tar = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device)

                    actions[:,:3] = (step/total_step)*(pos_tar - pos_pre)+pos_pre
                    actions[:,3:7] = rotations
                    # actions[:,3] = 1
                    # actions[:,4:6] = 0
                    actions[:, -2:] = (torch.norm(handle_bbox_tensor[:,0,:] - handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.04)/2

                    i=2
                    _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = True, env_id=i)
                    _draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
                    _draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
                    for i in range((actions.shape[0])):
                        _draw_line(self.vec_env.task, actions[i,:3].cpu(), actions[i,:3].cpu() - torch.tensor([0.6, 0, 0]), clear = False, env_id=i)
                        _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = False, env_id=i)

                    next_obs, rews, dones, infos = self.vec_env.step(actions)

            # stage 3 grasp
            for step in range(1):
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

                    actions[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.13, 0, 0], device = self.vec_env.task.device) 
                    actions[:,3:7] = rotations
                    # actions[:,3] = 1
                    # actions[:,4:6] = 0
                    actions[:, -2:] = 0# (torch.norm(handle_bbox_tensor[:,0,:] - handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.005)/2

                    i=2
                    _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = True, env_id=i)
                    _draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
                    _draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
                    for i in range((actions.shape[0])):
                        _draw_line(self.vec_env.task, actions[i,:3].cpu(), actions[i,:3].cpu() - torch.tensor([0.6, 0, 0]), clear = False, env_id=i)
                        _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = False, env_id=i)

                    next_obs, rews, dones, infos = self.vec_env.step(actions)


            # stage 4 interaction
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

                    actions[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.33, 0, 0], device = self.vec_env.task.device)
                    actions[:,3:7] = rotations
                    actions[:, -2:] = 0# (torch.norm(handle_bbox_tensor[:,0,:] - handle_bbox_tensor[:,3,:], dim  = -1, keepdim=True) + 0.005)/2

                    i=2
                    _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = True, env_id=i)
                    _draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
                    _draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
                    for i in range((actions.shape[0])):
                        _draw_line(self.vec_env.task, actions[i,:3].cpu(), actions[i,:3].cpu() - torch.tensor([0.6, 0, 0]), clear = False, env_id=i)
                        _draw_line(self.vec_env.task, actions[i,:3].cpu(), self.vec_env.task.hand_rigid_body_tensor[i, :3].cpu(), clear = False, env_id=i)

                    next_obs, rews, dones, infos = self.vec_env.step(actions)
            else:
                pass
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
            #         raise NotImplementedError
            #     part_pos = self.vec_env.task.cabinet_part_rigid_body_tensor[:, :3]
            #     part_rot = self.vec_env.task.cabinet_part_rigid_body_tensor[:, 3:7]
            #     hand_rot = self.vec_env.task.hand_rigid_body_tensor[..., 3:7]
            #     hand_grip_dir = quat_axis(hand_rot, 2)
            #     hand_sep_dir = quat_axis(hand_rot, 1)
            #     hand_down_dir = quat_axis(hand_rot, 0)
            #     _draw_line(self.vec_env.task, self.vec_env.task.hand_rigid_body_tensor[0, 3:7].cpu(), self.vec_env.task.hand_rigid_body_tensor[0, 3:7].cpu() + hand_grip_dir[0].cpu(), clear = True,)
            #     next_obs, rews, dones, infos = self.vec_env.step(actions)
            # stage 3 interaction
