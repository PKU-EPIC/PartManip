from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import numpy as np
from.franka_pose_cabinet_base import FrankaPoseCabinetBase
from .utils.load_env import _load_cam
from pointnet2_ops import pointnet2_utils
import open3d as o3d
from utils.gym_info import COLOR20
import cv2
import os
from .utils.process_pointcloud import compute_point_cloud_state, sample_points, PointCloud_format
from typing import List
import importlib
from perception.structures.point_cloud import PointCloud
# from perception.datasets.gapartnet_new import apply_voxelization
from perception.utils.pose_fitting import estimate_pose_from_npcs
from perception.structures.point_cloud import PointCloud
from perception.structures.instances import Instances
from .utils.get_observation import _get_pc_observation, _refresh_pc_observation, _refresh_observation
import torch
from .utils.misc import _draw_line
from .utils.get_running_pc import get_pc_isaac_tensor
# from .utils.get_running_bbox import _draw_bbox
# from .utils.reset import _partial_reset, reset
# from .utils.get_observation import _refresh_observation
from .utils.get_reward import get_reward_done, get_reward_done_all
from .utils.load_env import _load_obj_asset, _load_franka, _load_obj, _create_ground_plane, _place_agents
from .utils.compute import *
from .utils.get_observation import _get_base_observation,_refresh_observation
from .utils.perform_action import _perform_actions
from .utils.misc import _draw_line, save, draw_bbox
from .utils.reset import _partial_reset, reset
from .utils.step import step, control_step
from .utils.get_running_bbox import get_bbox_for_now, get_bbox_from_world_to_isaac, _draw_bbox
from utils.gym_info import *
# import torch
from perception.structures.point_cloud import PointCloud
from perception.datasets.gapartnet_new import apply_voxelization
from perception.utils.pose_fitting import estimate_pose_from_npcs

class FrankaPoseCabinetPC(FrankaPoseCabinetBase):
    def __init__(
        self, 
        cfg, 
        sim_params, 
        physics_engine, 
        device_type, 
        device_id, 
        headless, 
        log_dir = None
    ):
        if cfg["save_video"]:
            self.video_save_root = cfg["save_video_root"]
            if not os.path.exists(self.video_save_root):
                os.mkdir(self.video_save_root)
            root_category = cfg["save_video_root"] + "/{}".format(cfg["task"]["target_part_category"])
            if not os.path.exists(root_category):
                os.mkdir(root_category)

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless, log_dir)
        if not self.cfg["obs"]["pc"]["use_pseudo_pc"]:
            for i_, env_ptr in enumerate(self.env_ptr_list):
                self.gym.set_rigid_body_segmentation_id(env_ptr, self.franka_actor, self.hand_rigid_body_index, 3)
                self.gym.set_rigid_body_segmentation_id(env_ptr, self.franka_actor, self.hand_lfinger_rigid_body_index, 3)
                self.gym.set_rigid_body_segmentation_id(env_ptr, self.franka_actor, self.hand_rfinger_rigid_body_index, 3)
                self.gym.set_rigid_body_segmentation_id(env_ptr, self.cabinet_actor_list[i_], self.env_base_rigid_id_list[i_], 0) # base
                self.gym.set_rigid_body_segmentation_id(env_ptr, self.cabinet_actor_list[i_], self.env_part_rigid_id_list[i_], 1) # part
                if self.target_part in ["door", "drawer", "handle"]:
                    self.gym.set_rigid_body_segmentation_id(env_ptr, self.cabinet_actor_list[i_], self.env_handle_rigid_id_list[i_], 2) # handle

        print("Finish super initialization~")
        self.pc_downsample_num = self.cfg["obs"]["pc"]["pcDownSampleNum"]
        self.pc_feature_dim = cfg["obs"]["pc"]["pointFeatureDim"]
        self.pc_obs_num = cfg["obs"]["pc"]["numPCObservations"]
        self.obs_buf = None
        self.task_meta["mask_dim"] = self.pc_obs_num - 3 # TODO 
        self.pre_random_sample_num = cfg["obs"]["pc"]["pre_random_sample_num"]
        self.z_p_bar = cfg["obs"]["pc"]["z_p_bar"]
        self.z_n_bar = cfg["obs"]["pc"]["z_n_bar"]
        self.x_p_bar = cfg["obs"]["pc"]["x_p_bar"]
        self.x_n_bar = cfg["obs"]["pc"]["x_n_bar"]
        self.y_p_bar = cfg["obs"]["pc"]["y_p_bar"]
        self.y_n_bar = cfg["obs"]["pc"]["y_n_bar"]
        self.use_pc = cfg["obs"]["use_pc"]

        self.pc_feats = None

        if cfg["obs"]["pc"]["visualizePointcloud"] == True :
            import open3d as o3d
            from .utils.misc import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else :
            self.pointCloudVisualizer = None 
        
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

        # if cfg["save_video"]:
        #     self.video_save_root = "/data2/ziming/video"
        #     if not os.path.exists(self.video_save_root):
        #         os.mkdir(self.video_save_root)
            # self.video_save_path = self.video_save_root + "/" + "{}.mp4".format(self.cfg["task"]["target_part_category"])
            # video_file = os.path.join(self.video_save_path)
            # self.video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 25,
            #                                     (self.image_shape[1], self.image_shape[0]))

        # import pdb
        # pdb.set_trace()

    def _place_agents(self, env_num, spacing, use_cam = True):
        if not self.cfg["obs"]["pc"]["use_pseudo_pc"]:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
            self.camera_props.enable_tensors = True
            self.cameras = []
            self.camera_depth_tensor_list = []
            self.camera_rgb_tensor_list = []
            self.camera_seg_tensor_list = []
            self.camera_view_matrix_inv_list = []
            self.camera_proj_matrix_list = []

            self.camera_u = torch.arange(0, 
                self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, 
                self.camera_props.height, device=self.device)

            self.camera_v2, self.camera_u2 = torch.meshgrid(
                    self.camera_v, self.camera_u, indexing='ij')

            self.env_origin = torch.zeros((self.num_envs, 3), 
                device=self.device, dtype=torch.float)
            
        if self.cfg["save_video"]:
            self.image_shape = [1080, 1920]
            self.video_camera_props = gymapi.CameraProperties()
            self.video_camera_props.width = self.image_shape[1]
            self.video_camera_props.height = self.image_shape[0]
            self.video_camera_props.enable_tensors = True
            self.video_cameras = []
            self.video_camera_depth_tensor_list = []
            self.video_camera_rgb_tensor_list = []
            self.video_camera_seg_tensor_list = []
            self.video_camera_view_matrix_inv_list = []
            self.video_camera_proj_matrix_list = []
            self.video_file_list = []
            self.video_writer_list = []

            self.video_camera_u = torch.arange(0, 
                self.video_camera_props.width, device=self.device)
            self.video_camera_v = torch.arange(0, 
                self.video_camera_props.height, device=self.device)

            self.video_camera_v2, self.video_camera_u2 = torch.meshgrid(
                    self.video_camera_v, self.video_camera_u, indexing='ij')

            self.env_origin = torch.zeros((self.num_envs, 3), 
                device=self.device, dtype=torch.float)

            return super()._place_agents(env_num=env_num, spacing=spacing, use_cam=True)
        
        return super()._place_agents(env_num=env_num, spacing=spacing, use_cam=use_cam)
        
        # return super()._place_agents(env_num, spacing, False)
    
    def _load_cam(self, env_ptr, env_id):
        
        if self.cfg["obs"]["pc"]["use_pseudo_pc"] or not self.cfg["env"]["enableCameraSensors"]:
            return None
        return _load_cam(self, env_ptr, env_id)
    
    def compute_point_cloud_state(self, depth_bar, asymm_obs=False, add_mask=None, return_img = False):
        # import pdb
        # pdb.set_trace()
        if self.cfg["obs"]["pc"]["use_pseudo_pc"]:
            if self.target_part in ["door", "handle"]:
                pc_new = get_pc_isaac_tensor(self, self.cabinet_dof_tensor[:,0], 0)
                # pc_new = get_pc_isaac_tensor(self, torch.tensor([0.2],device = self.device), 0)
            elif self.target_part == "drawer":
                pc_new = get_pc_isaac_tensor(self, self.cabinet_dof_tensor[:,0], 1)
            pc = []
            for i in range(self.num_envs):
                import torch
                pointcloud_fps = torch.cat((self.pseudo_pc_init[i][self.pseudo_pc_base_mask[i]], pc_new[i][self.pseudo_pc_part_mask[i]], pc_new[i][self.pseudo_pc_handle_mask[i]]), dim = 0)
                seg = pointcloud_fps[:,0].clone()
                seg[...] = 0.
                seg[self.pseudo_pc_base_mask[i]] = 1
                seg[self.pseudo_pc_part_mask[i]] = 2
                seg[self.pseudo_pc_handle_mask[i]] = 3
                pointcloud_fps = torch.cat((pointcloud_fps, seg.reshape(-1,1)), dim = 1)
                pc.append(pointcloud_fps)
            pc = torch.stack(pc)
            return pc
        if return_img:
            pointcloud_fps, imgs = compute_point_cloud_state(self, depth_bar, asymm_obs, add_mask, return_img=True)
            return pointcloud_fps, imgs
        pointcloud_fps = compute_point_cloud_state(self, depth_bar, asymm_obs, add_mask)
        if self.cfg["gen_pc"]:
            import torch
            path = str(self.selected_asset_path_list[0]).split("/")[-1]
            torch.save(pointcloud_fps,f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/pc_20000/{path}_pc_data.pth")
            # torch.save(self.selected_asset_path_list,f"/scratch/genghaoran/RL-Pose/PoseOrientedGym/assets/pc/{}_pc_path.pth")
            exit(123)
        if self.cfg["task"]["use_fixed_feature"]:
            PointCloudList, mask_list = PointCloud_format(pointcloud_fps)
            global_features =self._inference_perception_model(PointCloudList, None)
            return pointcloud_fps, global_features
        return pointcloud_fps
        
        
        # for i in range(self.num_envs):
            # img_0 = self.camera_rgb_tensor_list[i][0].cpu().numpy()
            # img_1 = self.camera_rgb_tensor_list[i][1].cpu().numpy()
            # img_2 = self.camera_rgb_tensor_list[i][2].cpu().numpy()
            # img_rgb_0_bgr = cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR)
            # img_rgb_1_bgr = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
            # img_rgb_2_bgr = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)
            # img_seg_0 = COLOR20[self.camera_seg_tensor_list[i][0].long().cpu().numpy()]
            # img_seg_1 = COLOR20[self.camera_seg_tensor_list[i][1].long().cpu().numpy()]
            # img_seg_2 = COLOR20[self.camera_seg_tensor_list[i][2].long().cpu().numpy()]
            # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/seg_{i}_0.png", img_seg_0)
            # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/seg_{i}_1.png", img_seg_1)
            # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/seg_{i}_2.png", img_seg_2)
            # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/rgb_{i}_0.png", img_rgb_0_bgr)
            # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/rgb_{i}_1.png", img_rgb_1_bgr)
            # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/rgb_{i}_2.png", img_rgb_2_bgr)
        # exit(123)
        import pdb
        
        # pdb.set_trace()
        
        self.perception_model = self._load_perception_model(self.cfg["inference"]["perception_model"])
        # pdb.set_trace()
        bboxes = self._inference_perception_model([pointcloud_fps[0][...,:6],])
        # pdb.set_trace()

        
        # img_0 = self.camera_rgb_tensor_list[0][0].cpu().numpy()
        # img_1 = self.camera_rgb_tensor_list[0][1].cpu().numpy()
        # img_2 = self.camera_rgb_tensor_list[0][2].cpu().numpy()
        # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/bbox_debug/rgb_bbox_{0}_0.png", img_seg_0)
        # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/bbox_debug/rgb_bbox_{0}_2.png", img_seg_2)
        # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/bbox_debug/rgb_bbox_{0}_1.png", img_seg_1)
        
        
        import open3d as o3d
        # pdb.set_trace()
        pcd = o3d.geometry.PointCloud()
        pcd_bbox = o3d.geometry.PointCloud()
        pcd_bbox.points = o3d.utility.Vector3dVector(np.array(bboxes[0][0])[:, :3])
        bbox_color = np.array(bboxes[0][0])[:, :3]* 0.0
        bbox_color[:,0] = 1.0
        pcd_bbox.colors = o3d.utility.Vector3dVector(bbox_color)
        
        pcd.points = o3d.utility.Vector3dVector(pointcloud_fps[0, :, :3].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(pointcloud_fps[0, :, 3:6].cpu().numpy()/255.0)
        self.pointCloudVisualizer.add_geometry(pcd)
        
        self.pointCloudVisualizer.add_geometry(pcd_bbox)
        while True :
            self.pointCloudVisualizer.update(pcd)
        
        return pointcloud_fps #, global_features

    def sample_points(self, points, sample_num=1000, sample_method='random'):
        return sample_points(self, points, sample_num, sample_method)

    def _get_base_observation(self, suggested_gt=None):
        return _get_pc_observation(self, suggested_gt, self.use_pc)

    def _refresh_observation(self, suggested_gt=None) :
        return _refresh_pc_observation(self, suggested_gt, self.use_pc)
    
    def reset(self, to_reset = "all") :
        return reset(self, to_reset)

    def _get_reward_done(self):
        return get_reward_done(self)
      
    def _get_base_observation(self, suggested_gt=None, pregrasp=False) :
        return _get_base_observation(self, suggested_gt)


    # def _perform_actions(self, actions):
    #     return _perform_actions(self, actions)

    # def step(self, actions) :
    #     return step(self, actions)

    # def control_step(self, actions):
    #     return control_step(self, actions)

    def _partial_reset(self, to_reset = "all") :
        return _partial_reset(self, to_reset)
   
    def reset(self, to_reset = "all") :
        return reset(self, to_reset)
    
    def save_video(self, last_frame = False):
        self.gym.render_all_camera_sensors(self.sim)
        # t = time.time()
        # print("cam pointcloud", t - s)

        self.gym.start_access_image_tensors(self.sim)
        camera_rgb_tensor = torch.stack([torch.stack(i) for i in self.video_camera_rgb_tensor_list])[:, :, :, :, :3].reshape(-1, self.image_shape[0], self.image_shape[1], 3)
        self.gym.end_access_image_tensors(self.sim)
        image = np.uint8(camera_rgb_tensor.cpu())
        for i in range(image.shape[0]):
            video_writer = self.video_writer_list[i]
            image[i] = cv2.cvtColor(image[i], cv2.COLOR_RGB2BGR)
            video_writer.write(image[i])
            if last_frame:
                video_writer.release()
        # image = cv2.cvtColor(camera_rgb_tensor, cv2.COLOR_RGB2BGR)
        # self.video_writer.write(image)
        
        if last_frame:
            # self.video_writer.release()
            cv2.destroyAllWindows()
            exit()
        pass

    # def save(self, path, iteration) :
    #     return save(self, path, iteration)

    # def load(self, path, iteration) :
    #     pass
    
    # def update(self, it=0) :
    #     pass
    
    # def train(self) :        # changing mode to eval
    #     self.train_mode = True

    # def eval(self) :        # changing mode to eval
    #     self.train_mode = False

    # def intervaledRandom_(self, tensor, dist, lower=None, upper=None) :
    #     tensor += torch.rand(tensor.shape, device=self.device)*dist*2 - dist
    #     if lower is not None and upper is not None :
    #         torch.clamp_(tensor, min=lower, max=upper)
 

