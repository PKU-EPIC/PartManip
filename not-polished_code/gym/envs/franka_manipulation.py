
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
    quat_mul, tensor_clamp
import numpy as np
from requests import head
import torch
from envs.base.base_env import BaseEnv
from tqdm import tqdm
import os,sys
from os.path import join as pjoin
from random import shuffle
import json
import yaml
from isaacgym.gymtorch import wrap_tensor
from utils.gym_info import *
from rich.progress import track
from glob import glob
from pathlib import Path
from .utils.get_reward import get_reward_done, get_reward_done_all
from .utils.load_env import _load_obj_asset, _load_franka, _load_obj, _create_ground_plane, _place_agents
from .utils.compute import *
from .utils.get_observation import _get_base_observation,_refresh_observation
from .utils.perform_action import _perform_actions
from .utils.misc import _draw_line, save
from .utils.reset import _partial_reset, reset
from .utils.step import step, control_step
from .utils.get_running_bbox import get_bbox_for_now, get_bbox_from_world_to_isaac, _draw_bbox, \
    get_bbox_isaac, get_bbox_for_now_tensor, get_bbox_from_world_to_isaac_tensor, \
    get_bbox_isaac_tensor, _draw_bbox_tensor, get_bbox_pt, get_bbox_isaac_tensor_nohandle, get_bbox_pt_nohandle
from .utils.perform_action import _get_qpos_from_hand_pose
from pytorch3d.transforms import quaternion_invert, matrix_to_quaternion
from isaacgym.gymtorch import wrap_tensor
# for any representation
# only focus on loadind object, franka and acquire their information

class FrankaPoseCabinetBase(BaseEnv):
    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        device_type,
        device_id,
        headless, 
        log_dir = None,
    ):
        # init cfg, sim, phy, device

        print(cfg)
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.log_dir = log_dir
        self.up_axis = "z"
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.target_part = cfg["task"]["target_part_category"]


        # asset num
        self.cabinet_num_train = cfg["env"]["AssetNumTrain"]
        self.cabinet_num_valIntra = cfg["env"]["AssetNumValIntra"]
        self.cabinet_num_valInter = cfg["env"]["AssetNumValInter"]
        self.env_per_asset = cfg["env"]["EnvPerAsset"]
        self.asset_root = cfg["env"]["asset"]["assetRoot"]
        
        # asset info
        self.cabinet_assets_path = pjoin(self.asset_root, self.target_part)
        self.train_asset_root = Path(pjoin(self.cabinet_assets_path, "train"))
        self.valIntra_asset_root = Path(pjoin(self.cabinet_assets_path, "valIntra"))
        self.valInter_asset_root = Path(pjoin(self.cabinet_assets_path, "valInter"))
        self.total_train_asset_paths = list(self.train_asset_root.iterdir())
        self.total_valIntra_asset_paths = list(self.valIntra_asset_root.iterdir())
        self.total_valInter_asset_paths = list(self.valInter_asset_root.iterdir())

        path_all = ''
        new_path = self.total_train_asset_paths
        if self.target_part == "drawer":
            # import pdb
            # pdb.set_trace()
            path_to_remove = [
                Path("assets/drawer/train/StorageFurniture-48855-link_0-handle_0-joint_0-handlejoint_0"),
            ]
            asset_to_remove = ["47207", "46537", "19855", "30666"]
            for p in path_to_remove:
                if p in new_path:
                    new_path.remove(p)
                    continue
            
            for a in asset_to_remove:
                for p in new_path:
                    if a in str(p):
                        new_path.remove(p)
            for a in asset_to_remove:
                for p in self.total_valIntra_asset_paths:
                    if a in str(p):
                        self.total_valIntra_asset_paths.remove(p)
            for a in asset_to_remove:
                for p in self.total_valInter_asset_paths:
                    if a in str(p):
                        self.total_valInter_asset_paths.remove(p)
            # print(self.total_valInter_asset_paths[14])
            ###for debug###     
            #new_path = [Path("/data2/ziming/RL-Pose/PoseOrientedGym/assets/drawer/train/Table-30666-link_5-handle_5-joint_5-handlejoint_5")]
            # new_path = [
            #         Path("assets/drawer/train/Dishwasher-12085-link_1-handle_0-joint_1-handlejoint_0"),
            #         Path("assets/drawer/train/StorageFurniture-47235-link_0-handle_0-joint_0-handlejoint_0"),
            #         Path("assets/drawer/train/StorageFurniture-48169-link_0-handle_0-joint_0-handlejoint_0")]
            # self.total_train_asset_paths = new_path
            # print(new_path[48])
            # exit()
            #self.total_valInter_asset_paths = [Path("assets/drawer/valInter/Table-27189-link_0-handle_0-joint_0-handlejoint_0")]
            
            self.total_train_asset_paths = new_path
        elif self.target_part in ["door"]:
            path_to_remove = [
                Path("assets/door/train/StorageFurniture-46037-link_0-handle_0-joint_0-handlejoint_0"),
                Path("assets/door/train/StorageFurniture-41083-link_1-handle_5-joint_1-handlejoint_5"),
            ]
            for p in path_to_remove:
                if p in new_path:
                    new_path.remove(p)
            # door_no_demo = [0, 2, 4, 7, 9, 10, 11, 12, 13, 15, 21, 22, 24, 26, 27, 29, 31, 33, 35, 36, 41, 42, 43, 44, 46, 48, 49, 52, 53, 56, 57, 60, 65, 67, 68, 71, 75, 78, 79, 80, 81, 82, 84, 86, 87, 90, 98, 99, 100, 101, 103, 108, 110, 111, 113, 114, 117, 119, 120, 122, 123, 126, 128, 129, 131, 132, 134, 136, 137, 139, 142, 145, 146, 148, 149, 151, 153, 157, 161, 163, 165, 167, 168, 172, 173, 174, 176, 177, 180, 182, 186, 187, 191, 192, 195, 198, 199, 200, 201, 202, 204, 206, 207, 213, 230, 231, 232, 234, 235, 237, 240, 241, 244, 245, 248, 249, 253, 255, 257, 258, 259, 260, 263, 265, 268, 272, 273, 275, 277, 278, 280, 281, 283, 286, 287, 288, 289, 292, 294, 295, 297, 298, 299, 300, 302, 306, 308, 310, 311, 312, 316, 317, 319, 320, 322, 323, 327, 328, 329, 330, 332, 337, 338, 341, 342, 350, 351, 356, 359, 360, 361, 362, 363, 364]
            # new_path = []
            # # import pdb
            # # pdb.set_trace()
            # for i in door_no_demo:
            #     new_path.append(self.total_train_asset_paths[i])
            self.total_train_asset_paths = new_path
            #self.total_train_asset_paths = [Path("assets/door/train/Dishwasher-12092-link_0-handle_0-joint_0-handlejoint_0")]
            # /data2/ziming/RL-Pose/PoseOrientedGym/assets/door/train/Microwave-7221-link_0-handle_0-joint_0-handlejoint_0
            #self.total_valInter_asset_paths = [Path("assets/door/valInter/Oven-7220-link_1-handle_1-joint_1-handlejoint_1")]
        
        elif self.target_part == "button":
            asset_to_remove = ["103452", "103425", "103351", "TrashCan"]
            path_to_remove = []
            for p in path_to_remove:
                if p in new_path:
                    new_path.remove(p)
                    continue
            
            for a in asset_to_remove:
                for p in new_path:
                    if a in str(p):
                        new_path.remove(p)
            for a in asset_to_remove:
                for p in self.total_valIntra_asset_paths:
                    if a in str(p):
                        self.total_valIntra_asset_paths.remove(p)
            for a in asset_to_remove:
                for p in self.total_valInter_asset_paths:
                    if a in str(p):
                        self.total_valInter_asset_paths.remove(p)
            self.total_train_asset_paths = new_path
            #self.total_train_asset_paths = [Path("assets/button/train/Remote-100013-link_39-link_34-NONE-joint_34")]
            #self.total_valInter_asset_paths = [Path("assets/button/valInter/Phone-103813-link_7-link_1-NONE-joint_1")]
        elif self.target_part == "handle":
            path_to_remove = []
            #"103543", "7221", "25493", "7310", 
            asset_to_remove = ["47207", "46537", "19855", "30666", "TrashCan"]
            for p in path_to_remove:
                if p in new_path:
                    new_path.remove(p)
                    continue
            
            for a in asset_to_remove:

                for p in new_path:
                    if a in str(p):
                        new_path.remove(p)
            for a in asset_to_remove:
                for p in self.total_valIntra_asset_paths:
                    if a in str(p):
                        self.total_valIntra_asset_paths.remove(p)
            for a in asset_to_remove:
                for p in self.total_valInter_asset_paths:
                    if a in str(p):
                        self.total_valInter_asset_paths.remove(p)

            self.total_train_asset_paths = new_path
        else:
            raise NotImplementedError
            

        if cfg["env"]["use_easy_data"]:
            self.total_train_asset_paths = assets_to_use_easy
        elif cfg["env"]["single_data"] is not None:
            self.total_train_asset_paths = [Path(cfg["env"]["single_data"])]
        elif len(assets_to_use) != 0:
            self.total_train_asset_paths = assets_to_use

        if cfg["env"]["use_asset_with_demo"]:
            self.total_train_asset_paths = assets_to_imitate
        
        # import pdb
        # pdb.set_trace()
        #elif cfg["task"]["target_part_category"] == 'drawer':
        #    self.total_train_asset_paths = assets_to_use_drawer
        # print(self.total_train_asset_paths)
        if self.cabinet_num_train < 0:
            self.cabinet_num_train = len(self.total_train_asset_paths)
        if self.cabinet_num_valIntra < 0:
            self.cabinet_num_valIntra = len(self.total_valIntra_asset_paths)
        if self.cabinet_num_valInter < 0:
            self.cabinet_num_valInter = len(self.total_valInter_asset_paths)

        self.env_num_train = self.cabinet_num_train * self.env_per_asset
        self.env_num_valIntra = self.cabinet_num_valIntra * self.env_per_asset
        self.env_num_valInter = self.cabinet_num_valInter * self.env_per_asset
        self.total_train_asset = len(self.total_train_asset_paths)
        # import pdb
        # pdb.set_trace()
        self.total_valIntra_asset = len(self.total_valIntra_asset_paths)
        self.total_valInter_asset = len(self.total_valInter_asset_paths)

        self.cabinet_num = self.cabinet_num_train + self.cabinet_num_valIntra\
            + self.cabinet_num_valInter
        self.env_num = self.env_num_train + self.env_num_valIntra + self.env_num_valInter
        
        self.total_cabinet_train_name_list = []
        self.total_cabinet_train_linkname_list = []
        self.total_cabinet_train_handlename_list = []
        self.total_cabinet_train_jointname_list = []
        self.total_cabinet_train_handlejoint_list = []
        self.total_cabinet_valIntra_name_list = []
        self.total_cabinet_valIntra_linkname_list = []
        self.total_cabinet_valIntra_handlename_list = []
        self.total_cabinet_valIntra_jointname_list = []
        self.total_cabinet_valIntra_handlejoint_list = []
        self.total_cabinet_valInter_name_list = []
        self.total_cabinet_valInter_linkname_list = []
        self.total_cabinet_valInter_handlename_list = []
        self.total_cabinet_valInter_jointname_list = []
        self.total_cabinet_valInter_handlejoint_list = []

        if self.target_part in ["door", "drawer", "handle"]:
            for path in self.total_train_asset_paths:
                # print(path.name)
                # import pdb
                
                # pdb.set_trace()
                self.total_cabinet_train_name_list.append(path.name)
                self.total_cabinet_train_linkname_list.append(path.name.split("-")[-4])
                self.total_cabinet_train_handlename_list.append(path.name.split("-")[-3])
                self.total_cabinet_train_jointname_list.append(path.name.split("-")[-2])
                self.total_cabinet_train_handlejoint_list.append(path.name.split("-")[-1])
                # if "TrashCan" in path:
                #     self.total_cabinet_train_name_list.append(path.name)
                #     self.total_cabinet_train_linkname_list.append(path.name.split("-")[-3])
                #     # self.total_cabinet_train_handlename_list.append(path.name.split("-")[-3])
                #     self.total_cabinet_train_jointname_list.append(path.name.split("-")[-1])
                #     # self.total_cabinet_train_handlejoint_list.append(path.name.split("-")[-1])
                    
            for path in self.total_valIntra_asset_paths:
                self.total_cabinet_valIntra_name_list.append(path.name)
                self.total_cabinet_valIntra_linkname_list.append(path.name.split("-")[-4])
                self.total_cabinet_valIntra_handlename_list.append(path.name.split("-")[-3])
                self.total_cabinet_valIntra_jointname_list.append(path.name.split("-")[-2])
                self.total_cabinet_valIntra_handlejoint_list.append(path.name.split("-")[-1])
            for path in self.total_valInter_asset_paths:
                self.total_cabinet_valInter_name_list.append(path.name)
                self.total_cabinet_valInter_linkname_list.append(path.name.split("-")[-4])
                self.total_cabinet_valInter_handlename_list.append(path.name.split("-")[-3])
                self.total_cabinet_valInter_jointname_list.append(path.name.split("-")[-2])
                self.total_cabinet_valInter_handlejoint_list.append(path.name.split("-")[-1])
            assert(self.cabinet_num_train <= self.total_train_asset)
            assert(self.cabinet_num_valIntra <= self.total_valIntra_asset)
            assert(self.cabinet_num_valInter <= self.total_valInter_asset)
        elif  self.target_part == "button":
            for path in self.total_train_asset_paths:
                # print(path.name)
                self.total_cabinet_train_name_list.append(path.name)
                self.total_cabinet_train_linkname_list.append(path.name.split("-")[-3])
                # self.total_cabinet_train_handlename_list.append(path.name.split("-")[-3])
                self.total_cabinet_train_jointname_list.append(path.name.split("-")[-1])
                # self.total_cabinet_train_handlejoint_list.append(path.name.split("-")[-1])
            for path in self.total_valIntra_asset_paths:
                self.total_cabinet_valIntra_name_list.append(path.name)
                self.total_cabinet_valIntra_linkname_list.append(path.name.split("-")[-3])
                # self.total_cabinet_valIntra_handlename_list.append(path.name.split("-")[-3])
                self.total_cabinet_valIntra_jointname_list.append(path.name.split("-")[-1])
                # self.total_cabinet_valIntra_handlejoint_list.append(path.name.split("-")[-1])
            for path in self.total_valInter_asset_paths:
                self.total_cabinet_valInter_name_list.append(path.name)
                self.total_cabinet_valInter_linkname_list.append(path.name.split("-")[-3])
                # self.total_cabinet_valInter_handlename_list.append(path.name.split("-")[-3])
                self.total_cabinet_valInter_jointname_list.append(path.name.split("-")[-1])
                # self.total_cabinet_valInter_handlejoint_list.append(path.name.split("-")[-1])
            assert(self.cabinet_num_train <= self.total_train_asset)
            assert(self.cabinet_num_valIntra <= self.total_valIntra_asset)
            assert(self.cabinet_num_valInter <= self.total_valInter_asset)


        # import pdb
        # pdb.set_trace()
        self.obj_loaded = False
        self.franka_loaded = False
        self.franka_name = cfg["env"]["robotName"]
        self.franka_file = cfg["env"]["asset"]["robot"][self.franka_name]["filePath"]
        self.num_actions = cfg["env"]["asset"]["robot"][self.franka_name]["numActions"]
        self.num_qpose_actions = cfg["env"]["asset"]["robot"][self.franka_name]["numActions"]
        # import pdb
        # pdb.set_trace()
        if "ik" in self.cfg["env"]["driveMode"]:
            self.num_actions = cfg["env"]["asset"]["robot"][self.franka_name]["ikNumActions"]
            if "axis_angle" in self.cfg["env"]["driveMode"]:
                self.num_actions -= 1
                
        # env, obs, task meta
        self.env_name = cfg["env"]["env_name"]
        self.obs_dim = 0
        self.state_dim = self.num_qpose_actions * 2 + 2 + 16
        if self.cfg["obs"]["state"]["use_bbox_form"] == "vertex": # 48
            self.state_dim += 48
        elif self.cfg["obs"]["state"]["use_bbox_form"] == "edges": # 24
            self.state_dim += 24
        if self.cfg["obs"]["state"]["use_noise_feature"]:
            self.state_dim += 112


        if self.cfg["obs"]["state"]["use_robot_qpose"]: # numActions * 2 # 22 for slider
            self.obs_dim += self.num_qpose_actions * 2
        if self.cfg["obs"]["state"]["use_cabinet_qpose"]: # 2
            self.obs_dim += 2
        if self.cfg["obs"]["state"]["use_hand_root_pose"]: # 16
            self.obs_dim += 16
        if self.cfg["obs"]["state"]["use_part_center"]: # 3
            self.obs_dim += 3
        if self.cfg["obs"]["state"]["use_cabinet_bbox"]:
            if self.cfg["obs"]["state"]["use_bbox_form"] == "vertex": # 48
                self.obs_dim += 48
            elif self.cfg["obs"]["state"]["use_bbox_form"] == "edges": # 24
                self.obs_dim += 24
        if self.cfg["obs"]["state"]["use_noise_feature"]:
            self.obs_dim += 112
        print("Env: number of cabinets", self.cabinet_num)
        print("Env: number of environments", self.env_num)
        self.task_meta = {
            "train_cabinet_num": self.cabinet_num_train,
            "valIntra_cabinet_num": self.cabinet_num_valIntra,
            "valInter_cabinet_num": self.cabinet_num_valInter,
            "env_per_asset": self.env_per_asset,
            "max_episode_length": self.max_episode_length,
            "obs_dim": self.obs_dim,
            "train_env_num": self.cabinet_num_train * self.env_per_asset,
            "valIntra_env_num": self.cabinet_num_valIntra * self.env_per_asset,
            "valInter_env_num": self.cabinet_num_valInter * self.env_per_asset,
            "need_update": True, # TODO
        }
        self.cabinet_target_joint_lower_limits_tensor = torch.zeros(
            (self.cabinet_num, 1), device=self.device)
        self.cabinet_target_joint_upper_limits_tensor = torch.zeros(
            (self.cabinet_num, 1), device=self.device)
        self.cabinet_open_dir_tensor = torch.zeros(
            (self.cabinet_num,), device=self.device)


        self.env_ptr_list = []
        self.cabinet_actor_list = []
        
        # create sim, load object, actor
        super().__init__(
            cfg = self.cfg,
            enable_camera_sensors=cfg["env"]["enableCameraSensors"]
        )
        #print("after super init", wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)))
        
        # from simulator acquire tensors
        # print(self.selected_asset_path_list)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # print(int(gymapi.VhacdParams.max_convex_hulls))

        #print("191", wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)))
        self.root_tensor = wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))

        if "ik" in self.cfg["env"]["driveMode"]:
            self.jacobian_tensor = wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "franka"))
        self.dof_state_tensor_all = wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        
        self.rigid_body_tensor_all = wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))

        self.dof_state_tensor_used, self.dof_state_mask, initial_env_dof_sate_list = \
            deal_with_dof_state(self.dof_state_tensor_all, self.part_dof_ids, self.cabinet_dof_num, 
            self.franka_num_dofs, self.env_num, self.env_per_asset)
        
        if self.target_part in ["door", "drawer", "handle"]:
            self.rigid_body_tensor_used, self.rigid_state_mask = \
                deal_with_rigid_state(self.rigid_body_tensor_all, self.part_rigid_ids,
                self.handle_rigid_ids,self.base_rigid_ids,self.rigid_num_list, 
                self.franka_rigid_num, self.env_num, self.env_per_asset)
        else:
            self.rigid_body_tensor_used, self.rigid_state_mask = \
                deal_with_rigid_state_nohandle(self.rigid_body_tensor_all, self.part_rigid_ids,
                self.base_rigid_ids,self.rigid_num_list, 
                self.franka_rigid_num, self.env_num, self.env_per_asset)
            
        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)
        self.actions = torch.zeros((self.env_num, self.num_actions), device=self.device)

        self.initial_dof_used = self.dof_state_tensor_used.clone()
        self.initial_dof_states = self.dof_state_tensor_all.clone()
        self.initial_root_states = self.root_tensor.clone()
        # precise for slices of tensors
        env_ptr = self.env_ptr_list[0]
        franka1_actor = self.franka_actor

        self.hand_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,franka1_actor,"panda_hand",gymapi.DOMAIN_ENV) # 10
        self.hand_lfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,franka1_actor,"panda_leftfinger",gymapi.DOMAIN_ENV) # 11
        self.hand_rfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,franka1_actor,"panda_rightfinger",gymapi.DOMAIN_ENV) # 12
        
        # self.cabinet_base_rigid_body_index = self.hand_rfinger_rigid_body_index + 1
        # self.cabinet_part_rigid_body_index = self.hand_rfinger_rigid_body_index + 2
        # self.cabinet_handle_rigid_body_index = self.hand_rfinger_rigid_body_index + 3
        self.cabinet_dof_index = self.franka_num_dofs

        self.hand_rigid_body_tensor = self.rigid_body_tensor_used\
            [:, self.hand_rigid_body_index, :] # N*13
        self.franka_dof_tensor = self.dof_state_tensor_used\
            [:, :self.franka_num_dofs, :] # N*11*2
        self.cabinet_dof_tensor = self.dof_state_tensor_used\
            [:, self.cabinet_dof_index, :] # N*2
        self.cabinet_dof_tensor_spec = self.cabinet_dof_tensor.view(
            self.cabinet_num, self.env_per_asset, -1) # M*(N/M)*2

        # self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self, self.cabinet_dof_tensor[:,0])
            
        if self.target_part in ["door", "handle"]:
            if cfg["task"]["use_first_bbox"] == 'pt':
                self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_pt(self)
            else:
                self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self, self.cabinet_dof_tensor[:,0], 0)
        
        elif self.target_part == "drawer":            
            if cfg["task"]["use_first_bbox"] == 'pt':
                self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_pt(self)
            else:
                self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self, self.cabinet_dof_tensor[:,0], 1)
        elif self.target_part == "button":
            if cfg["task"]["use_first_bbox"] == 'pt':
                self.part_bbox_tensor = get_bbox_pt_nohandle(self)
            else:
                self.part_bbox_tensor = get_bbox_isaac_tensor_nohandle(self, self.cabinet_dof_tensor[:,0], 1)
        else:
            raise NotImplementedError
        
        
        self.init_part_bbox_tensor = self.part_bbox_tensor
        if self.target_part in ["door", "drawer", "handle"]:
            self.init_handle_bbox_tensor = self.handle_bbox_tensor

        # tmp_tensor = torch.arange(self.num_envs, device=self.device).long()
        # exit(123)
        self.cabinet_base_rigid_body_tensor = self.rigid_body_tensor_used[:, self.franka_num_dofs, :] 
        self.cabinet_part_rigid_body_tensor = self.rigid_body_tensor_used[:, self.franka_num_dofs+1, :] 
        if self.target_part in ["door", "drawer", "handle"]:
            self.cabinet_handle_rigid_body_tensor = self.rigid_body_tensor_used[:, self.franka_num_dofs+2, :] 
        
        # self.cabinet_handle_pos_tensor = self.cabinet_handle_rigid_body_tensor[:,:3]
        self.franka_root_tensor = self.root_tensor[:, 0, :] # N*13
        self.cabinet_root_tensor = self.root_tensor[:, 1, :] # N*13

        if cfg["task"]["target_state"] == "close" or cfg["task"]["target_state"] == "push_button":
            self.cabinet_dof_target = torch.zeros_like(self.dof_state_tensor_used[
            :, self.cabinet_dof_index, 0], device=self.device)
        else:
            self.cabinet_dof_target = torch.ones_like(self.dof_state_tensor_used[
            :, self.cabinet_dof_index, 0], device=self.device) * self.cfg["task"]["openProportion"]

        self.dof_dim = self.franka_num_dofs + 1
        self.pos_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.vel_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.eff_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.pos_act_all = torch.zeros_like(self.dof_state_tensor_all[:,0], device = self.device)
        self.vel_act_all = torch.zeros_like(self.dof_state_tensor_all[:,0], device = self.device)
        self.eff_act_all = torch.zeros_like(self.dof_state_tensor_all[:,0], device = self.device)

        self.stage = torch.zeros((self.num_envs), device=self.device)
        from scipy.spatial.transform import Rotation as R
        matrix = torch.tensor(R.from_quat(object_init_pose_r_np).as_matrix().astype(np.float32), device = self.device).reshape(1, 3,3).repeat_interleave(self.num_envs, dim=0)
        self.part_axis_xyz_tensor = torch.bmm(matrix, self.part_axis_xyz_tensor_init.reshape(-1, 3, 1)).reshape(-1,3) + torch.tensor(+object_init_pose_p_np.astype(np.float32), device = self.device)
        self.part_axis_dir_tensor = torch.bmm(matrix, self.part_axis_dir_tensor_init.reshape(-1, 3, 1)).reshape(-1,3)

        if cfg["task"]["target_state"] == "close" :
            self.cabinet_dof_coef = -1.0
            self.success_dof_states = self.cabinet_target_joint_lower_limits_tensor[:, 0].clone()
            # import pdb
            # pdb.set_trace()
            if cfg['task']['target_part_category'] == "drawer":
                #self.initial_dof_states.view(self.cabinet_num, self.env_per_asset, -1, 2)[:,:,self.cabinet_dof_index+1, 0]\
                #    =torch.ones((self.cabinet_num, 1), device=self.device) * cfg['task']['target_part_initial_open_proportion']
                
                for mask_l in self.dof_state_mask:
                    for i in range(len(mask_l)):
                        if i != self.cabinet_dof_index:
                            continue
                        self.initial_dof_states[mask_l[i], 0] = cfg['task']['target_part_initial_open_proportion']

                # self.initial_dof_states[self.dof_state_mask][:, self.cabinet_dof_index, 0]\
                #     = torch.ones((self.cabinet_num * self.env_per_asset), device=self.device) * cfg['task']['target_part_initial_open_proportion']
                
                
                # self.initial_dof_used.view(self.cabinet_num,     ###############????????????
                #     self.env_per_asset, -1, 2)[:,:,self.cabinet_dof_index, 0]\
                #     =torch.ones((self.cabinet_num, 1), device=self.device) * cfg['task']['target_part_initial_open_proportion']
            elif cfg['task']['target_part_category'] in ["door", "handle"]:
                for mask_l in self.dof_state_mask:
                    for i in range(len(mask_l)):
                        if i != self.cabinet_dof_index:
                            continue
                        self.initial_dof_states[mask_l[i], 0] = cfg['task']['target_part_initial_open_proportion']

                # self.initial_dof_used.view(self.cabinet_num, 
                    # self.env_per_asset, -1, 2)[:,:,self.cabinet_dof_index, 0]\
                    # =torch.ones((self.cabinet_num, 1), device=self.device) * cfg['task']['target_part_initial_open_proportion']
            else:
                assert(0)
            # self.gym.refresh_jacobian_tensors(self.sim)
            # self.gym.set_dof_state_tensor(
            #     self.sim,
            #     gymtorch.unwrap_tensor(self.initial_dof_states)
            # )
            # self.dof_state_tensor = self.initial_dof_states
            # self.gym.set_dof_state_tensor(
            #     self.sim,
            #     gymtorch.unwrap_tensor(self.dof_state_tensor)
            # )
            # import pdb
            # pdb.set_trace() 
        else :  # open, push button
            self.cabinet_dof_coef = +1.

            self.success_dof_states = torch.ones((self.cabinet_num,), 
                device=self.device) * self.cfg["task"]["openProportion"] # raw 0.05

            self.initial_dof_used.view(self.cabinet_num, 
                self.env_per_asset, -1, 2)[:, :, self.cabinet_dof_index,0]\
                = torch.zeros((self.cabinet_num, 1), device=self.device)
                
        self.initial_dof_states_plan = None
        # from isaacgym.gymtorch import wrap_tensor
        # print("before pose init", wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)))å
        # initialization of pose
        #if self.cfg["env"]["usePreGraspPose"]:
            # if  self.cfg["env"]["useGtBbox"]:
            #     part_bbox_tensor = self.part_bbox_tensor
            #     handle_bbox_tensor = self.handle_bbox_tensor
            # else:
            #     raise NotImplementedError
            # part_pos = self.cabinet_part_rigid_body_tensor[:, :3]
            # part_rot = self.cabinet_part_rigid_body_tensor[:, 3:7]
            # hand_rot = self.hand_rigid_body_tensor[..., 3:7]
            # handle_out = self.handle_bbox_tensor[:, 0] - self.handle_bbox_tensor[:, 4]
            # handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
            # handle_long = self.handle_bbox_tensor[:, 1] - self.handle_bbox_tensor[:, 0]
            # handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
            # handle_short = self.handle_bbox_tensor[:, 3] - self.handle_bbox_tensor[:, 0]
            # handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
            # hand_pose = torch.zeros((handle_out.shape[0], 7), device = self.device)
            # rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))

            # hand_pose[:,:3] = (handle_bbox_tensor[:,0,:] + handle_bbox_tensor[:,6,:])/2 + torch.tensor([0.2, 0, 0], device = self.device)
            # hand_pose[:,3:7] = rotations
            # #print(hand_pose)
            # print("qpose", _get_qpos_from_hand_pose(self, hand_pose))
            
            # initial_tensor = torch.zeros((self.pos_act_all.shape[0],2), device = self.device)
            # # print(initial_tensor.shape)
            # # print("initial_tensor",initial_tensor.shape)
            # # print(" initial_tensor[self.dof_state_mask].view(self.env_num, -1, 2)", initial_tensor[self.dof_state_mask].view(self.env_num, -1, 2).shape)
            # a = initial_tensor[self.dof_state_mask]
            # #print(self.dof_state_mask)
            # #print(a.shape)
            # a.view(self.env_num, -1, 2)[...,:-1,0] = _get_qpos_from_hand_pose(self, hand_pose)#[:,:-2]# = torch.zeros((hand_pose.shape[0], self.franka_num_dofs))
            # #print(a.shape)
            # for i in range(a.shape[0]):
            #     initial_tensor[self.dof_state_mask[i]] = a[i]
            
            # #print(_get_qpos_from_hand_pose(self, hand_pose))
            # # print(self.franka_dof_tensor.shape)
            # # print(self.cabinet_dof_tensor.shape)
            # # initial_tensor[:,:-2] = _get_qpos_from_hand_pose(self, hand_pose)
            # # self.env_num = 1

            # pre = self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(initial_tensor), gymtorch.unwrap_tensor(torch.arange(self.env_num, device = self.device, dtype = torch.int32)*2), self.env_num,)
            # #print("pre", pre)
            # self.initial_dof_states = initial_tensor
            # #print("init", initial_tensor)
            # pass
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        #exit()
        #print("init", self.initial_dof_states)
        self.cabinet_reset_position_noise = 0
        self.cabinet_reset_rotation_noise = 0
        self.cabinet_reset_dof_pos_interval = 0
        self.cabinet_reset_dof_vel_interval = 0
        self.franka_reset_position_noise = 0
        self.franka_reset_rotation_noise = 0
        self.franka_reset_dof_pos_interval = 0
        self.franka_reset_dof_vel_interval = 0

        # params for success rate
        
        self.success = torch.zeros((self.env_num,), device=self.device)
        self.success_rate = torch.zeros((self.env_num,), device=self.device)
        self.current_success_rate = torch.zeros((self.env_num,), device = self.device)
        self.success_queue = torch.zeros((self.env_num, 1), device=self.device)
        self.success_idx = torch.zeros((self.env_num,), device=self.device).long()
        self.success_buf = torch.zeros((self.env_num,), device=self.device).long()
        self.success_grasp_buf = torch.zeros((self.env_num,), device=self.device).long()

        self.average_reward = None

        # flags for switching between training and evaluation mode
        self.train_mode = True
        # if cfg["task"]["use_qpose_state"] :
        #     self.qpose_num = (self.franka_num_dofs - 4)*2+2 
        # else:
        #     self.qpose_num = 0

        if cfg["task"]["canonicalize"]:
            
            if self.target_part in ["door", "drawer", "handle"]:
                if cfg["task"]["canon_space"] == 'part': ######???????
                    self.canon_center = (self.part_bbox_tensor[:, 0, :] + self.part_bbox_tensor[:, 6, :]) / 2
                    out = self.part_bbox_tensor[:, 0] - self.part_bbox_tensor[:, 4]
                    long = self.part_bbox_tensor[:, 1] - self.part_bbox_tensor[:, 0]
                    short = self.part_bbox_tensor[:, 3] - self.part_bbox_tensor[:, 0]
                elif cfg["task"]["canon_space"] == 'handle':
                    self.canon_center = (self.handle_bbox_tensor[:, 0, :] + self.handle_bbox_tensor[:, 6, :]) / 2
                    out = self.handle_bbox_tensor[:, 0] - self.handle_bbox_tensor[:, 4]
                    long = self.handle_bbox_tensor[:, 1] - self.handle_bbox_tensor[:, 0]
                    short = self.handle_bbox_tensor[:, 3] - self.handle_bbox_tensor[:, 0]
                else:
                    raise NotImplementedError
            else:
                self.canon_center = (self.part_bbox_tensor[:, 0, :] + self.part_bbox_tensor[:, 6, :]) / 2
                out = self.part_bbox_tensor[:, 0] - self.part_bbox_tensor[:, 4]
                long = self.part_bbox_tensor[:, 1] - self.part_bbox_tensor[:, 0]
                short = self.part_bbox_tensor[:, 3] - self.part_bbox_tensor[:, 0]
            
            
            if cfg["task"]["use_first_bbox"] == 'pt':
                out_dir = torch.tensor([1.0, 0, 0], device=self.device)
                out_dir = out_dir.repeat(self.env_num, 1)
                long_dir = torch.tensor([0, 0, 1.0], device=self.device)
                long_dir = long_dir.repeat(self.env_num, 1)
                short_dir = torch.tensor([0, 1.0, 0], device=self.device)
                short_dir = short_dir.repeat(self.env_num, 1)
            else :
                out_dir = out / torch.norm(out, dim = 1, keepdim=True)
                long_dir = long / torch.norm(long, dim = 1, keepdim=True)
                short_dir = short /torch.norm(short, dim = 1, keepdim=True)

            self.canon_rot_matrix = torch.cat([out_dir.view(-1, 1, 3), short_dir.view(-1, 1, 3), long_dir.view(-1, 1, 3)], dim = 1)
            self.canon_quaternion_rot = matrix_to_quaternion(self.canon_rot_matrix)
            self.canon_quaternion_rot_invert = quaternion_invert(self.canon_quaternion_rot)


    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(
            self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._place_agents(self.num_envs, self.cfg["env"]["envSpacing"])

    def _load_franka(self, env_ptr, env_id):
        _load_franka(self, env_ptr=env_ptr, env_id=env_id)
        
    def _load_obj_asset(self, progress):
        return _load_obj_asset(self, progress)

    def _load_obj(self, env_ptr, env_id):
        _load_obj(self, env_ptr, env_id)

    def _place_agents(self, env_num, spacing, use_cam = False):
        ##############这里之前改的不适配state base, 所以先吧所有use_cam都设置成false了#####
        if self.cfg["obs"]["pc"]["use_pseudo_pc"] or not self.cfg["obs"]["use_pc"]:
            return _place_agents(self, env_num, spacing, use_cam=False)
        return _place_agents(self, env_num, spacing, use_cam = True)

    def _create_ground_plane(self):
        return _create_ground_plane(self)

    def _get_reward_done(self):
        #return get_reward_done_all(self)
        return get_reward_done(self)
      
    def _get_base_observation(self, suggested_gt=None, pregrasp=False) :
        return _get_base_observation(self, suggested_gt)

    def _refresh_observation(self, pregrasp=False) :
        return _refresh_observation(self)

    def _perform_actions(self, actions):
        return _perform_actions(self, actions)

    def step(self, actions) :
        return step(self, actions)

    def control_step(self, actions):
        return control_step(self, actions)

    def _partial_reset(self, to_reset = "all") :
        return _partial_reset(self, to_reset)
   
    def reset(self, to_reset = "all") :
        return reset(self, to_reset)

    def save(self, path, iteration) :
        return save(self, path, iteration)

    def load(self, path, iteration) :
        pass
    
    def update(self, it=0) :
        pass
    
    def train(self) :        # changing mode to eval
        self.train_mode = True

    def eval(self) :        # changing mode to eval
        self.train_mode = False

    def intervaledRandom_(self, tensor, dist, lower=None, upper=None) :
        tensor += torch.rand(tensor.shape, device=self.device)*dist*2 - dist
        if lower is not None and upper is not None :
            torch.clamp_(tensor, min=lower, max=upper)
 
