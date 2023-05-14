import yaml
import wandb
import sys,os
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
from .gym_info import task_names, algo_names
from .gym_info import *
import random

from collections import abc
from argparse import ArgumentParser

import torch

def parse_name(args, cfg, cfg_algo):
    import datetime

    currtime = "{0:%Y-%m-%d_%H:%M:%S}".format(datetime.datetime.now())
    if cfg["obs"]["use_pc"]:
        type_ = "pc"
    else:
        type_ = "state"
    if cfg["obs"]["pc"]["use_pseudo_pc"]:
        pc = "pseudo"
    else:
        pc= "cam"
    grasp = "G-" + cfg["task"]["pregrasp_type"]
    if cfg["obs"]["state"]["use_cabinet_bbox"]:
        bbox = "B-" + cfg["obs"]["state"]["use_bbox_type"]
    else:
        bbox = "Nobbox"
    if args.algo in ["heuristics"]:
        nsteps = backbone = ""
    else:
        backbone = cfg_algo["policy"]["backbone_type"]
        nsteps = cfg_algo["learn"]["nsteps"]
    step = f"N-{nsteps}"
    envs = str(cfg["env"]["AssetNumTrain"]) + "_" + str(cfg["env"]["AssetNumValIntra"]) + "_" + str(cfg["env"]["AssetNumValInter"]) + "_" + str(cfg["env"]["EnvPerAsset"])
    obs ="" 
    if cfg["obs"]["state"]["use_robot_qpose"]: obs += "T" 
    else: obs += "F"
    if cfg["obs"]["state"]["use_cabinet_qpose"]:obs += "T"
    else: obs += "F"
    if cfg["obs"]["state"]["use_cabinet_bbox"]:obs += "T"
    else: obs += "F"
    if cfg["obs"]["state"]["use_hand_root_pose"]:obs += "T"
    else: obs += "F"
    if cfg["obs"]["state"]["use_part_center"]:obs += "T"
    else: obs += "F"
    if cfg["obs"]["pc"]["numPCObservations"]==7:obs += "T"
    else: obs += "F"
    
    dirveMode = cfg["env"]["driveMode"]
    if args.algo in ["heuristics"]:
        lr = ""
    else:
        lr = str(cfg_algo["policy"]["dagger_loss_supervise"]) + "_" + str(cfg_algo["policy"]["dagger_loss_seg"]) + "_" + str(cfg_algo["policy"]["dagger_loss_RL"]) + "_" + str(cfg_algo["policy"]["dagger_loss_decay_proportion"]) + "_"

    setting = ""
    if args.algo in ["heuristics"]:
        pass
    else:
        if cfg_algo["policy"]["use_dagger_activation"]: setting += "Act_"
        else: setting += "Noa_"
        if cfg_algo["policy"]["actor_freeze"]: setting += "Fre_"
        else: setting += "Nof_"
    


    pc_num = str(cfg["obs"]["pc"]["pcDownSampleNum"])

    if args.algo in ["heuristics"]:
        freeze = ""
    else:
        if cfg_algo["policy"]["freeze"]:
            freeze = "freeze"
        else:
            freeze = "free"
    cfg["env"]["env_name"] =  cfg["log"]["save_name"] + "_" + cfg["task"]["target_part_category"] + "_" + type_ +"_"+ pc +"_"
    if backbone != None and backbone != "None":
        cfg["env"]["env_name"] += backbone + "_"
    cfg["env"]["env_name"] += dirveMode + "_"+ setting + lr + grasp +"_"+ bbox +"_"+step+"_"+envs + "_"+obs + "_" +pc_num + "_" +freeze
    
    cfg["log"]["group_name"] = cfg_algo["algo"]+"_"+cfg["log"]["group_name"]+"_"+cfg["env"]["env_name"]
    cfg["env"]["env_name"] += "_" + currtime
    cfg["log"]["save_name"] = cfg["env"]["env_name"]
    save_name =  cfg["log"]["save_name"]
    group_name =  cfg["log"]["group_name"]
    return save_name, group_name
    

def add_args(parser, cfg, prefix=""):
    '''
    Generate argparser from config file automatically.

    Examples:
        cfg = {
            'A': {'B': 1.0, 'C': False},
            'D': [1, 2, 3],
            'E': 'xxx',
            'F': None,
        }

        To set cfg['A']['B']=2.0, use "--A.B 2.0" in command.
        To set cfg['A']['D']=[2, 3, 4], use "--A.D 2 3 4" in command.
        To set cfg['A']['C']=True, use "--A.C" in command.
    
    NOTE:
        1. We can't change the type of variable, i.e. can't use "--A.E 2" in command.
        2. None in config file is set to str as default.
    '''
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v:
                parser.add_argument("--" + prefix + k, default=None, action="store_false")
            else:
                parser.add_argument("--" + prefix + k, default=None, action="store_true")
        elif isinstance(v, int):
            parser.add_argument("--" + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument("--" + prefix + k, type=float)
        elif isinstance(v, str) or v is None:
            parser.add_argument("--" + prefix + k)
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + ".")
        elif isinstance(v, abc.Iterable):
            parser.add_argument("--" + prefix + k, type=type(v[0]), nargs="+")
        else:
            print(f"WARNING: cannot parse key {prefix + k} of type {type(v)}")

    return parser

def set_np_formatting():
    
    edgeitems=30
    infstr="inf"
    linewidth=4000
    nanstr="nan"
    precision=2
    suppress=False
    threshold=1000
    formatter=None
    
    np.set_printoptions(
        edgeitems=edgeitems, infstr=infstr, linewidth=linewidth, nanstr=nanstr, precision=precision,
        suppress=suppress, threshold=threshold, formatter=formatter,
    )

def warn_task_name():
    raise Exception("Unrecognized task!")

def warn_algo_name():
    raise Exception("Unrecognized algo!")

def set_seed(seed, torch_deterministic = False):
    if seed == -1 and torch_deterministic:
        seed = 2333
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed) # random seed for random module
    np.random.seed(seed) # for np module
    torch.manual_seed(seed) # for pytorch module
    os.environ['PYTHONHASHSEED'] = str(seed) # for os env Python hash seed
    torch.cuda.manual_seed(seed) # cuda manual seed
    torch.cuda.manual_seed_all(seed) # cuda manual seed all

    if torch_deterministic: # torch deterministic
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return seed

def load_cfg(args):
    with open(pjoin(os.getcwd(), args.algo_config), "r") as f:
        cfg_algo = yaml.load(f, Loader=yaml.SafeLoader)
    with open(pjoin(os.getcwd(), args.task_config), "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        
    if args.model_dir != None and args.model_dir != "":
        args.ckpt = args.model_dir
    if args.ckpt != None and args.ckpt != "":
        args.model_dir = args.ckpt
    if args.control != None and args.control != "":
        cfg["env"]["driveMode"] = args.control

        
    
    #override
    # if args.asset_num_valIntra != 0 or args.asset_num_train != 0 or args.asset_num_valInter != 0:
    #     cfg["env"]["EnvPerAsset"] = args.env_per_asset
    # if args.asset_num_train != 0:
    #     cfg["env"]["AssetNumTrain"] = args.asset_num_train
    # if args.asset_num_valIntra != 0:
    #     cfg["env"]["AssetNumValIntra"] = args.asset_num_valIntra
    # if args.asset_num_valInter != 0:
    #     cfg["env"]["AssetNumValInter"] = args.asset_num_valInter
    try:
        cfg["env"]["AssetNumTrain"], cfg["env"]["AssetNumValIntra"], cfg["env"]["AssetNumValInter"], cfg["env"]["EnvPerAsset"] = \
            [int(x) for x in args.asset_num.split("_")]
    except:
        pass
        
    # Overwrite coefficient in cfg["task"]
    if args.seed is not None: cfg_algo["seed"] = args.seed
    if args.rot_coef is not None: cfg["task"]["rot_coef"] = args.rot_coef
    if args.handle_coef is not None: cfg["task"]["handle_coef"] = args.handle_coef
    if args.vel_coef is not None: cfg["task"]["vel_coef"] = args.vel_coef
    if args.dist_coef is not None: cfg["task"]["dist_coef"] = args.dist_coef
    if args.tip_coef is not None: cfg["task"]["tip_coef"] = args.tip_coef
    if args.dicouple_coef is not None: cfg["task"]["dicouple_coef"] = args.dicouple_coef
    if args.close_coef is not None: cfg["task"]["close_coef"] = args.close_coef
    cfg["task"]["use_soft_dist"] = args.use_soft_dist
    cfg["task"]["use_finger_dist"] = args.use_finger_dist
    cfg_algo["algo"] = args.algo
    ###for close task###
    if cfg["task"]["task"] == "close":
        if cfg["task"]["target_part_category"] == "door":
            cfg["task"]["target_part_initial_open_proportion"] = 0.78
        elif cfg["task"]["target_part_category"] == "drawer":
            cfg["task"]["target_part_initial_open_proportion"] = 0.5
    if cfg["task"]["target_part_category"] == "button":
        cfg["task"]["task"] = "push_button"
        cfg["env"]["franka_scale"] = 2.5
    if cfg["task"]["target_part_category"] == "handle":
        cfg["task"]["task"] = "grasp"
    if args.category is not None: 
        cfg["task"]["target_part_category"] = args.category
        if cfg["task"]["target_part_category"] == "door" and cfg["task"]["task"] == "open":
            cfg["task"]["openProportion"] = 0.53
        if cfg["task"]["target_part_category"] == "drawer" and cfg["task"]["task"] == "open":
            cfg["task"]["openProportion"] = 0.2
    
    # if args.openProportion is not None: cfg["task"]["openProportion"] = args.openProportion
    
    if args.contact_offset is not None: sim_params_physx_contact_offset = args.contact_offset
    cfg["env"]["use_easy_data"] = args.use_easy_data
    
    
    # import pdb
    # pdb.set_trace()
    #####################
    if args.noptepochs is not None: 
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["learn"]["noptepochs"] = args.noptepochs
    if args.minibatches is not None: 
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["learn"]["nminibatches"] = args.minibatches
    if args.eval_round is not None: 
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["learn"]["eval_round"] = args.eval_round
    if args.eval_freq is not None: 
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["learn"]["eval_freq"] = args.eval_freq
            cfg_algo["learn"]["save_interval"] = args.eval_freq
    if args.desired_kl is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["learn"]["desired_kl"] = args.desired_kl
    # print (cfg_algo["learn"]["eval_freq"] )
    # exit()
    cfg["debug"] = args.debug
    if cfg["debug"]:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["use_pc"] = False
            cfg_algo["policy"]["GAIL_debug"] = True
        cfg["log"]["use_wandb"] = False
        

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    if args.target is not None: cfg["task"]["task"] = args.target
    

    # if args.algo not in no_training_algos:
    #     if cfg["task"]["task"] == 'grasp':
    #         cfg["env"]["maxEpisodeLength"] = 100
    #         cfg_algo["learn"]["nsteps"] = 100
    #     elif cfg["task"]["task"] == 'open':
    #         cfg["env"]["maxEpisodeLength"] = 200
    #         cfg_algo["learn"]["nsteps"] = 200    
            
    if args.robot_name is not None :
        cfg["env"]["robotName"] = args.robot_name

    if args.canonical_space != None:
        if args.canonical_space == "None":
            cfg["task"]["canonicalize"] = False
            if args.algo in ["heuristics"]:
                pass
            else:
                cfg_algo["policy"]["canonicalize"] = False
        else:
            cfg["task"]["canonicalize"] = True
            cfg["task"]["canon_space"] = args.canonical_space
            if args.algo in ["heuristics"]:
                pass
            else:
                cfg_algo["policy"]["canonicalize"] = True
                cfg_algo["policy"]["canon_space"] = args.canonical_space

    
    cfg["task"]["use_qpose_state"] = args.use_qpose_state
    if args.algo in ["heuristics"]:
        pass
    else:
        cfg_algo["policy"]["use_dagger_activation"] = args.use_dagger_activation
        cfg_algo["policy"]["actor_freeze"] = args.actor_freeze
    if args.use_first_bbox is not None:
        cfg["task"]["use_first_bbox"] = args.use_first_bbox
        # if args.use_first_bbox:
        # cfg["task"]["use_bbox"] = True

    if args.gen_pc:
        cfg["gen_pc"] = True

    cfg["name"] = args.task
    cfg["headless"] = args.headless
    cfg["graphics_device_id"] = args.graphics_device_id
    if args.gen_pc:
        cfg["gen_pc"] = args.gen_pc
    else:
        cfg["gen_pc"] = False

    # if not cfg["task"]["use_bbox"]:
    #     cfg["env"]["numObservations"] -= 24
    
    if args.algo in ["heuristics"]:
        pass
    else:
        if cfg_algo["policy"].get("debug", False):
            cfg_algo["policy"]["task_meta"]["state_dim"] = 56
            cfg_algo["policy"]["use_pc"] = False


    # if cfg_algo["policy"]["use_residual_traj"]:
    #     cfg["env"]["numObservations"] += 3
    #     cfg_algo["policy"]["task_meta"]["state_dim"] += 3

    if cfg["env"]["robotName"] == "FrankaSlider" or cfg["env"]["robotName"] == "FrankaSlider_newtips":
        qpose_num = 16
    elif cfg["env"]["robotName"] == "Frankafixed" or cfg["env"]["robotName"] == "Frankafixed_newtips":
        qpose_num = 12
    else:
        raise NotImplementedError

    # if not cfg["task"]["use_qpose_state"]:
    #     cfg["env"]["numObservations"] -= qpose_num

    if args.expert_dir is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["expert_policy"]["expert_path"] = args.expert_dir

    if args.single_data is not None:
        cfg["env"]["single_data"] = args.single_data
    else:
        cfg["env"]["single_data"] = None

    if args.use_bbox_pos is not None:
        if args.use_bbox_pos == "None":
            cfg["obs"]["state"]["use_cabinet_bbox"] = False
        else:
            cfg["obs"]["state"]["use_cabinet_bbox"] = True
            cfg["obs"]["state"]["use_bbox_type"] = args.use_bbox_pos
        
    if args.obs == "state":
        #这里写的有问题, 在collect data的时候obs需要用pc, 但是model不用pc
        if args.algo != "collect_data":
            cfg_algo["policy"]["use_pc"] = False
        cfg["env"]["enableCameraSensors"] = False
        cfg["obs"]["use_pc"] = False
        cfg["obs"]["state"]["use_part_center"] = False
        cfg["obs"]["state"]["use_cabinet_qpose"] = True
        cfg["obs"]["state"]["use_cabinet_bbox"] = True 
    elif args.obs == "pc":
        cfg["obs"]["use_pc"] = True
        # cfg_algo["policy"]["use_pc"] = True
        if args.algo != "collect_data":
            if args.algo in ["heuristics"]:
                pass
            else:
                cfg_algo["policy"]["use_pc"] = True
        if args.pc == "pseudo":
            cfg["obs"]["pc"]["use_pseudo_pc"] = True
        elif args.pc == "cam":
            cfg["env"]["enableCameraSensors"] = True
        else:
            print("Do not know which point cloud to be used")
            exit(123)
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["backbone_type"] = args.backbone
    else:
        print("Do not know the observation type")
        exit(123)

    cfg["task"]["pregrasp_type"] = args.pregrasp
    if args.pregrasp == "part":
        cfg["task"]["use_part_pregrasp"] = True
        cfg["env"]["usePreGraspPose"] = True
    elif args.pregrasp == "handle":
        cfg["task"]["use_part_pregrasp"] = False
        cfg["env"]["usePreGraspPose"] = True
    elif args.pregrasp == "None":
        cfg["env"]["usePreGraspPose"] = False
    else:
        print("Not know whether to use pregrasp pose, use --pregrasp to indicate part or handle or None")
        exit(123)

    if args.save_name != "NONE":
        cfg["log"]["save_name"] = args.save_name
    if args.group_name != "NONE":
        cfg["log"]["group_name"] = args.group_name

    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    if args.disc_feature is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            if args.disc_feature == 1:
                cfg_algo["discriminator"]["use_feature"] = True
            else:
                cfg_algo["discriminator"]["use_feature"] = False
    
    if args.ckpt is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["ckpt"] = args.ckpt

    if args.disc_obs is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            if args.disc_obs == 1:
                cfg_algo["discriminator"]["use_gt_obs"] = True
            else:
                cfg_algo["discriminator"]["use_gt_obs"] = False

    if args.expert_model is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            if args.expert_model == 1:
                cfg_algo["policy"]["use_expert"] = True
            else:
                cfg_algo["policy"]["use_expert"] = False

    if args.nsteps is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["learn"]["nsteps"] = args.nsteps

    if args.disc_reward_perportion is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["disc_reward_perportion"] = args.disc_reward_perportion

    if args.freeze_backbone:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["Spconv"]["freeze"] = True
            cfg_algo["policy"]["freeze"] = True
    else:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["Spconv"]["freeze"] = False
            cfg_algo["policy"]["freeze"] = False

    if args.self_imitation:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["use_self_imitation"]= True
    
    if args.use_seg:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["use_seg"]= True

    if args.concat_part_center:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["concat_part_center"]= True
            cfg_algo["policy"]["Spconv"]["in_channels"] += 3

    if args.use_part_pregrasp:
        cfg["task"]["use_part_pregrasp"] = True

    if args.sac_bc is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            assert(cfg_algo["learn"]["batch_size"] is not None)
            cfg_algo["learn"]["batch_size"] = args.sac_bc 

    if args.disc_bc is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["discriminator"]["disc_bc"] = args.disc_bc 

    if args.sac_buffer is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["learn"]["replay_size"] = args.sac_buffer 

    if args.max_batch is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["max_batch"] = args.max_batch

    if args.dagger_loss_supervise is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["dagger_loss_supervise"] = args.dagger_loss_supervise
    
    if args.dagger_loss_RL is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["dagger_loss_RL"] = args.dagger_loss_RL
        
    if args.rl_backbone_freeze is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["rl_backbone_freeze"] = args.rl_backbone_freeze
        
    if args.dagger_loss_seg is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["dagger_loss_seg"] = args.dagger_loss_seg
        
    if args.dagger_loss_decay_proportion is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["dagger_loss_decay_proportion"] = args.dagger_loss_decay_proportion
    
    if args.disc_update_freq is not None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["discriminator"]["update_freq"] = args.disc_update_freq
    
    if args.algo in ["heuristics"]:
        pass
    else:
        cfg_algo["policy"]["save_obs_path"] = args.save_obs_path

    if args.use_asset_with_demo:
        cfg["env"]["use_asset_with_demo"] = args.use_asset_with_demo
    if args.use_discriminator: 
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["policy"]["use_discriminator"] = args.use_discriminator
    if args.expert_buffer_size != None:
        if args.algo in ["heuristics"]:
            pass
        else:
            cfg_algo["expert_policy"]["expert_buffer_size"] = args.expert_buffer_size
    if args.action_normalization != None:
        cfg["env"]["action_normalization"] = args.action_normalization
        if args.action_normalization == "clip":
            if args.algo in ["heuristics"]:
                pass
            else:
                cfg_algo["expert_policy"]["action_clip"] = True
                cfg_algo["expert_policy"]["action_normalization"] = None
                cfg_algo["policy"]["action_clip"] = True
                cfg_algo["policy"]["action_normalization"] = None
        elif args.action_normalization == "tanh":
            if args.algo in ["heuristics"]:
                pass
            else:
                cfg_algo["expert_policy"]["action_normalization"] = args.action_normalization
                cfg_algo["policy"]["action_normalization"] = args.action_normalization
        elif args.action_normalization == " None":
            if args.algo in ["heuristics"]:
                pass
            else:
                cfg_algo["expert_policy"]["action_clip"] = False
                cfg_algo["expert_policy"]["action_normalization"] = None
                cfg_algo["policy"]["action_clip"] = False
                cfg_algo["policy"]["action_normalization"] = None
            
        else:
            raise NotImplementedError
    if args.bc_pretrain is not None:
        assert(args.bc_pretrain in ["all", "backbone", "None"])
        if args.algo in ["heuristics"]:
            pass
        else:
            if args.bc_pretrain != "None":
                cfg_algo["policy"]["use_bc_pretrain"] = True
            else:
                cfg_algo["policy"]["use_bc_pretrain"] = False
                
    if args.joint_training is not None:
        if args.joint_training == "None":
            if args.algo in ["heuristics"]:
                pass
            else:
                cfg_algo["policy"]["joint_update"] = False
        else:
            if args.algo in ["heuristics"]:
                pass
            else:
                cfg_algo["policy"]["joint_update"] = True
                cfg_algo["policy"]["update_which"] = args.joint_training
    if args.new_setting:
        cfg["env"]["asset"]["new_setting"] = True
    if args.save_video:
        cfg["save_video"] = True
        cfg["env"]["enableCameraSensors"] = True
    if args.use_general_camera:
        cfg["use_general_camera"] = True

    # import datetime

    # currtime = "{0:%Y-%m-%d_%H:%M:%S}".format(datetime.datetime.now())
    # if cfg["obs"]["use_pc"]:
    #     type_ = "pc"
    # else:
    #     type_ = "state"
    # if cfg["obs"]["pc"]["use_pseudo_pc"]:
    #     pc = "pseudo"
    # else:
    #     pc= "cam"
    # grasp = "G-" + cfg["task"]["pregrasp_type"]
    # if cfg["obs"]["state"]["use_cabinet_bbox"]:
    #     bbox = "B-" + cfg["obs"]["state"]["use_bbox_type"]
    # else:
    #     bbox = "Nobbox"
    # if args.algo in ["heuristics"]:
    #     nsteps = backbone = ""
    # else:
    #     backbone = cfg_algo["policy"]["backbone_type"]
    #     nsteps = cfg_algo["learn"]["nsteps"]
    # step = f"N-{nsteps}"
    # envs = str(cfg["env"]["AssetNumTrain"]) + "_" + str(cfg["env"]["AssetNumValIntra"]) + "_" + str(cfg["env"]["AssetNumValInter"]) + "_" + str(cfg["env"]["EnvPerAsset"])
    # obs ="" 
    # if cfg["obs"]["state"]["use_robot_qpose"]: obs += "T" 
    # else: obs += "F"
    # if cfg["obs"]["state"]["use_cabinet_qpose"]:obs += "T"
    # else: obs += "F"
    # if cfg["obs"]["state"]["use_cabinet_bbox"]:obs += "T"
    # else: obs += "F"
    # if cfg["obs"]["state"]["use_hand_root_pose"]:obs += "T"
    # else: obs += "F"
    # if cfg["obs"]["state"]["use_part_center"]:obs += "T"
    # else: obs += "F"
    # if cfg["obs"]["pc"]["numPCObservations"]==7:obs += "T"
    # else: obs += "F"
    
    # dirveMode = cfg["env"]["driveMode"]
    # if args.algo in ["heuristics"]:
    #     lr = ""
    # else:
    #     lr = str(cfg_algo["policy"]["dagger_loss_supervise"]) + "_" + str(cfg_algo["policy"]["dagger_loss_seg"]) + "_" + str(cfg_algo["policy"]["dagger_loss_RL"]) + "_" + str(cfg_algo["policy"]["dagger_loss_decay_proportion"]) + "_"

    # setting = ""
    # if args.algo in ["heuristics"]:
    #     pass
    # else:
    #     if cfg_algo["policy"]["use_dagger_activation"]: setting += "Act_"
    #     else: setting += "Noa_"
    #     if cfg_algo["policy"]["actor_freeze"]: setting += "Fre_"
    #     else: setting += "Nof_"
    


    # pc_num = str(cfg["obs"]["pc"]["pcDownSampleNum"])

    # if args.algo in ["heuristics"]:
    #     freeze = ""
    # else:
    #     if cfg_algo["policy"]["freeze"]:
    #         freeze = "freeze"
    #     else:
    #         freeze = "free"
    # cfg["env"]["env_name"] =  cfg["log"]["save_name"] + "_" + cfg["task"]["target_part_category"] + "_" + type_ +"_"+ pc +"_"
    # if backbone is not None and backbone is not "None":
    #     cfg["env"]["env_name"] += backbone + "_"
    # cfg["env"]["env_name"] += dirveMode + "_"+ setting + lr + grasp +"_"+ bbox +"_"+step+"_"+envs + "_"+obs + "_" +pc_num + "_" +freeze
    
    # cfg["log"]["group_name"] = cfg_algo["algo"]+"_"+cfg["log"]["group_name"]+"_"+cfg["env"]["env_name"]
    # cfg["env"]["env_name"] += "_" + currtime
    # cfg["log"]["save_name"] = cfg["env"]["env_name"]
    # save_name =  cfg["log"]["save_name"]
    # group_name =  cfg["log"]["group_name"]
    save_name, group_name = parse_name(args, cfg, cfg_algo)
    logdir = args.logdir
    log_id = logdir+f"/{args.task}_{args.algo}/{group_name}/{save_name}"
    logdir = os.path.realpath(log_id)
    
    if args.test :
        cfg_algo["learn"]["test"] = True
        cfg["log"]["use_wandb"] = False

    #print(cfg)
    #print(cfg_algo)
    return cfg, cfg_algo, logdir


def parse_sim_params_(cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    # sim_params.num_client_threads = cfg['slices']   #????
    sim_params.use_gpu_pipeline = (cfg['device_type'] != 'cpu')
    
    if cfg['physics_engine'] == 'flex':
        if cfg['device_type'] != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
        cfg['physics_engine'] = gymapi.SIM_FLEX
    elif cfg['physics_engine'] == 'physx':
        cfg_physx = cfg['sim']['physx']
        for k in cfg_physx.keys():
            setattr(sim_params.physx, k, cfg_physx[k])
        sim_params.physx.use_gpu = (cfg['device_type'] != 'cpu')
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        cfg['physics_engine'] = gymapi.SIM_PHYSX
    else:
        raise NotImplementedError

    return sim_params



def parse_sim_params(args, cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = sim_params_dt
    sim_params.num_client_threads = args.slices

    sim_params.physx.solver_type = sim_params_physx_solver_type
    sim_params.physx.num_position_iterations = sim_params_physx_num_position_iterations
    sim_params.physx.num_velocity_iterations = sim_params_physx_num_velocity_iterations
    sim_params.physx.num_threads = sim_params_physx_num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.num_subscenes = args.subscenes
    sim_params.physx.max_gpu_contact_pairs = sim_params_physx_max_gpu_contact_pairs
    sim_params.physx.rest_offset = sim_params_physx_rest_offset
    sim_params.physx.bounce_threshold_velocity = sim_params_physx_bounce_threshold_velocity
    sim_params.physx.max_depenetration_velocity = sim_params_physx_max_depenetration_velocity
    sim_params.physx.default_buffer_size_multiplier = sim_params_physx_default_buffer_size_multiplier
    sim_params.physx.contact_offset = sim_params_physx_contact_offset

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads
    
    return sim_params

# def parse_sim_params(args, cfg, cfg_train):
#     # initialize sim
#     sim_params = gymapi.SimParams()
#     sim_params.dt = sim_params_dt
#     sim_params.num_client_threads = args.slices

#     sim_params.physx.solver_type = sim_params_physx_solver_type
#     sim_params.physx.num_position_iterations = sim_params_physx_num_position_iterations
#     sim_params.physx.num_velocity_iterations = sim_params_physx_num_velocity_iterations
#     sim_params.physx.num_threads = sim_params_physx_num_threads
#     sim_params.physx.use_gpu = args.use_gpu
#     sim_params.physx.num_subscenes = args.subscenes
#     sim_params.physx.max_gpu_contact_pairs = sim_params_physx_max_gpu_contact_pairs
#     sim_params.physx.rest_offset = sim_params_physx_rest_offset
#     sim_params.physx.bounce_threshold_velocity = sim_params_physx_bounce_threshold_velocity
#     sim_params.physx.max_depenetration_velocity = sim_params_physx_max_depenetration_velocity
#     sim_params.physx.default_buffer_size_multiplier = sim_params_physx_default_buffer_size_multiplier
#     sim_params.physx.contact_offset = sim_params_physx_contact_offset

#     sim_params.use_gpu_pipeline = args.use_gpu_pipeline
#     sim_params.physx.use_gpu = args.use_gpu

#     # if sim options are provided in cfg, parse them and update/override above:
#     if "sim" in cfg:
#         gymutil.parse_sim_config(cfg["sim"], sim_params)

#     # Override num_threads if passed on the command line
#     if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
#         sim_params.physx.num_threads = args.num_threads
    
#     return sim_params

def get_args():
    custom_params = [
        # action
        {"name": "--test", "action": "store_true", "default": False, "help": "Test?"},
        {"name": "--headless", "action": "store_true", "default": False, "help": "viewer?"},
        {"name": "--use_camera", "action": "store_true", "default": False, "help": "use cam"},
        {"name": "--randomize", "action": "store_true", "default": False, "help": "phys dm rdm"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--use_soft_dist", "action": "store_true", "default": False,
            "help": "soft dense reward, defalt: False(training)"},
        {"name": "--use_finger_dist", "action": "store_true", "default": False,
            "help": "use finger center as dist dense reward defalt: False(training)"},
        {"name": "--use_easy_data", "action": "store_true", "default": False,
            "help": "use easy data(50 items), defalt: False(training)"},   
        {"name": "--use_qpose_state", "action": "store_true", "default": False,
            "help": "use qpose or not"},   
        {"name": "--gen_pc", "action": "store_true", "default": False,
            "help": "gen_pc"},   
        {"name": "--freeze_backbone", "action": "store_true", "default": False,
            "help": "freeze backbone or not"},
        {"name": "--self_imitation", "action": "store_true", "default": False,
            "help": "use self imitation in gail "}, 
        {"name": "--use_part_pregrasp", "action": "store_true", "default": False,
            "help": "pregrasp in front of part"},  
        {"name": "--debug", "action": "store_true", "default": False, "help": "debug mode"},  
        {"name": "--use_asset_with_demo", "action": "store_true", "default": False, "help": "use_asset_with_demo"},  
        {"name": "--use_discriminator", "action": "store_true", "default": False, "help": "use discrimitator for addtional reward"},  
        {"name": "--use_seg", "action": "store_true", "default": False, "help": "use_segmentation head for additional supervised learning gradient"},  
        {"name": "--use_dagger_activation", "action": "store_true", "default": False, "help": "use_dagger_activation"},  
        {"name": "--actor_freeze", "action": "store_true", "default": False, "help": "actor_freeze"},  
        {"name": "--rl_backbone_freeze", "action": "store_true", "default": False, "help": "rl_backbone_freeze"},  
        {"name": "--concat_part_center", "action": "store_true", "default": False, "help": "concat_part_center"},
        {"name": "--save_video", "action": "store_true", "default": False, "help": "save video"},
        {"name": "--use_general_camera", "action": "store_true", "default": False, "help": "general camera"},
        {"name": "--new_setting", "action": "store_true", "default": False, "help": "new_setting"},
        
        # str
        {"name": "--task", "type": str, "default": None, "help": "Task name"},
        {"name": "--algo", "type": str, "default": None, "help": "Algo name"},
        {"name": "--task_config", "type": str, "default": None, "help": "task config"},
        {"name": "--algo_config", "type": str, "default": None, "help": "algo config"},
        {"name": "--target", "type": str, "default": None, "help": "Target: open/grasp"},
        {"name": "--category", "type": str, "default": None, "help": "category: door/drawer"},
        {"name": "--save_name", "type": str, "default": "NONE", "help": "wandb save name"},
        {"name": "--group_name", "type": str, "defaut": "NONE", "help": "wandb group name"},

        {"name": "--device", "type": str, "default": "default", "help": "rl device"},
        {"name": "--single_data", "type": str, "default": None, "help": "a path for a single data"},
        {"name": "--ckpt", "type": str, "default": None, "help": "a path for checkpoint to be resumed"},
        {"name": "--control", "type": str, "default": None, "help": "control mode"},
        
        # {"name": "--graphics_device_id", "type": str, "default": "graphics_device_id", "help": "graphics_device_id"},
        # {"name": "--sim_device", "type": str, "default": "sim_device", "help": "sim_device_id"},

        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--model_dir", "type": str, "default": "","help": "Choose a model dir"},
        {"name": "--expert_dir", "type": str, "default": None,"help": "Choose a expert dir"},
        {"name": "--robot_name", "type": str, "default": None ,"help": "Choose a franka type"},
        {"name": "--canonical_space", "type":str, "default": None,
            "help": "canonical space: handle/part"}, 
        {"name": "--use_first_bbox", "type":str, "default": None,
            "help": "first_bbox , pt/gt"}, 
        {"name": "--use_bbox_pos", "type": str, "default": None,
            "help": "bbox, gt/pt"}, 
        {"name": "--obs", "type": str, "default": "pc", "help": "state or pc"},
        {"name": "--pc", "type": str, "default": "pseudo", "help": "cam or pseudo"},
        {"name": "--backbone", "type": str, "default": None, "help": "pn or spconv"},
        {"name": "--pregrasp", "type": str, "default": None, "help": "part or handle or None"},
        {"name": "--save_obs_path", "type": str, "default": None, "help": "psave_obs_path"},
        {"name": "--action_normalization", "type": str, "default": None, "help": "action_normalization"},
        {"name": "--bc_pretrain", "type": str, "default": None, "help": "bc pretrain, all / backbone"},
        {"name": "--joint_training", "type": str, "default": None, "help": "all /backbone / None"},
        {"name": "--asset_num", "type": str, "default": "0_0_0_0", "help": "assets number"},

        # int
        {"name": "--seed", "type": int, "help": "Random seed"},

        {"name": "--asset_num_train", "type": int, "default": 0, "help": "Num assets"},
        {"name": "--asset_num_valIntra", "type": int, "default": 0, "help": "Num assets"},
        {"name": "--asset_num_valInter", "type": int, "default": 0, "help": "Num assets"},
        {"name": "--env_per_asset", "type": int, "default": 1, "help": "Num assets"},

        {"name": "--episode_length", "type": int, "default": 0, "help": "episode length"},
        {"name": "--eval_round", "type": int, "default": None, "help": "eval round"},  
        {"name": "--eval_freq", "type": int, "default": None, "help": "eval_freq"},  
        {"name": "--noptepochs", "type": int, "default": None, "help": "epoch"},
        {"name": "--minibatches", "type": int, "default": None, "help": "number of minibatchs"},
        {"name": "--sac_bc", "type": int, "default": None, 
                "help": "batch_size in sac, true batch = batch_size * train_env_num"},
        {"name": "--disc_bc", "type": int, "default": None, 
                "help": "batch_size in discriminator"},
        {"name": "--sac_buffer", "type": int, "default": None, 
                "help": "replay buffer_size in discriminator"},

        {"name": "--disc_update_freq", "type": int, "default": None, "help": "use pregrasp or not"},
        {"name": "--disc_feature", "type": int, "default": None, "help": "discriminator use features"},
        {"name": "--disc_obs", "type": int, "default": None, "help": "discriminator use GT observations"},
        {"name": "--expert_model", "type": int, "default": 1, "help": "use expert model directly or use expert demostrations"},
        {"name": "--nsteps", "type": int, "default": None, "help": "nsteps update once"}, 
        {"name": "--max_batch", "type": int, "default": None, "help": "max batchsize for backbone forward pass"},
        {"name": "--expert_buffer_size", "type": int, "default": None, "help": "max buffer size for the expert buffer"},
        
        #float
        {"name": "--rot_coef", "type": float, "default": None,
            "help": "rotation coefficient in tasks/reward.py"},
        {"name": "--handle_coef", "type": float, "default": None,
            "help": "rotation coefficient in tasks/reward.py"},
        {"name": "--vel_coef", "type": float, "default": None,
            "help": "velocity coefficient in tasks/reward.py"},
        {"name": "--dist_coef", "type": float, "default": None,
            "help": "distance coefficient in tasks/reward.py"},
        {"name": "--tip_coef", "type": float, "default": None,
            "help": "tip coefficient in tasks/reward.py"},
        {"name": "--dicouple_coef", "type": float, "default": None,
            "help": "tip coefficient in tasks/reward.py"},
        {"name": "--close_coef", "type": float, "default": None,
            "help": "tip coefficient in tasks/reward.py"},
        {"name": "--openProportion", "type": float, "default": None,
            "help": "success open proportion"},
        {"name": "--contact_offset", "type": float, "default": None,
            "help": "maximal contact distance"},
        {"name": "--desired_kl", "type": float, "default": None,
            "help": "kl target"},
        {"name": "--disc_reward_perportion", "type": float, "default": None,
            "help": "disc_reward perportion"},
        {"name": "--dagger_loss_supervise", "type": float, "default": None,
            "help": "disc_reward perportion"},
        {"name": "--dagger_loss_RL", "type": float, "default": None,
            "help": "rl perportion"},
        {"name": "--dagger_loss_seg", "type": float, "default": None,
            "help": "seg perportion"},
        {"name": "--dagger_loss_decay_proportion", "type": float, "default": None,
            "help": "disc_reward perportion"},
    ]
    args = gymutil.parse_arguments(
        description = "PoseOrientedPolicy",
        custom_parameters=custom_params
    )
    
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else "cpu"

    # import pdb
    # pdb.set_trace()
    # print(args)
    if args.test:
        args.train = False
    else:
        args.train = True

    if args.task not in task_names:
        warn_task_name()
    if args.algo not in algo_names:
        warn_algo_name()

    return args

def process_cfgs():
    description = "Isaac Gym RL"
    partial_parser = ArgumentParser(description=description)
    partial_parser.add_argument('--taskcfg', default='open_door')
    partial_parser.add_argument('--algocfg', default='ppo')
    partial_args, others = partial_parser.parse_known_args()
    
    # read configs     
    with open(pjoin(os.getcwd(), 'cfg/base_cfg.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    with open(pjoin(os.getcwd(), f"cfg/tasks/{partial_args.taskcfg}.yaml"), 'r') as f:
        cfg['task'] = yaml.load(f, Loader=yaml.SafeLoader)
    with open(pjoin(os.getcwd(), f"cfg/algos/{partial_args.algocfg}.yaml"), 'r') as f:
        cfg['algo'] = yaml.load(f, Loader=yaml.SafeLoader)
        
    # auto generated parsers from config
    # to change xx in base_cfg.yaml, use "--xx"
    # to change xx in {$ALGO}.yaml, use "--algo.xx"
    # to change xx in {$TASK}.yaml, use "--task.xx"
    parser = ArgumentParser(description=description)
    add_args(parser, cfg)
    args = parser.parse_args(others)
    args_dict = vars(args)

    # over-write cfg
    print('#'*50)
    print('#'*18, 'parser info', '#'*18)
    print(f'algo cfg: {partial_args.algocfg}, task cfg: {partial_args.taskcfg}')
    
    for k, v in args_dict.items():
        if v is not None:
            k_lst = k.split('.')
            in_place = cfg
            for kk in k_lst[:-1]:
                in_place = in_place[kk]
            print(f'overwrite {k} from {in_place[k_lst[-1]]} to {v}!')
            in_place[k_lst[-1]] = v 
    print('#'*50)
    
    # set device
    if cfg['device_type'] == 'cpu':
        cfg['device'] = 'cpu'
    else:
        cfg['device'] = cfg['device_type'] + ":" + str(cfg['device_id'])
    if cfg['graphics_device_id'] == -1:
        cfg['graphics_device_id'] = cfg['device_id']  
    
    sim_params = parse_sim_params(cfg) 
    
    ##################################
    cfg['algo_name'] = cfg['algo']['algo']
    cfg['task_name'] = cfg['task']['task']
    ##################################

    return cfg, sim_params
        
class Logger:
    def __init__(self, cfg, exp_name, task_name, algo_name):
        cfg_log = cfg['log']

        self.max_iterations = cfg['algo']['max_iterations']
        self.wandb_flag = False 
        self.tensorboard_flag = False  
        self.screen_flag = False 

        group_name = f"{task_name}_{algo_name}" if cfg_log['group'] is None else cfg_log['group']
        id_name = exp_name if cfg_log['id'] is None else cfg_log['id']

        # for algos to save model
        self.save_ckpt_dir = pjoin(cfg_log['log_root'], 'ckpts', group_name, id_name)
        self.save_pose_dir = pjoin(cfg_log['log_root'], 'scene_pose', group_name, id_name)
        self.save_video_dir = pjoin(cfg_log['log_root'], 'video', group_name, id_name)

        if 'wandb' in cfg_log['mode']:
            if cfg['resume']:
                wandb_resume = "allow"
            else:
                wandb_resume = None 
            cfg_wandb = cfg_log['wandb']
            wandb.init(
                dir=cfg_log['log_root'],
                project=cfg_wandb['project'],
                entity=cfg_wandb['entity'],
                group=group_name,
                id=id_name,
                config=cfg,         # record all cfgs
                sync_tensorboard=True,
                monitor_gym=True,   
                resume=wandb_resume, 
                )
            self.wandb_flag = True 
            print(f"save wandb log to {pjoin(cfg_log['log_root'], 'wandb')}")

        # if use wandb, no need to use tensorboard any more
        if 'tensorboard' in cfg_log['mode'] and 'wandb' not in cfg_log['mode']:
            tensorboard_dir = pjoin(cfg_log['log_root'], 'tensorboard', group_name, id_name)
            self.writer = SummaryWriter(log_dir=tensorboard_dir) 
            self.tensorboard_flag = True 
            print(f"save tensorboard log to {tensorboard_dir}")

        if 'screen' in cfg_log['mode']:
            self.pad = 35
            self.width = 80
            self.screen_flag = True 

    def info(self, record_dict, iteration):
        if self.wandb_flag:
            wandb.log(data=record_dict, step=iteration)

        if self.tensorboard_flag:
            for k,v in record_dict.items():
                self.writer.add_scalar(k, v, iteration)
        
        if self.screen_flag:
            log_str = '#' * self.width + '\n'
            log_str += f" \033[1m Learning iteration {iteration}/{self.max_iterations} \033[0m ".center(self.width, ' ') + '\n\n'
            for k,v in record_dict.items():
                log_str += f"{k:<{self.pad}}: {v:.6f} \n"
            print(log_str)
        return 

    def update_resume_path(self, resume_path):
        return pjoin(os.path.dirname(self.save_ckpt_dir), resume_path)
