from isaacgym import gymapi

from collections import abc
import os, yaml
from os.path import join as pjoin
from argparse import ArgumentParser


def parse_sim_params(cfg):
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

def process_cfgs():
    description = "Isaac Gym RL"
    partial_parser = ArgumentParser(description=description)
    partial_parser.add_argument('--taskcfg', default='grasp_cube')
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
    print('#'*10, 'parser info')
    print(f'algo cfg: {partial_args.algocfg}, task cfg {partial_args.taskcfg}')
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

    # copy some cfg to cfg_task and cfg_algo
    task_keys = ['headless', 'physics_engine', 'device_id', 'graphics_device_id', 'device', 'save_video']
    algo_keys = ['resume', 'test_only', 'device', 'save_pose', 'save_video','pretrain']
    
    for k in task_keys:
        cfg['task'][k] = cfg[k]
    for k in algo_keys:
        cfg['algo'][k] = cfg[k]
    cfg['algo']['model']['clipAction'] = cfg['task']['clipActions']
    cfg['algo']['succ_value'] = cfg['task']['succ_value']
    cfg['task']['num_envs'] = cfg['algo']['num_envs']
    cfg['task']['learn_input_mode'] = cfg['algo']['obs_mode'] # input mode for learning.
    cfg['task']['add_proprio_obs'] = cfg['algo']['add_proprio_obs']
    cfg['algo_name'] = cfg['algo']['algo']
    cfg['task_name'] = cfg['task']['task']

    return cfg, sim_params
