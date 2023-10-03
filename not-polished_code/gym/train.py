from gc import freeze
import os
# os.environ['MESA_VK_DEVICE_SELECT'] = '10de:1e04' 
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7' 

import isaacgym
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym import gymapi
from utils.parse_task import load_env
from utils.gym_info import task_names, algo_names, no_training_algos
from utils.config import get_args, parse_sim_params, set_np_formatting, load_cfg, set_seed
from utils.parse_algo import process_pregrasp_ppo, process_dagger_ppo
import datetime
import wandb



def main():
    # prepare
    set_np_formatting()
    args = get_args()
    cfg, cfg_algo, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg)
    set_seed(cfg_algo.get("seed", -1), cfg_algo.get("torch_deterministic", False))
    cfg["env"]["env_name"] += "_" + str(cfg_algo["seed"])
    print("------------------ log saving information -------------------")
    print(cfg["log"]["group_name"],"$$$", cfg["env"]["env_name"])
    print("------------------------------------------------------------")
    if args.algo in algo_names:
        if cfg["log"]["use_wandb"]:
            # wandb.init(
            #     # project="pose-rl",
            #     # group=cfg["log"]["group_name"],
            #     # entity="pose-rl",
            #     # config=cfg_algo,
            #     name=cfg["log"]["save_name"],
            #     )
            wandb.init(
                project="pose-rl",
                group=cfg["log"]["group_name"],
                entity="pose-rl",
                config=cfg_algo,
                sync_tensorboard=True,
                name=cfg["log"]["save_name"],
                resume="allow",
                # monitor_gym=True,
            )
            wandb_writer = wandb
        else:
            wandb_writer = None
        task, env = load_env(args, cfg, cfg_algo, sim_params, logdir)
        
        

        poserl = eval('process_{}'.format(args.algo))(args, env, cfg, cfg_algo, logdir, wandb_writer)
        if args.algo in no_training_algos:
            poserl.run(num_eval_iterations=cfg_algo["eval"]["iterations"])
        else:
            poserl.run(num_learning_iterations=cfg_algo["learn"]["max_iterations"], 
                log_interval=cfg_algo["learn"]["save_interval"])
    
    else:
        print("Unrecognized algorithm!")
    # print(task, env)

    # print(cfg)
    # print(cfg_algo)


if __name__ == "__main__":
    main()
    
# def set_seed(seed, torch_deterministic, exp_name, resume):
#     if 'seed' in exp_name:
#         seed = int(exp_name.split('seed')[-1])
#     elif resume is not None: 
#         seed = int(resume.split('/')[-2].split('seed')[-1])
#     elif not torch_deterministic:
#         seed = np.random.randint(0, 10000)
    
#     if 'seed' not in exp_name:
#         exp_name = exp_name + f'_seed{seed}'

#     print("Setting seed: {}".format(seed))

#     # NOTE: even with the following codes, I still can't reproduce the results exactly
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     if torch_deterministic:
#         # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
#         os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#         torch.set_deterministic(True)
#     else:
#         torch.backends.cudnn.benchmark = True
#         torch.backends.cudnn.deterministic = False

#     return seed, exp_name

# def main():
#     # prepare
#     set_np_formatting()
#     cfg, sim_params = process_cfgs()
#     import pdb
    
#     pdb.set_trace()
#     cfg['seed'], cfg['exp_name'] = set_seed(cfg['seed'], cfg['torch_deterministic'], cfg['exp_name'], cfg['resume']) 
#     logger = Logger(cfg=cfg, exp_name=cfg['exp_name'], task_name=cfg['task_name'], algo_name=cfg['algo_name'])
#     if cfg['resume'] is not None:
#         cfg['algo']['resume'] = cfg['resume'] = logger.update_resume_path(cfg['resume'])

#     import pdb
#     pdb.set_trace()
#     # create env
#     env = eval(cfg['task_name'])(cfg=cfg['task'], base_cfg=cfg, sim_params=sim_params)
#     # run algorithms
#     runner = eval(cfg['algo_name'])(env, cfg['algo'], logger, base_cfg=cfg)
    
#     runner.run()
    
#     return 