from algorithms import ppo, dagger, bc
from tasks import grasp_cube, open_drawer
from utils import process_cfgs, Logger

import os
import numpy as np
import random
import torch


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)

def set_seed(seed, torch_deterministic, exp_name, resume):
    if 'seed' in exp_name:
        seed = int(exp_name.split('seed')[-1])
    elif resume is not None: 
        try:
            seed = int(resume.split('/')[-2].split('seed')[-1])
        except:
            seed = 1234
    elif not torch_deterministic and seed == -1:
        seed = np.random.randint(0, 10000)
    
    if 'seed' not in exp_name:
        exp_name = exp_name + f'_seed{seed}'

    print("Setting seed: {}".format(seed))

    # NOTE: even with the following codes, I still can't reproduce the results exactly
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed, exp_name

def main():
    set_np_formatting() 
    # load cfgs and also support auto-generated args to overwrite cfgs
    cfg, sim_params = process_cfgs()

    # set seed 
    cfg['seed'], cfg['exp_name'] = set_seed(cfg['seed'], cfg['torch_deterministic'], cfg['exp_name'], cfg['resume']) 

    # set logger
    logger = Logger(cfg=cfg, exp_name=cfg['exp_name'], task_name=cfg['task_name'], algo_name=cfg['algo_name'])
    if cfg['resume'] is not None:
        cfg['algo']['resume'] = cfg['resume'] = logger.update_resume_path(cfg['resume'])
    if cfg['pretrain'] is not None:
        cfg['algo']['pretrain'] = cfg['pretrain'] = logger.update_resume_path(cfg['pretrain'])

    # create env
    env = eval(cfg['task_name'])(cfg=cfg['task'], sim_params=sim_params)
    # run algorithms
    runner = eval(cfg['algo_name'])(env, cfg['algo'], logger)

    runner.run()

    return

if __name__ == '__main__':
    main()