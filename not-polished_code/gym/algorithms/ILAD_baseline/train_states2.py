#!/usr/bin/env python3

"""Train an agent from states."""

import argparse
import pickle
import sys
import numpy as np
import gym
import os

from mjrl.algos.behavior_cloning_2 import BC
from mjrl.algos.lfa import LFA
from mjrl.algos.trpo import TRPO
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.baselines.mlp_value import MLPValue
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.pc_mlp import PCMLP
from mjrl.utils.train_agent import train_agent
from tpi.core.config import assert_cfg
from tpi.core.config import cfg

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from envs.base.vec_task import VecTaskPythonArm, VecTaskPythonArmPC
from envs.franka_pose_cabinet_base import FrankaPoseCabinetBase
from envs.franka_pose_cabinet_campc import FrankaPoseCabinetPC

class Spec():
    def __init__(self, env=None, env_name="relocate-mug-1"):
        self.observation_dim = env.reset().shape[0]
        self.action_dim = env.action_spec[0].shape[0]
        self.env_id = env_name

        print(f'observation dim: {self.observation_dim}, action dim: {self.action_dim}')

def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train agent from states'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file',
        required=True,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See pycls/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def train(cfg):
    # Construct the env
    
    env_info = cfg.ENV_NAME.split('-')
    task = env_info[0]
    obj_name = env_info[1]
    obj_scale = float(env_info[2])
    friction = (1, 0.5, 0.01)

    # e = SHAPENETPCRelocate(has_renderer=False, obj_poses=f'./shapenet_{obj_name}_poses.npy',
    #                 friction=friction)
    e = VecTaskPythonArmPC(task, rl_device)
    spec = Spec(e, cfg.ENV_NAME)

    # add current position for object reference
    sys.path.append(os.getcwd())
    
    # Construct the policy
    #only use action / obs dim
    if cfg.USE_LFA:
         policy = PCMLP(
            spec, hidden_sizes=cfg.POLICY_WS, seed=cfg.RNG_SEED,
            init_log_std=cfg.POLICY_INIT_LOG_STD,
            min_log_std=cfg.POLICY_MIN_LOG_STD
        )
    else:
        policy = MLP(
        spec, hidden_sizes=cfg.POLICY_WS, seed=cfg.RNG_SEED,
        init_log_std=cfg.POLICY_INIT_LOG_STD,
        min_log_std=cfg.POLICY_MIN_LOG_STD
        )       

    # Load policy from checkpoint
    if cfg.CHECKPOINT_POLICY:
        with open(cfg.CHECKPOINT_POLICY, 'rb') as f:
            policy = pickle.load(f)

    # Load the demos
    demo_paths = None
    if (
        cfg.BC_INIT or cfg.USE_LFA
    ):
        with open(cfg.DEMO_FILE, 'rb') as f:
            demo_paths = pickle.load(f)
            num_traj = len(demo_paths)
            num_traj = int(min(cfg.DEMO_RATIO, num_traj))
            demo_paths = demo_paths[:num_traj]
            # print(f'num traj: {num_traj}')

    # Initialize w/ behavior cloning
    if cfg.BC_INIT:
        print('==== Training with BC ====')
        bc_agent = BC(demo_paths, policy=policy, epochs=10, batch_size=32, lr=1e-3, category=obj_name, \
                        object_scale=obj_scale)
        bc_agent.train(random_embedding=cfg.RANDOM_EMBEDDING)

    # Construct the baseline for PG
    #only use action / obs dim
    baseline = MLPBaseline(
        spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3, use_gpu=True
    )
    # Load baseline from checkpoint
    if cfg.CHECKPOINT_BASELINE:
        with open(cfg.CHECKPOINT_BASELINE, 'rb') as f:
            baseline = pickle.load(f)

    # Construct the agent
    if cfg.USE_LFA:
        value = MLPValue(spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3, use_gpu=True)
        agent = LFA(
            spec, policy, baseline, value,
            demo_paths=(demo_paths if cfg.USE_LFA else None),
            normalized_step_size=0.1, seed=cfg.RNG_SEED,
            lam_0=cfg.LFA_LAM0, lam_1=cfg.LFA_LAM1,
            lam_2=cfg.LFA_LAM2, lam_3=cfg.LFA_LAM3,
            use_eval_adv=cfg.LFA_EVAL_ADV, save_logs=True,
            dapg_baseline=cfg.DAPG_BASELINE,
            bc_finetune=cfg.BC_FINETUNE,
            bc_agent=bc_agent, priority=cfg.PRIORITY,
            ft_batchsize=cfg.FT_BATCHSIZE,
            ft_lr=cfg.FT_LR,
            ft_interval=cfg.FT_INTERVAL,
            pre_emb = None
        )
    else:
        print('using TRPO for RL algo')
        agent = TRPO(
                spec, policy, baseline,
                normalized_step_size=0.1,
                seed=cfg.RNG_SEED,
                save_logs=True)

    # Train the agent
    print('==== Training with RL ====')
        
    if cfg.USE_LFA:
        env_name = cfg.ENV_NAME
        task = env_name.split('-')[0]
        obj = env_name.split('-')[1]
        demo_property = cfg.DEMO_FILE.split('/')[-1]
        demo_property = demo_property.split('.pkl')[0]
        demo_property = demo_property.replace(f'_{obj}', '').replace(f'{task}_', '')

    job_name = f'{cfg.ENV_NAME}_seed{cfg.RNG_SEED}'

    train_agent(
        job_name=job_name,
        agent=agent,
        seed=cfg.RNG_SEED,
        niter=cfg.NUM_ITER,
        gamma=0.995,
        gae_lambda=0.97,
        num_cpu=1,
        sample_mode='trajectories',
        num_traj=cfg.NUM_TRAJ,
        save_freq=cfg.SAVE_FREQ,
        evaluation_rollouts=cfg.EVAL_RS,
        density=cfg.DENSITY
    )


if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg()
    cfg.freeze()
    train()
