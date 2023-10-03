import logging
logging.disable(logging.CRITICAL)

from tpi.core.config import cfg
from tabulate import tabulate
from mjrl.utils.make_train_plots import make_train_plots
#from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.trajectory_sampler import sample_paths_parallel
import numpy as np
import pickle
import time as timer
import os
import shutil
import copy

def train_agent(job_name, agent,
                seed = 0,
                niter = 101,
                gamma = 0.995,
                gae_lambda = None,
                num_cpu = 1,
                sample_mode = 'trajectories',
                num_traj = 50,
                num_samples = 50000, # has precedence, used with sample_mode = 'samples'
                save_freq = 10,
                evaluation_rollouts = None,
                plot_keys = ['stoc_pol_mean'],
                **kwargs,
                ):

    np.random.seed(seed)
    job_dir = os.path.join(cfg.JOB_DIR, job_name)
    os.makedirs(job_dir, exist_ok=True)
    cfg_file = os.path.join(job_dir, 'config.yaml')
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)
    previous_dir = os.getcwd()
    #print(f'previous dir: {previous_dir}')
    poses = ['mug_poses.npy', 'shapenet_poses.npy', 'shapenet_mug_poses.npy',
            'shapenet_remote_poses.npy', 'shapenet_can_poses.npy',
            'shapenet_bottle_poses.npy', 'shapenet_camera_poses.npy']
   
    for posefile in poses:
        if os.path.isfile(posefile):
            shutil.copyfile(posefile, f'{job_dir}/{posefile}')

    #if os.path.isfile('shapenet_poses.npy'):
    #    shutil.copyfile('shapenet_poses.npy', f'{job_dir}/shapenet_poses.npy')

    
    os.chdir(job_dir) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')
    best_policy = copy.deepcopy(agent.policy)
    best_perf = -1e8
    train_curve = best_perf*np.ones(niter)
    mean_pol_perf = 0.0
    #e = GymEnv(agent.env.env_id)
    e = agent.env

    if cfg.INVDYN_ONPG_DUMP_INIT:
        pickle.dump(agent.policy, open('iterations/initial_policy.pickle', 'wb'))

    for i in range(niter):
        print("......................................................................................")
        print("ITERATION : %i " % i)
        if train_curve[i-1] > best_perf:
            best_policy = copy.deepcopy(agent.policy)
            best_perf = train_curve[i-1]
        N = num_traj if sample_mode == 'trajectories' else num_samples
        args = dict(N=N, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu)
        stats = agent.train_step(**args)
        train_curve[i] = stats[0]
        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths_parallel(N=evaluation_rollouts, policy=agent.policy, num_cpu=num_cpu,
                                               env_name=e.env_id, mode='evaluation', pegasus_seed=seed, **kwargs)
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)
        # if i % save_freq == 0 and i > 0:
        #     if agent.save_logs:
        #         agent.logger.save_log('logs/')
        #         make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
        #     policy_file = 'policy_%i.pickle' % i
        #     baseline_file = 'baseline_%i.pickle' % i
        #     inverse_file = 'inversedyn_%i.pickle' % i
        #     pickle.dump(agent.policy, open('iterations/' + policy_file, 'wb'))
        #     pickle.dump(agent.baseline, open('iterations/' + baseline_file, 'wb'))
        #     pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
        #     try:
        #         pickle.dump(agent.inverse_model, open('iterations/' + inverse_file, 'wb'))
        #     except:
        #         pass
        #     if cfg.USE_INVDYN_ONPG:
        #         for k in range(cfg.INVDYN_ONPG_ENS_N):
        #             invdyn_file = 'invdyn_m{}_{}.pickle'.format(k, i)
        #             invdyn_model_data = {
        #                 'model_state': agent.invdyn_models[k].state_dict(),
        #                 'cfg': cfg.dump()
        #             }
        #             if cfg.INVDYN_ONPG_NORM:
        #                 invdyn_model_data['norm_stats'] = agent.get_norm_stats()
        #             pickle.dump(invdyn_model_data, open('iterations/' + invdyn_file, 'wb'))

        # print results to console
        if i == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        print("[ %s ] %4i %5.2f %5.2f %5.2f " % (timer.asctime(timer.localtime(timer.time())),
                                                 i, train_curve[i], mean_pol_perf, best_perf))
        result_file = open('results.txt', 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (i, train_curve[i], mean_pol_perf, best_perf))
        result_file.close()
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       agent.logger.get_current_log().items()))
            print(tabulate(print_data))

    # final save
    pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
    if agent.save_logs:
        agent.logger.save_log('logs/')
        make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
    os.chdir(previous_dir)
