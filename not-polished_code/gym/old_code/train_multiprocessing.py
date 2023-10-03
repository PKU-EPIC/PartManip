from multiprocessing import Process

from time import sleep
import os

def run(cmd):
    os.system(cmd)
    
def get_state_cmd(control_mode, seed, category, canonical, ckpt= None, cuda=0):
    if category == "door":
        open=0.53
    else:
        open=0.2
    cmd = f"python train.py \
        --task=FrankaPoseCabinetPC \
        --task_config=cfg/test_pc.yaml \
        --algo=pregrasp_ppo \
        --algo_config=cfg/algo/pregrasp_ppo_pn.yaml \
        --rl_device=cuda:{cuda} \
        --sim_device=cuda:{cuda} \
        --category {category} \
        --seed {seed} \
        --group_name state_formal_{category} \
        --save_name {category}_{control_mode}_{canonical}_{seed} \
        --asset_num_train 200 \
        --asset_num_valIntra 15 \
        --asset_num_valInter 15 \
        --env_per_asset 5 \
        --rot_coef 0.2 \
        --handle_coef 2 \
        --vel_coef 0 \
        --dist_coef 1 \
        --tip_coef 0 \
        --dicouple_coef 1 \
        --openProportion {open} \
        --minibatches 2 \
        --headless \
        --nsteps 5 \
        --backbone pn \
        --pc cam \
        --obs state \
        --canonical_space {canonical} \
        --pregrasp part \
        --eval_freq 20 \
        --eval_round 2 \
        --control {control_mode}"
    return cmd
    
def get_pc_cmd(pc_type, backbone_type, control_mode, ckpt):
    cmd = f"python train.py \
        --task=FrankaPoseCabinetPC \
        --task_config=cfg/test_pc.yaml \
        --algo=pregrasp_ppo_pn \
        --algo_config=cfg/algo/pregrasp_ppo_pn.yaml \
        --rl_device=cuda:0 \
        --sim_device=cuda:0  \
        --vision_device=cuda:0 \
        --graphics_device_id=0 \
        --seed=9999 \
        --group_name formal_1019 \
        --save_name drawer \
        --asset_num_train 20 \
        --asset_num_valIntra 10 \
        --asset_num_valInter 10 \
        --env_per_asset 5 \
        --rot_coef 0.2 \
        --handle_coef 2 \
        --vel_coef 0 \
        --dist_coef 2 \
        --tip_coef 0 \
        --dicouple_coef 1 \
        --openProportion 0.53 \
        --minibatches 2 \
        --headless \
        --nsteps 5 \
        --backbone {backbone_type} \
        --pc {pc_type} \
        --obs pc \
        --canonical_space handle \
        --pregrasp part  \
        --eval_freq 1 --control {control_mode} --ckpt {ckpt}"
    return cmd
    
if __name__ == "__main__":
    process_list = []

    #########################state base start#####################################
    p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=1412, category="drawer", canonical="None", cuda=4),))
    p.start()
    process_list.append(p)
    sleep(300)
    p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=526, category="drawer", canonical="None", cuda=6),))
    p.start()
    process_list.append(p)
    sleep(300)
    p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=1234, category="drawer", canonical="None", cuda=6),))
    p.start()
    process_list.append(p)
    sleep(300)
    p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=77, category="drawer", canonical="None", cuda=7),))
    p.start()
    process_list.append(p)
    sleep(300)
    p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=4321, category="drawer", canonical="None", cuda=7),))
    p.start()
    process_list.append(p)
    sleep(300)

    # p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=1412, category="door", canonical="None", cuda=4),))
    # p.start()
    # process_list.append(p)
    # sleep(100)
    # p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=526, category="door", canonical="None", cuda=6),))
    # p.start()
    # process_list.append(p)
    # sleep(100)
    # p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=1234, category="door", canonical="None", cuda=6),))
    # p.start()
    # process_list.append(p)
    # sleep(100)
    # p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=77, category="door", canonical="None", cuda=7),))
    # p.start()
    # process_list.append(p)
    # sleep(100)
    # p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=4321, category="door", canonical="None", cuda=7),))
    # p.start()
    # process_list.append(p)
    # sleep(100)

    p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=1412, category="drawer", canonical="handle", cuda=2),))
    p.start()
    process_list.append(p)
    sleep(300)
    p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=526, category="drawer", canonical="handle", cuda=2),))
    p.start()
    process_list.append(p)
    sleep(300)
    p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=1234, category="drawer", canonical="handle", cuda=3),))
    p.start()
    process_list.append(p)
    sleep(300)
    p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=77, category="drawer", canonical="handle", cuda=3),))
    p.start()
    process_list.append(p)
    sleep(300)
    p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=4321, category="drawer", canonical="handle", cuda=4),))
    p.start()
    process_list.append(p)
    sleep(300)

    # p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=1412, category="door", canonical="handle", cuda=2),))
    # p.start()
    # process_list.append(p)
    # sleep(100)
    # p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=526, category="door", canonical="handle", cuda=2),))
    # p.start()
    # process_list.append(p)
    # sleep(100)
    # p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=1234, category="door", canonical="handle", cuda=3),))
    # p.start()
    # process_list.append(p)
    # sleep(100)
    # p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=77, category="door", canonical="handle", cuda=3),))
    # p.start()
    # process_list.append(p)
    # sleep(100)
    # p = Process(target = run, args = (get_state_cmd(control_mode="pos", seed=4321, category="door", canonical="handle", cuda=4),))
    # p.start()
    # process_list.append(p)
    # sleep(100)
    
    ##################state end######################

    # p = Process(target = run, args = (get_state_cmd("maniskill", "/scratch/genghaoran/RL-Pose/PoseOrientedGym/logs/FrankaPoseCabinetPC_pregrasp_ppo_pn/pregrasp_ppo_pnformal/state_cam_None_grasp-part_bboxgt_nstep-10_200_20_20_10_TTTTTT_4000_free_maniskill_drawer_2022-10-18_10:30:10_algo-seed-1/model_308.tar"),))
    # p.start()
    # process_list.append(p)
    # sleep(300)
    
    # p = Process(target = run, args = (get_state_cmd("pos", "/scratch/genghaoran/RL-Pose/PoseOrientedGym/logs/FrankaPoseCabinetPC_pregrasp_ppo_pn/pregrasp_ppo_pnformal/state_cam_None_grasp-part_bboxgt_nstep-10_20_2_2_10_TTTTTT_4000_free_pos_drawer_2022-10-17_22:54:26_algo-seed-1/model_62.tar"),))
    # p.start()
    # process_list.append(p)
    # sleep(300)
    
    # p = Process(target = run, args = (get_state_cmd("ik_abs", "/scratch/genghaoran/RL-Pose/PoseOrientedGym/logs/FrankaPoseCabinetPC_pregrasp_ppo_pn/pregrasp_ppo_pnformal/state_cam_None_grasp-part_bboxgt_nstep-10_200_20_20_10_TTTTTT_4000_free_ik_abs_drawer_2022-10-18_10:28:59_algo-seed-1/model_275.tar"),))
    # p.start()
    # process_list.append(p)
    # sleep(300)
    
    # p = Process(target = run, args = (get_state_cmd("ik_rela", "/scratch/genghaoran/RL-Pose/PoseOrientedGym/logs/FrankaPoseCabinetPC_pregrasp_ppo_pn/pregrasp_ppo_pnformal/state_cam_None_grasp-part_bboxgt_nstep-10_200_20_20_10_TTTTTT_4000_free_ik_rela_drawer_2022-10-18_10:29:32_algo-seed-1/model_275.tar"),))
    # p.start()
    # process_list.append(p)
    # sleep(300)
    
    
    
    # p = Process(target = run, args = (get_pc_cmd("pseudo", "pn", "ik_abs","/scratch/genghaoran/RL-Pose/PoseOrientedGym/logs/FrankaPoseCabinetPC_pregrasp_ppo_pn/pregrasp_ppo_pnformal_1017/pc_pseudo_pn_grasp-handle_bboxgt_nstep-10_20_10_10_20_TTTTTT_4000_free_ik_abs_drawer_2022-10-18_10:32:44_algo-seed-1/model_90.tar"),))
    # p.start()
    # process_list.append(p)
    # sleep(300)
    
    # p = Process(target = run, args = (get_pc_cmd("pseudo", "pn", "pos","/scratch/genghaoran/RL-Pose/PoseOrientedGym/logs/FrankaPoseCabinetPC_pregrasp_ppo_pn/pregrasp_ppo_pnformal_1017/pc_pseudo_pn_grasp-handle_bboxgt_nstep-10_20_10_10_20_TTTTTT_4000_free_pos_drawer_2022-10-18_10:33:13_algo-seed-1/model_91.tar"),))
    # p.start()
    # process_list.append(p)
    # sleep(300)
    
    # p = Process(target = run, args = (get_pc_cmd("pseudo", "pn", "ik_abs","/scratch/genghaoran/RL-Pose/PoseOrientedGym/logs/FrankaPoseCabinetPC_pregrasp_ppo_pn/pregrasp_ppo_pnformal_1017/pc_pseudo_pn_grasp-part_bboxgt_nstep-10_20_10_10_15_TTTTTT_4000_free_ik_abs_drawer_2022-10-18_10:34:22_algo-seed-1/model_111.tar"),))
    # p.start()
    # process_list.append(p)
    # sleep(300)
    
    # p = Process(target = run, args = (get_pc_cmd("pseudo", "pn", "pos","/scratch/genghaoran/RL-Pose/PoseOrientedGym/logs/FrankaPoseCabinetPC_pregrasp_ppo_pn/pregrasp_ppo_pnformal_1017/pc_pseudo_pn_grasp-part_bboxgt_nstep-10_15_10_10_20_TTTTTT_4000_free_pos_drawer_2022-10-18_10:33:46_algo-seed-1/model_98.tar"),))
    # p.start()
    # process_list.append(p)
    # sleep(300)
    
    for p in process_list:
        p.join()
    print("finish")
    