# drawer
python gym/train.py --task=FrankaPoseCabinetBase --algo=pregrasp_ppo \
--task_config=cfg/task_cfg.yaml --algo_config=cfg/algo/pregrasp_ppo.yaml \
--sim_device=cuda:0 --graphics_device_id=0 --seed=0526 --headless \
--group_name state --save_name 0.2_2_0_1_0_2 \
--rot_coef 0.2 --handle_coef 2 --vel_coef 0 --dist_coef 1 --tip_coef 0 --dicouple_coef 2 \
--target open --category drawer --minibatches 2 --nsteps 5 --noptepochs 8 \
--canonical_space handle --pregrasp part --obs state --eval_round 3 --eval_freq 10 \
--action_normalization clip --control ik_abs_axis_angle --asset_num 200_40_40_5 \
--test --ckpt xxxx.ckpt

# door
python gym/train.py --task=FrankaPoseCabinetBase --algo=pregrasp_ppo \
--task_config=cfg/task_cfg.yaml --algo_config=cfg/algo/pregrasp_ppo.yaml \
--sim_device=cuda:0 --graphics_device_id=0 --seed=0526 --headless \
--group_name state --save_name 0.2_2_0_1_0_2 \
--rot_coef 0.2 --handle_coef 2 --vel_coef 0 --dist_coef 1 --tip_coef 0 --dicouple_coef 2 \
--target open --category door --minibatches 2 --nsteps 5 --noptepochs 8 \
--canonical_space handle --pregrasp part --obs state --eval_round 3 --eval_freq 10 \
--action_normalization clip --control ik_abs_axis_angle --asset_num 2_2_2_5 \
--test --ckpt xxxx.ckpt