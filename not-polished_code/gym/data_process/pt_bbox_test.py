import sys,os
from glob import glob
from pathlib import Path
# part = "door"
# split = "train"
fail = []
for part in ["door", "drawer"]:
    for split in ["train", "valIntra", "valInter"]:
        paths = glob(f"assets/{part}/{split}/*")
        # print(paths)
        
        for path in paths:
            import json
            name = path +"/perception_info.json"
            try:
                with open(name, "r") as read_content:
                    data = json.load(read_content)
                    print(data.keys())
                    print(data["part_iou"], data["handle_iou"])
            except:
                fail.append(path)
print(fail)
                

# python train.py --single_data=assets/door/train/StorageFurniture-40147-link_0-handle_0-joint_0-handlejoint_0  --task=FrankaPoseCabinetPC --task_config=cfg/test_pc.yaml --algo=ppo_pn --algo_config=cfg/algo/ppo_pn.yaml --rl_device=cuda:0 --sim_device=cuda:0 --graphics_device_id=0 --seed=9999 --group_name debug_for_all --save_name haoran_pc_open_test_camera --asset_num_train 1 --asset_num_valIntra 0 --asset_num_valInter 0 --env_per_asset 1 --enableCameraSensors