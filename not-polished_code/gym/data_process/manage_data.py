from avaliable_assets import data
import numpy as np
from pose_utils import get_urdf_mobility
import os
print(data)
num_ = 0

task_part = "drawer"
split = "train"
if task_part == "door":
    joint_type = "revolute"
elif task_part == "drawer":
    joint_type = "prismatic"


for category_name in data[task_part].keys():
    data_category = data[task_part][category_name]
    for i in data_category:
        id, cl, num = i
        # id = 44781
        print(id)
        urdf_path = f"/data2/haoran/data/SAPIEN/SAPIEN_v3/all_pre_modified/{id}/mobility_relabel.urdf"

        urdf_data = get_urdf_mobility(urdf_path)
        print(num , urdf_data)
        # exit(123)
        type = np.array(urdf_data["joint"]["type"])
        # joint_type = "revolute"
        childs = np.array(urdf_data["joint"]["child"])
        # print(type == joint_type)
        # print(urdf_data["joint"]["child"])
        # print()
       
        link_ids = childs[type == joint_type]
        assert(num == len(link_ids))
        for j in range(num):
            
            try:
                link_id = link_ids[j]
                num_ += 1
                path_src =  f'/data2/haoran/data/SAPIEN/SAPIEN_v3/all_used_modified_unordered/{id}'
                path_dest = f'/data2/haoran/RL-Pose/PoseOrientedGym/assets/{task_part}/{split}/{category_name}_{id}_link{link_id}'
                os.system(f"cp -r {path_src} {path_dest}")
        
            except:
                print("???", id)
        
            # exit(123)
        # exit(123)

print(num_)