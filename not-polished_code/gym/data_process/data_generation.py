import os,sys
from glob import glob
import torch
import xml.etree.ElementTree as ET
import numpy as np
import os
import json
import random
#/data2/haoran/data/SAPIEN/SAPIEN_v3/all_used_modified/seen_category-CoffeeMachine-seen_instance-103031/mobility_new.urdf
data_root = "/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified/"
all_categorys = ["Box", "Remote", "Microwave", "Camera", "Dishwasher", "WashingMachine", "CoffeeMachine",
"Toaster", "StorageFurniture", "KitchenPot", "Safe", "Oven", "Phone", "Refrigerator", "Table", "TrashCan"]
categorys = ["CoffeeMachine", "Box", "KitchenPot", "TrashCan", ]
# categorys = ["Remote", "Phone", "WashingMachine", "Microwave",]
all_available = {}
all_available_num = {}
for part_id in range(4):
    if part_id != 4:
        continue
    part_name = ["door", "drawer", "button", "lid", "handle"][part_id]
    part_type = [["revolute"],[ "prismatic"], ["prismatic"], ["revolute","prismatic"], ["fixed"]][part_id]
    all_available[part_name] = {}
    all_available_num[part_name] = {}
    for split_id in range(3):
        split = ["train", "valIntra", "valInter"][split_id]
        sc = ["seen_category","seen_category","unseen_category"][split_id]
        si = ["seen_instance", "unseen_instance", "unseen_instance"][split_id]
        all_available[part_name][split] = {}
        all_available_num[part_name][split] = {}
        for category_id in range(len(categorys)):
            category = categorys[category_id]
            paths = sorted(glob(data_root + f"{sc}-{category}-{si}-*/mobility_new.urdf"))
            # print(paths)
            available = []
            for path in paths:
                id = path.split("/")[-2].split("-")[-1]
                if not os.path.exists(f"/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/{id}/bbox_info.json"):
                    # print(path)
                    continue
                print(id)
                # with open(f"/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/{id}/bbox_info.json", "r") as f:
                #     bbox_info = json.load(f)
                # print(bbox_info)
                # if "fixed" in bbox_info["parent_joint_name"]:
                #     exit(123)
                # continue
                # exit(123)
                # for i__ in range(len(bbox_info[""]))
                tree_urdf = ET.parse(path)
                num_real_links = len(tree_urdf.findall('link'))
                root_urdf = tree_urdf.getroot()
                # print(num_real_links)
                links = list(root_urdf.iter('link'))
                link_names = [link.attrib['name'] for link in links]
                joints = list(root_urdf.iter('joint'))
                joint_names = [joint.attrib["name"] for joint in joints]
                joint_childs = [link.attrib["link"] for joint in joints for link in list(joint.iter('child'))]
                joint_parents = [link.attrib["link"] for joint in joints for link in list(joint.iter('parent'))]
                joint_types = [joint.attrib["type"] for joint in joints]
                # print(link_names)
                # print(joint_names)
                # print(joint_childs)
                # print(joint_parents)
                # print(joint_types)
                # joint_names = np.array(joint_names)
                # joint_childs = np.array(joint_childs)
                # joint_parent = np.array(joint_parents)
                ids = []
                parent_joint = []
                for i_,child in enumerate(joint_childs):
                    if part_id <= 1:
                        if child.find("handle") != -1:
                            parent_link_name = joint_parents[i_]
                            parent_joint_id = joint_childs.index(parent_link_name)
                            if joint_types[parent_joint_id] in part_type :
                                ids.append(i_)
                                parent_joint.append(joint_names[parent_joint_id])
                    if part_id == 2:
                        # parent_link_name = joint_parents[i_]
                        # parent_joint_id = joint_childs.index(parent_link_name)
                        if joint_types[i_] in part_type:
                            ids.append(i_)
                    if part_id == 3:
                        # parent_link_name = joint_parents[i_]
                        # parent_joint_id = joint_childs.index(parent_link_name)
                        if joint_types[i_] in part_type:
                            ids.append(i_)
                    if part_id == 4:
                        # parent_link_name = joint_parents[i_]
                        # parent_joint_id = joint_childs.index(parent_link_name)
                        if joint_types[i_] in part_type:
                            import pdb
                            pdb.set_trace()
                            ids.append(i_)
                            # parent_joint.append(joint_names[parent_joint_id])
                    # import pdb
                    # pdb.set_trace()

                # print(ids)
                if len(ids) >= 5:
                    ids = np.array(ids)[np.random.choice(len(ids), 5, replace=False)]
                for _,i in enumerate(ids):
                    if _ >=5:
                        continue
                    import os
                    os.makedirs(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/{part_name}_/{split}", exist_ok=True)
                    src = "/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/" + f"{id}/"
                    dest = f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/{part_name}_/{split}/{category}-{id}-{joint_parents[i]}-{joint_childs[i]}-NONE-{joint_names[i]}"
                    # if not os.path.exists(f"/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/{id}/bbox_info.json"):
                    #     continue

                    # print(f"/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/{id}/bounding_box.json")
                    # exit(123)
                    print(dest)
                    # import pdb
                    # pdb.set_trace()
                    available.append(dest.split("/")[-1])
                    cmd = f"cp -r {src} {dest}"
                    # os.system(cmd)
                # print(available)
            
            all_available[part_name][split][category] = available
            all_available_num[part_name][split][category] = len(available)
print(all_available)
print(all_available_num)
import json
with open("/data2/haoran/RL-Pose/PoseOrientedGym/assets/avail.json", 'w', encoding='utf-8') as f: 
    json.dump(all_available, f)
                
                
# /data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified/unseen_category-KitchenPot-unseen_instance-100033
# /data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified/unseen_category-KitchenPot-unseen_instance-100056