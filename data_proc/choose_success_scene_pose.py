import numpy as np
import trimesh
from os.path import join as pjoin

# saved pose path. each npy files is collected in each time step, e.g. 100.npy is collected in 100 step.
pose_path = './logs/scene_pose/grasp_cube_ppo/debug_seed6786/Iter40000/100.npy'
pose_dict = np.load(pose_path, allow_pickle=True).item()

# each npy files contain 2048 scene collected by paralleled environments
print(pose_dict.keys())
for k in pose_dict.keys():
    print(k, pose_dict[k].shape)

succ_num = []
for env in range(pose_dict['success'].shape[0]):
    if pose_dict['success'][env]:
        succ_num.append(env)
print(succ_num[:100])
print(len(succ_num))
