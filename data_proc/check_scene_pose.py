import numpy as np
import trimesh
from os.path import join as pjoin

visual_path = []
# franka joints path
hand_base_path = '../assets/franka_description/meshes/visual'
for i in range(8):
    visual_path.append(f'link{i}.obj')
visual_path.extend(['hand.obj', 'finger.stl','finger.stl'])
for i in range(len(visual_path)):
    visual_path[i] = pjoin(hand_base_path, visual_path[i])

# obj path
# obj_path = '../assets/objs/bottle/1ef68777bfdb7d6ba7a07ee616e34cd7/model.obj'
# visual_path.append(obj_path)

# saved pose path. each npy files is collected in each time step, e.g. 100.npy is collected in 100 step.
pose_path = '<path to>/Iter40000/70.npy'
pose_dict = np.load(pose_path, allow_pickle=True).item()

# each npy files contain 2048 scene collected by paralleled environments
print(pose_dict.keys())
for k in pose_dict.keys():
    print(k, pose_dict[k].shape)

# visualize point clouds
env_ids = [0]
for env in env_ids:
    total_pc = []
    for i in range(len(visual_path)+1):
        pos = pose_dict['pos'][env,i]
        rot = pose_dict['rot'][env,i]
        if i == len(visual_path)-1 or i == len(visual_path)-2:
            print(rot)
        if i != len(visual_path):
            mesh = trimesh.load(visual_path[i], force='mesh')
            pc = trimesh.sample.sample_surface(mesh, 1024)[0]
        else:
        # object 
            pc = trimesh.sample.sample_surface_even(trimesh.creation.box(), 512)[0] *0.05
            
        transformed_pc = np.matmul(np.array(pc), rot.transpose(-1,-2)) + pos[None, :] 
        total_pc.append(transformed_pc)

    total_pc = np.concatenate(total_pc, axis=0)
    np.savetxt(f'debug_fk{env}.txt', total_pc)


