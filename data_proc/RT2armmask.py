import numpy as np
import trimesh
from os.path import join as pjoin
import pyrender
import cv2
import os 

os.environ['PYOPENGL_PLATFORM'] = 'egl'

scene_id = 0
step_id = 70
cam_id = 2
cam_path = f'/data1/DynamicRLNeRF/data/grasping_cube_train_0202_mitsuba_rand/scenes/scene_{str(scene_id).zfill(5)}/step_{str(step_id).zfill(5)}/camera_pose.npy'
pose_path = f'/home/jiayichen/RL-NeRF/NerfRL/logs/scene_pose/grasp_cube_ppo/debug_seed6786/Iter40000/{step_id}.npy'
hand_base_path = '../assets/franka_description/meshes/visual'


# camera 
scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02],
                        bg_color=[1.0, 1.0, 1.0])

import math 
horizontal_fov = 69.75
fx = 640 / 2.0 / math.tan((horizontal_fov / 180.0 * math.pi) / 2.0)


yfov = math.atan((360/2/fx)) * 2

print(fx, yfov)
camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=640/360)
cam_RT = np.load(cam_path)  # 3,4,4
scene.add(camera, pose=cam_RT[cam_id])


# hand path
visual_path = []
for i in range(8):
    visual_path.append(f'link{i}.obj')
visual_path.extend(['hand.obj', 'finger.stl','finger.stl'])
for i in range(len(visual_path)):
    visual_path[i] = pjoin(hand_base_path, visual_path[i])


# hand pose
pose_dict = np.load(pose_path, allow_pickle=True).item()
total_pc = []
seg_node_map = {}
for i in range(len(visual_path)):
    pos = pose_dict['pos'][scene_id, i]  # 3
    rot = pose_dict['rot'][scene_id, i]  # 3, 3
    pose = np.concatenate([rot, pos[:, None]], -1)  # 3, 4
    pose = np.concatenate([pose, np.zeros((1,4))], -2)    # 4, 4
    pose[-1,-1] = 1
    tm = trimesh.load(visual_path[i], force='mesh')
    tm.apply_transform(pose)
    m = pyrender.Mesh.from_trimesh(tm)
    nc = pyrender.Node(mesh=m, matrix=np.eye(4))
    scene.add_node(nc)
    seg_node_map[nc] = 1

    pc = trimesh.sample.sample_surface(tm, 1024)[0]
    transformed_pc = np.matmul(np.array(pc), rot.transpose(-1,-2)) + pos[None, :] 
    total_pc.append(transformed_pc)

total_pc = np.concatenate(total_pc, axis=0)
# np.savetxt(f'debug_pc.txt', total_pc)


r = pyrender.OffscreenRenderer(viewport_width=640,
                                viewport_height=360,
                                point_size=1.0)

seg, depth = r.render(scene, flags=pyrender.constants.RenderFlags.SEG, seg_node_map=seg_node_map)
# seg, depth = r.render(scene)
rgb = cv2.imread(f'/data1/DynamicRLNeRF/data/grasping_cube_train_0202_mitsuba_rand/scenes/scene_{str(scene_id).zfill(5)}/step_{str(step_id).zfill(5)}/rgb/{str(cam_id).zfill(4)}.png')
cv2.imwrite(f'{str(cam_id).zfill(4)}.jpg', seg)
cv2.imwrite('debug.jpg', seg*rgb)
# cv2.imwrite('debug_rgb.jpg', rgb)