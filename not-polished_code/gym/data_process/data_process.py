import pickle

from pose_utils import get_urdf_mobility
import torch
import numpy as np
import cv2
import open3d as o3d
import json
# from isaacgym.torch_utils import *
import numpy as np
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    eg: np.dot(rotation_matrix(axis, theta), v) 
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


HEIGHT = int(800)
WIDTH = int(800)
K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
                [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

def map2image(pts, rgb):
    # input为每个shape的info，取第idx行
    image_rgb = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    num_point = pts.shape[0]
    # print(num_point)
    # print(pts)
    # print(rgb.shape)

    point2image = {}
    for i in range(num_point):
        x = pts[i][0]
        y = pts[i][1]
        z = pts[i][2]
        x_new = (np.around(x * K[0][0] / z + K[0][2])).astype(dtype=int)
        y_new = (np.around(y * K[1][1] / z + K[1][2])).astype(dtype=int)
        point2image[i] = (y_new, x_new)

    # 还原原始的RGB图
    for i in range(num_point):
        # print(i, point2image[i][0], point2image[i][1])
        if point2image[i][0]+1 >= HEIGHT or point2image[i][0] < 0 or point2image[i][1]+1 >= WIDTH or point2image[i][1] < 0:
            continue
        image_rgb[point2image[i][0]][point2image[i][1]] = rgb[i]
        image_rgb[point2image[i][0]+1][point2image[i][1]] = rgb[i]
        image_rgb[point2image[i][0]+1][point2image[i][1]+1] = rgb[i]
        image_rgb[point2image[i][0]][point2image[i][1]+1] = rgb[i]

    # rgb_pil = Image.fromarray(image_rgb, mode='RGB')
    # rgb_pil.save(os.path.join(save_path, f'{instance_name}_{task}.png'))
    return image_rgb

def draw_line(pt1,pt2,img, color=(int(255),int(0),int(255))):
    x_new = (np.around(pt1[0] * K[0][0] / pt1[2] + K[0][2])).astype(dtype=int)
    y_new = (np.around(pt1[1] * K[1][1] / pt1[2] + K[1][2])).astype(dtype=int)
    x_new_ = (np.around(pt2[0] * K[0][0] / pt2[2] + K[0][2])).astype(dtype=int)
    y_new_ = (np.around(pt2[1] * K[1][1] / pt2[2] + K[1][2])).astype(dtype=int)
    cv2.line(img,[x_new,y_new], [x_new_, y_new_],color=(int(255),int(0),int(255)),thickness=2)

def draw_bbox(img, bbox_list, trans):
    for i,bbox in enumerate(bbox_list):
        if len(bbox) == 0:
            continue
        # bbox = bbox * trans[0]+trans[1:4]
        K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        point2image = []
        for pts in bbox:
            x = pts[0]
            y = pts[1]
            z = pts[2]
            x_new = (np.around(x * K[0][0] / z + K[0][2])).astype(dtype=int)
            y_new = (np.around(y * K[1][1] / z + K[1][2])).astype(dtype=int)
            point2image.append([x_new, y_new])
        cl = [255,0,0]
        cv2.line(img,point2image[0],point2image[1],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[1],color=(255,0,0),thickness=1)
        cv2.line(img,point2image[1],point2image[2],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[2],color=(0,255,0),thickness=1)
        cv2.line(img,point2image[2],point2image[3],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[3],color=(0,0,255),thickness=1)
        cv2.line(img,point2image[3],point2image[0],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[5],point2image[6],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[6],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[0],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[6],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[3],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
    return img

def bbox_process(bbox, joint_xyz, axis, joint_type, qpos, ):
    if joint_type == "revolute":
        return np.dot(rotation_matrix(axis, qpos), (bbox - joint_xyz).T).T + joint_xyz
    elif joint_type == "prismatic":
        return bbox - axis*qpos

if __name__ == "__main__":
    id = "45948"
    cate = "Remote"
    joint_type_id = 1
    joint_type = ["revolute", 'prismatic', "fixed"][joint_type_id]
    # joint_type = 'revolute'
    target_part = [['hinge_door', "hinge_knob", "hinge_lid"],["slider_drawer", "slider_button", "slider_lid"], ["line_fixed_handle","round_fixed_handle"]][joint_type_id]

    bbox_file_name = f"/data2/haoran/data/source_data/source_data_2/valid/annotation/bbox/{cate}_{id}_00_000.pkl"
    pc_file_name = f"/data2/haoran/data/PartDetection_data_all/train/pth/{cate}_{id}_00_000.pth"
    meta_file_name = f"/data2/haoran/data/PartDetection_data_all/train/meta/{cate}_{id}_00_000.txt"
    cammeta_file_name = f"/data2/haoran/data/source_data/source_data_2/valid/metafile/{cate}_{id}_00_000.json"
    urdf_file_name = f"/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/{id}/mobility_relabel.urdf"
    new_urdf_file_name = f"/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/{id}/mobility_new.urdf"
    with open(cammeta_file_name,"r") as f:
        cammeta = json.load(f)

    qpos = cammeta['joint_qpos']
    urdf_info = get_urdf_mobility(urdf_file_name)[0]
    import pdb
    pdb.set_trace()
    print(urdf_info["joint"])
    new_urdf_info = get_urdf_mobility(new_urdf_file_name)[0]
    print(new_urdf_info["joint"])
    # exit(123)
    with open(bbox_file_name , "rb") as f:
        bbox_data = pickle.load(f)

    Rtilt_rot = np.array(cammeta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(cammeta['camera2world_translation']).reshape(1, 3)



    pc, rgb, semantic_label, instance_label, npcs_map = torch.load(pc_file_name)


    
    # exit(123)
    rgb = ((rgb + 1) * 127.5).astype(np.uint8)
    trans = np.loadtxt(meta_file_name)
    xyz = pc * trans[0] + trans[1:4]
    
    img = map2image(xyz, rgb)
    for i in range(len(bbox_data['bboxes_with_pose'])):
        bbox = bbox_data['bboxes_with_pose'][i]["bbox_3d"]
        bbox_pc = (bbox - Rtilt_trl) @ Rtilt_rot
        # img = draw_bbox(img, [bbox_pc], trans)
        if i ==1:
            break
    for i in range(len(bbox_list)):
        bbox = bbox_list[i]
        bbox_pc = (bbox - Rtilt_trl) @ Rtilt_rot
        # img = draw_bbox(img, [bbox_pc], trans)
    img_new = img

    joints = []
    roots = []
    joints_axis = []
    door_index = []
    joint_ids = [i for i in range(len(urdf_info['joint']["type"])) if urdf_info['joint']["type"][i] == joint_type]

    print(len(bbox_list))
    print(joint_ids)
    for _,i in enumerate(joint_ids):
        joint_base = np.array(urdf_info['joint']["xyz"][i])#.reshape(1,-1)
        new_base = np.array(urdf_info['link']["xyz"][i+1][0])#.reshape(1,-1)
        axis_base = np.array(urdf_info['joint']["axis"][i])#.reshape(1,-1)
        root_base = (joint_base + new_base)
        joint_axis_base = (joint_base + axis_base)

        joint_world = np.array([-joint_base[2], -joint_base[0], -joint_base[1]])
        root_world = np.array([-root_base[2], -root_base[0], -root_base[1]])
        joint_axis_world = np.array([-joint_axis_base[2], -joint_axis_base[0], -joint_axis_base[1]])
        axis_world = np.array([-axis_base[2], -axis_base[0], -axis_base[1]])

        joints.append(joint_world)
        roots.append(root_world)
        joints_axis.append(joint_axis_world)

        joint_cam = ((joint_world- Rtilt_trl) @ Rtilt_rot )[0]
        root_cam =  ((root_world - Rtilt_trl) @ Rtilt_rot)[0]
        joint_axis_cam =  ((joint_axis_world - Rtilt_trl) @ Rtilt_rot)[0]
        # x_new = (np.around(joint[0] * K[0][0] / joint[2] + K[0][2])).astype(dtype=int)
        # y_new = (np.around(joint[1] * K[1][1] / joint[2] + K[1][2])).astype(dtype=int)
        # x_new_ = (np.around(root[0] * K[0][0] / root[2] + K[0][2])).astype(dtype=int)
        # y_new_ = (np.around(root[1] * K[1][1] / root[2] + K[1][2])).astype(dtype=int)
        # x_new_joint_axis = (np.around(joint_axis[0] * K[0][0] / joint_axis[2] + K[0][2])).astype(dtype=int)
        # y_new_joint_axis = (np.around(joint_axis[1] * K[1][1] / joint_axis[2] + K[1][2])).astype(dtype=int)

        new_bbox = bbox_process(bbox_list[_], joint_world, axis_world, joint_type, qpos[i]) 
        new_bbox = (new_bbox - Rtilt_trl) @ Rtilt_rot
        bbox_tmp = (bbox_list[_] - Rtilt_trl - joint_world) @ Rtilt_rot
        img = draw_bbox(img, [new_bbox], trans)

        draw_line(joint_cam, root_cam, img, color=(int(255),int(0),int(255)))
        draw_line(joint_cam, joint_axis_cam, img ,color=(int(255),int(255),int(0)))
        # cv2.line(img,[x_new,y_new], [x_new_, y_new_],color=(int(255),int(0),int(255)),thickness=2)

        # cv2.line(img,[x_new,y_new], [x_new_, y_new_],color=(int(255),int(0),int(255)),thickness=2)
 
        # cv2.line(img,[x_new,y_new], [x_new_joint_axis, y_new_joint_axis],color=(int(255),int(255),int(0)),thickness=2)


    cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/{cate}_{id}_00_000.png", img_new)

    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(xyz @ Rtilt_rot.T + Rtilt_trl)

    # print(np.array(roots))
    # print(np.array(roots))
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(np.array(roots))

    # pcd3 = o3d.geometry.PointCloud()
    # pcd3.points = o3d.utility.Vector3dVector(np.array(roots))

    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries([
    #     pcd1,
    #     pcd3, 
    #     pcd2, 
    #     coord_frame,
    #     ])