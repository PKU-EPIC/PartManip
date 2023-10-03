import pickle

# from pose_utils import get_urdf_mobility
import torch
import numpy as np
import cv2
import open3d as o3d
import json
# from isaacgym.torch_utils import *
import numpy as np
import math
import xml.etree.ElementTree as ET

HEIGHT = int(800)
WIDTH = int(800)
K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
                [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

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

def map2image(pts, rgb):
    # input为每个shape的info，取第idx行
    image_rgb = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    num_point = pts.shape[0]

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

def bbox_process(bbox, joint_xyz, axis, joint_type, qpos, parent_joint_type=-1):

    if joint_type == "revolute":
        return np.dot(rotation_matrix(axis, -qpos), (bbox - joint_xyz).T).T + joint_xyz
    elif joint_type == "prismatic":
        return bbox - axis*qpos
    else:
        if parent_joint_type == "revolute":
            return np.dot(rotation_matrix(axis, -qpos), (bbox - joint_xyz).T).T + joint_xyz
        elif parent_joint_type == "prismatic": 
            return bbox - axis*qpos
        
def get_urdf_mobility(inpath, verbose=False):
    if not inpath.endswith(".urdf"):
        urdf_name = inpath + "/mobility_new.urdf"
    else:
        urdf_name = inpath
        inpath = '/'.join(inpath.split('/')[:-1])

    urdf_ins = {}
    tree_urdf = ET.parse(urdf_name)
    num_all_links = len(tree_urdf.findall('link'))
    root_urdf = tree_urdf.getroot()
    rpy_xyz = {}
    
    handle_parent = []
    handle_link_name  = []
    num_real_links = 0
    num_handle_links = 0

    for link in root_urdf.iter('link'):
        if link.attrib['name'] == 'base':
            num_real_links += 1
        elif link.attrib["name"].find("handle") != -1:
            num_handle_links += 1
        else:
            num_real_links += 1

    list_xyz = [None] * num_real_links
    list_rpy = [None] * num_real_links
    list_obj = [None] * num_real_links
    list_name = [None] * num_real_links
    # exit(123)
    for link in root_urdf.iter('link'):
        
        if link.attrib['name'] == 'base':
            index_link = 0

        elif "handle" in link.attrib["name"]:
            index_handle = index_link
            handle_parent.append(index_handle)
            handle_link_name.append(link.attrib["name"])
            continue
        else:
            index_link = int(link.attrib['name'].split('_')[1]) + 1  # since the name is base, link_0, link_1
        list_xyz[index_link] = []
        list_rpy[index_link] = []
        list_obj[index_link] = []
        list_name[index_link] = link.attrib["name"]
        for visual in link.iter('visual'):
            for origin in visual.iter('origin'):
                if 'xyz' in origin.attrib:
                    list_xyz[index_link].append([float(x) for x in origin.attrib['xyz'].split()])
                else:
                    list_xyz[index_link].append([0, 0, 0])
                if 'rpy' in origin.attrib:
                    list_rpy[index_link].append([float(x) for x in origin.attrib['rpy'].split()])
                else:
                    list_rpy[index_link].append([0, 0, 0])
            for geometry in visual.iter('geometry'):
                for mesh in geometry.iter('mesh'):
                    if 'home' in mesh.attrib['filename'] or 'work' in mesh.attrib['filename']:
                        list_obj[index_link].append(mesh.attrib['filename'])
                    else:
                        list_obj[index_link].append(inpath + '/' + mesh.attrib['filename'])

    rpy_xyz['xyz'] = list_xyz
    rpy_xyz['rpy'] = list_rpy  # here it is empty list
    urdf_ins['link'] = rpy_xyz
    urdf_ins['obj_name'] = list_obj
    urdf_ins["name"] = list_name
    rpy_xyz = {}
    list_type = [None] * (num_real_links - 1)
    list_name = [None] * (num_real_links - 1)
    list_parent = [None] * (num_real_links - 1)
    list_child = [None] * (num_real_links - 1)
    list_xyz = [None] * (num_real_links - 1)
    list_rpy = [None] * (num_real_links - 1)
    list_axis = [None] * (num_real_links - 1)
    list_limit = [[0, 0]] * (num_real_links - 1)
    # here we still have to read the URDF file
    handle_joint_names = []
    for joint in root_urdf.iter('joint'):
        """
        joint_index = int(joint.attrib['name'].split('_')[1])
        list_type[joint_index] = joint.attrib['type']
        """
        if joint.attrib["name"].find("handle") != -1:

            handle_joint_names.append(joint.attrib["name"])
            continue
        for child in joint.iter('child'):
            link_name = child.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            joint_index = link_index - 1  # !!! the joint's index should be the same as its child's index - 1
            list_child[joint_index] = link_index

        list_type[joint_index] = joint.attrib['type']
        list_name[joint_index] = joint.attrib['name']

        for parent in joint.iter('parent'):
            link_name = parent.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            list_parent[joint_index] = link_index

        for origin in joint.iter('origin'):
            if 'xyz' in origin.attrib:
                list_xyz[joint_index] = [float(x) for x in origin.attrib['xyz'].split()]
            else:
                list_xyz[joint_index] = [0, 0, 0]
            if 'rpy' in origin.attrib:
                list_rpy[joint_index] = [float(x) for x in origin.attrib['rpy'].split()]
            else:
                list_rpy[joint_index] = [0, 0, 0]
        for axis in joint.iter('axis'):  # we must have
            list_axis[joint_index] = [float(x) for x in axis.attrib['xyz'].split()]
        for limit in joint.iter('limit'):
            list_limit[joint_index] = [float(limit.attrib['lower']), float(limit.attrib['upper'])]
        # 特殊处理continuous的上下限，和render_utils的处理保持一致
        if joint.attrib['type'] == 'continuous':
            list_limit[joint_index] = [-10000.0, 10000.0]

    rpy_xyz['type'] = list_type
    rpy_xyz['parent'] = list_parent
    rpy_xyz['child'] = list_child
    rpy_xyz['xyz'] = list_xyz
    rpy_xyz['rpy'] = list_rpy
    rpy_xyz['axis'] = list_axis
    rpy_xyz['limit'] = list_limit
    rpy_xyz['name'] = list_name

    urdf_ins['joint'] = rpy_xyz
    urdf_ins['num_all_links'] = num_all_links
    urdf_ins['num_handle_links'] = num_handle_links
    urdf_ins["handle_name"] = handle_link_name
    urdf_ins["handle_parent"] = handle_parent
    urdf_ins["handle_joint_names"] = handle_joint_names
    
    return urdf_ins

def rotation_matrix_(theta1, theta2, theta3, order='xyz'):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        oreder = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    # c1 = np.cos(theta1 * np.pi / 180)
    # s1 = np.sin(theta1 * np.pi / 180)
    # c2 = np.cos(theta2 * np.pi / 180)
    # s2 = np.sin(theta2 * np.pi / 180)
    # c3 = np.cos(theta3 * np.pi / 180)
    # s3 = np.sin(theta3 * np.pi / 180)
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c3 = np.cos(theta3)
    s3 = np.sin(theta3)

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])

    return matrix


base_r = rotation_matrix_(1.570796326794897, 0.0, -1.570796326794897)
save_root = "/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered"
# ['4108']
from glob import glob
all_ids = [i.split("/")[-2].split("-")[-1] for i in sorted(glob("/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified/*/mobility_new.urdf"))]
unfinished_ids = all_ids.copy()
all_cats = [i.split("/")[-2].split("-")[-3] for i in sorted(glob("/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified/*/mobility_new.urdf"))]
# all_ids = ["100214"] # ["46922"]StorageFurnitur
# all_cats = ["Box"]
failure_cases = []
# all_bbox_info = []
for _, id in enumerate(all_ids):
    # id = "45948"
    cate = all_cats[_]#"StorageFurniture"
    print(id,cate)
    finish_flag = 0
    
    info = {}
    
    for img_id in range(32):
        img_id = str(img_id).zfill(2)
        # print(img_id)
        # continue
        try:
            bbox_file_name = f"/data2/haoran/data/source_data/source_data_2/valid/annotation/bbox/{cate}_{id}_00_0{img_id}.pkl"
            pc_file_name = f"/data2/haoran/data/PartDetection_data_all/train/pth/{cate}_{id}_00_0{img_id}.pth"
            meta_file_name = f"/data2/haoran/data/PartDetection_data_all/train/meta/{cate}_{id}_00_0{img_id}.txt"
            cammeta_file_name = f"/data2/haoran/data/source_data/source_data_2/valid/metafile/{cate}_{id}_00_0{img_id}.json"
            urdf_file_name = f"/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/{id}/mobility_relabel.urdf"
            new_urdf_file_name = f"/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/{id}/mobility_new.urdf"
            with open(cammeta_file_name,"r") as f:
                cammeta = json.load(f)
            pc, rgb, semantic_label, instance_label, npcs_map = torch.load(pc_file_name)

            rgb = ((rgb + 1) * 127.5).astype(np.uint8)
            trans = np.loadtxt(meta_file_name)
            xyz = pc * trans[0] + trans[1:4]

            img = map2image(xyz, rgb)
            img_new = img
            qpos = cammeta['joint_qpos']
            new_urdf_info = get_urdf_mobility(new_urdf_file_name)
            bbox_info = {}
            all_bbox_world = []
            all_bbox_link_name = []
            all_bbox_parent_joint_name = []
            all_bbox_class_name = []
            all_bbox_parent_joint_type = []
            all_bbox_axis_xyz_world = []
            all_bbox_axis_dir_world = []
            all_bbox_class_id = []
            all_bbox_instance_id = []

            for joint_type_id in range(3):
                joint_type = ["revolute", 'prismatic', "fixed"][joint_type_id]

                target_part = [['hinge_door', "hinge_knob", "hinge_lid"],["slider_drawer", "slider_button", "slider_lid"], ["line_fixed_handle","round_fixed_handle"]][joint_type_id]



                urdf_info = new_urdf_info

                with open(bbox_file_name , "rb") as f:
                    bbox_data = pickle.load(f)
                # print(bbox_data)
                # exit(123)
                bbox_list = [bbox_data['bboxes_with_pose'][i]["bbox_3d"] for i in range(len(bbox_data['bboxes_with_pose'])) if  bbox_data['bboxes_with_pose'][i]['category'] in target_part]
                bbox_cat_list = [bbox_data['bboxes_with_pose'][i]["category"] for i in range(len(bbox_data['bboxes_with_pose'])) if  bbox_data['bboxes_with_pose'][i]['category'] in target_part]
                bbox_catid_list = [bbox_data['bboxes_with_pose'][i]["category_id"] for i in range(len(bbox_data['bboxes_with_pose'])) if  bbox_data['bboxes_with_pose'][i]['category'] in target_part]
                # bbox_ins_list = [bbox_data['bboxes_with_pose'][i]["'category'"] for i in range(len(bbox_data['bboxes_with_pose'])) if  bbox_data['bboxes_with_pose'][i]['category'] in target_part]
                bbox_insid_list = [bbox_data['bboxes_with_pose'][i]["instance_id"] for i in range(len(bbox_data['bboxes_with_pose'])) if  bbox_data['bboxes_with_pose'][i]['category'] in target_part]

                Rtilt_rot = np.array(cammeta['world2camera_rotation']).reshape(3, 3)
                Rtilt_trl = np.array(cammeta['camera2world_translation']).reshape(1, 3)

                joints = []
                roots = []
                joints_axis = []
                door_index = []

                joint_ids = [i for i in range(len(urdf_info['joint']["type"])) if urdf_info['joint']["type"][i] == joint_type]
                if joint_type_id == 2:
                    joint_ids = [urdf_info['joint']["child"].index(urdf_info["handle_parent"][i]) for i in range(len(urdf_info["handle_parent"]))]

                for _,i in enumerate(joint_ids):

                    joint_base = np.array(urdf_info['joint']["xyz"][i])#.reshape(1,-1)
                    new_base = np.array(urdf_info['link']["xyz"][i+1][0])#.reshape(1,-1)
                    axis_base = np.array(urdf_info['joint']["axis"][i])#.reshape(1,-1)

                    root_base = (joint_base + new_base)
                    joint_axis_base = (joint_base + axis_base)

                    joint_world = np.array([-joint_base[2], -joint_base[0], joint_base[1]])
                    root_world = np.array([-root_base[2], -root_base[0], root_base[1]])
                    joint_axis_world = np.array([-joint_axis_base[2], -joint_axis_base[0], joint_axis_base[1]])
                    axis_world = np.array([-axis_base[2], -axis_base[0], axis_base[1]])

                    #
                    # joint_world = np.matmul(base_r,joint_base)
                    # root_world = np.matmul(base_r,root_base)
                    # joint_axis_world = np.matmul(base_r,joint_axis_base)
                    # axis_world = np.matmul(base_r,axis_base)
                    #

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
                    parent_joint_type = urdf_info['joint']["type"][i]

                    new_bbox_world = bbox_process(bbox_list[_], joint_world, axis_world, joint_type, qpos[i], parent_joint_type) 

                    all_bbox_world.append(new_bbox_world.tolist())
                    if joint_type_id != 2: 
                        all_bbox_link_name.append(urdf_info["name"][i+1])
                        all_bbox_parent_joint_name.append(urdf_info['joint']["name"][i])
                    else:
                        all_bbox_link_name.append(urdf_info['handle_name'][_])
                        all_bbox_parent_joint_name.append(urdf_info['handle_joint_names'][_])
                    all_bbox_parent_joint_type.append(joint_type)
                    all_bbox_axis_xyz_world.append(joint_world.tolist())
                    all_bbox_axis_dir_world.append(axis_world.tolist())
                    all_bbox_class_name.append(bbox_cat_list[_])
                    all_bbox_class_id.append(bbox_catid_list[_])
                    all_bbox_instance_id.append(bbox_insid_list[_])
                    
                    new_bbox_cam = (new_bbox_world - Rtilt_trl) @ Rtilt_rot
                    bbox_tmp = (bbox_list[_] - Rtilt_trl - joint_world) @ Rtilt_rot
                    img = draw_bbox(img, [new_bbox_cam], trans)


            cv2.imwrite(f"/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/{id}/{cate}_{id}.png", img_new)
            unfinished_ids.remove(id)
            finish_flag = 1

            bbox_info["bbox_world"] = all_bbox_world
            bbox_info["link_name"] =  all_bbox_link_name 
            bbox_info["parent_joint_name"] =  all_bbox_parent_joint_name
            bbox_info["class_name"] =   all_bbox_class_name
            bbox_info["parent_joint_type"] =  all_bbox_parent_joint_type
            bbox_info["axis_xyz_world"] =  all_bbox_axis_xyz_world
            bbox_info["axis_dir_world"] =   all_bbox_axis_dir_world 
            bbox_info["class_id"] =   all_bbox_class_id 
            bbox_info["instance_id "] =  all_bbox_instance_id 
            # with open('params.json', 'w') as f:  # 创建一个params.json文件
            #     f.write(json_str)
            json_str = json.dumps(bbox_info)
            f = open(f'/data2/haoran/data/SAPIEN/SAPIEN_v4/all_used_modified_unordered/{id}/bbox_info.json', 'w', encoding="utf8")
            f.write(json_str)
            print(img_id)
            break
        except:
            if img_id == 31:
                failure_cases.append((id,cate))
            continue
        
print(failure_cases)
print(unfinished_ids)


# ['100214', '48492', '100189', '100191', '100221', '100243', '100247', '101352', '102394', '102408', '102411', '102417', '102431', '102432', '102434', 
# '102442', '102505', '102520', '102527', '102528', '102539', '102831', '102845', '102876', '102892', '102398', '102403', '102472', '102536', '102873', 
# '102874', '102882', '102145', '103002', '103030', '103038', '103041', '103043', '103048', '103064', '103069', '103072', '103079', '103086', '103105', 
# '103110', '103118', '102901', '103037', '103074', '103082', '103092', '103128', '103129', '103143', '11826', '12349', '11700', '12065', '12428', '12484', 
# '12530', '12558', '12559', '12597', '7119', '7263', '7296', '7349', '100997', '101016', '101118', '100385', '100395', '100706', '101004', '101028', 
# '101104', '101121', '101133', '104038', '104040', '46127', '46132', '46768', '47235', '47391', '47963', '48623', '49062', '35059', '40453', '41004', 
# '45007', '45134', '45159', '45166', '45173', '45212', '45261', '45372', '45385', '45403', '45444', '45606', '45623', '45633', '45671', '45676', '45691',
# '45746', '45759', '45767', '45779', '45783', '45841', '45855', '45910', '45949', '45963', '46014', '46019', '46107', '46109', '46145', '46172', '46179', 
# '46277', '46380', '46437', '46481', '46598', '46641', '46653', '46655', '46699', '46744', '46889', '47021', '47133', '47182', '47233', '47419', '47443', 
# '47585', '48036', '48381', '48497', '48721', '48797', '49038', '49188', '103466', '103475', '103477', '103482', '103485', '103502', '103514', '103547', 
# '103548', '103549', '103553', '103556', '103558', '103560', '103469', '103486', '103524', '103561', '100283', '103369', '103528', '100015', '100017', 
# '100021', '100023', '100025', '100028', '100032', '100033', '100040', '100045', '100047', '100051', '100054', '100056', '100057', '100058', '100060', 
# '100619', '100623', '102080', '101773', '101908', '101917', '101921', '101930', '101931', '101943', '101946', '101947', '101971', '102018', '102044', 
# '102055', '102060', '7120', '7130', '7138', '7179', '7187', '7201', '7220', '7290', '7332', '103251', '103285', '103347', '103593', '103699', '103813', 
# '103814', '103828', '103886', '103892', '103917', '103925', '103927', '103935', '103941', '10068', '10143', '10144', '10373', '10489', '10586', '10620', 
# '10627', '10655', '10685', '10751', '10797', '10849', '10867', '10900', '10905', '10944', '11178', '11211', '11231', '11260', '11299', '11304', '11709', 
# '11712', '11846', '12036', '12038', '12042', '12050', '12054', '12055', '12059', '12248', '12249', '12250', '12252', '101363', '101564', '101579', '101583', 
# '101584', '101591', '101593', '101594', '101599', '101603', '101605', '101611', '101612', '101613', '101619', '101623', '102278', '102301', '102309', 
# '102311', '102316', '102318', '102380', '102381', '102384', '102387', '102389', '102418', '102423', '19179', '19825', '19836', '19855', '19898', '20043', 
# '20279', '20411', '20453', '20555', '20985', '21467', '22241', '22301', '22339', '22367', '22433', '22508', '22692', '23372', '23472', '23511', '23724',
# '23782', '23807', '24644', '24931', '25144', '25308', '25493', '25913', '26073', '26387', '26503', '26525', '26608', '26652', '26657', '26670', '26806', 
# '26875', '26899', '27044', '27189', '27267', '27619', '28668', '29133', '29525', '29557', '29921', '30238', '30341', '30663', '30666', '30739', '30857', 
# '30869', '31249', '31601', '32052', '32086', '32174', '32259', '32324', '32354', '32566', '32601', '32746', '32761', '32932', '33116', '33914', '33930', 
# '34178', '34610', '34617', '100731', '100732', '101377', '101378', '101380', '102154', '102155', '102156', '102158', '102160', '102165', '102171', '102181', 
# '102186', '102187', '102189', '102192', '102194', '102200', '102201', '102202', '102209', '102210', '102218', '102219', '102227', '102229', '102234', 
# '102244', '102252', '102254', '102256', '102257', '102992', '102996', '103008', '103010', '103013', '103633', '103634', '103646', '103647', '11229', 
# '11259', '11279', '11361', '11818', '11951', '12231', '12447', '12483', '4108']