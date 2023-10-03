import os
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import transforms3d.axangles as t
import math
import xml.etree.ElementTree as ET

# TODO 超参，用来调整tight bbox在每个方向的缩放倍数，以容纳point cloud的微小错位（原因未知）
EPSILON_L = 0.001
EPSILON_W = 0.001
EPSILON_H = 0.001

# TODO 全局变量，merge fixed handle的annotation方式，统一处理
FIXED_HANDLE_ANNO_SCHEME = {
    "CoffeeMachine": {
        1: 6,
        2: 16
    },
    "Dishwasher": {
        1: 2
    },
    "KitchenPot": {
        1: 5,
        2: 8,
        3: -1
    },
    "Microwave": {
        1: 1
    },
    "StorageFurniture": {
        1: 1,
        2: 2,
        3: 2,
        4: 3,
        5: 7,
        6: 3,
        7: 4,
        8: 9,
        9: 4
    },
    "TrashCan": {
        1: 8,
        2: 10,
        3: 2,
        4: 3,
        5: 1,
        6: 4,
        7: 13,
        8: 14,
        9: 15
    },
    "Box": {
        1: 10
    },
    "Oven": {
        1: 2
    },
    "Refrigerator": {
        1: 1,
        2: 2
    },
    "Safe": {
        1: 1
    },
    "Table": {
        1: 1,
        2: 7,
        3: 11,
        4: 12
    }
}


def draw_line(x, y, z, vector):
    points = []
    list = [(0 + i * 0.001) for i in range(3000)]
    for i in list:
        point = [x + vector[0] * i, y + vector[1] * i, z + vector[2] * i]
        points.append(point)
    return np.array(points)


def draw_bbox(bbox, category_id):
    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]

    points = []
    for i in range(bbox.shape[0]):
        points.append(bbox[i].reshape(-1).tolist())
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    # Use the same color for all lines
    colors = [cmap[category_id] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_bbox_with_pose(bbox, category_id):
    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]

    points = []
    for i in range(bbox.shape[0]):
        points.append(bbox[i].reshape(-1).tolist())
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    # Use the same color for all lines
    colors = [
        cmap[0], cmap[2], cmap[4], cmap[6], cmap[8], cmap[10], cmap[12], cmap[16], cmap[14], cmap[14], cmap[14],
        cmap[14]
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def load_model_split(inpath):
    vsplit = []
    fsplit = []
    dict_mesh = {}
    list_group = []
    list_xyz = []
    list_face = []
    with open(inpath, "r", errors='replace') as fp:
        line = fp.readline()
        cnt = 1
        while line:
            if len(line) < 2:
                line = fp.readline()
                cnt += 1
                continue
            xyz = []
            face = []
            if line[0] == 'g':
                list_group.append(line[2:])
            if line[0:2] == 'v ':
                vcount = 0
                while line[0:2] == 'v ':
                    xyz.append([float(coord) for coord in line[2:].strip().split()])
                    vcount += 1
                    line = fp.readline()
                    cnt += 1
                vsplit.append(vcount)
                list_xyz.append(xyz)

            if line[0] == 'f':
                fcount = 0
                while line[0] == 'f':
                    face.append([num for num in line[2:].strip().split()])
                    fcount += 1
                    line = fp.readline()
                    cnt += 1
                    if not line:
                        break
                fsplit.append(fcount)
                list_face.append(face)
            line = fp.readline()
            cnt += 1
    dict_mesh['v'] = list_xyz
    dict_mesh['f'] = list_face

    return dict_mesh, list_group, vsplit, fsplit


def get_all_objs(obj_file_list):
    pts_list = []  # for each link, a list of vertices
    name_list = []  # for each link, the .obj filenames
    offset = 0

    def read_obj_file(obj_file):
        try:
            tm = trimesh.load(obj_file)
            vertices_obj = np.array(tm.vertices)
        except:
            dict_mesh, _, _, _ = load_model_split(obj_file)
            vertices_obj = np.concatenate(dict_mesh['v'], axis=0)
        return vertices_obj

    for k, obj_files in enumerate(obj_file_list):  # each corresponds to a link
        cur_list = None
        if isinstance(obj_files, list):
            cur_list = obj_files
        elif obj_files is not None:
            cur_list = [obj_files]
        # collect all names & vertices
        part_pts = []
        name_objs = []
        for obj_file in cur_list:
            if obj_file is not None and not isinstance(obj_file, list):
                vertices_obj = read_obj_file(obj_file)
                part_pts.append(vertices_obj)
                name_obj = obj_file.split('.')[0].split('/')[-1]
                name_objs.append(name_obj)

        part_pts_all = np.concatenate(part_pts, axis=0)
        pts_list.append(part_pts_all + offset)
        name_list.append(name_objs)  # which should follow the right

    # vertices: a list of sublists,
    # sublists contain vertices in the whole shape (0) and in each part (1, 2, ..)
    vertices = [pts_list]
    for part in pts_list:
        vertices.append([part])

    norm_factors = []  # for each link, a float
    corner_pts = []
    # calculate bbox & corners for the whole object
    # as well as each part
    for j in range(len(vertices)):

        part_verts = np.concatenate(vertices[j], axis=0)  # merge sublists
        pmax, pmin = np.amax(part_verts, axis=0), np.amin(part_verts, axis=0)
        corner_pts.append([pmin, pmax])  # [index][left/right][x, y, z], numpy array
        norm_factor = np.sqrt(1) / np.sqrt(np.sum((pmax - pmin)**2))
        norm_factors.append(norm_factor)

    return vertices[1:], norm_factors, corner_pts


def get_model_pts(obj_file_list, is_debug=False):
    """
    For item obj_category/item,
    get_urdf(_mobility) returns obj_file_list:
        [[objs of 0th link], [objs of 1st link], ...]
    This function reads these obj files,
    and calculates the following for each link (part) --- except the first link (base link)
    - model parts,
    - norm_factors, 1 / diagonal of bbox
    - corner_pts, the corners of the bbox

    """
    if obj_file_list is not None and obj_file_list[0] == []:
        if is_debug:
            print('removing the first obj list, which corresponds to base')
        obj_file_list = obj_file_list[1:]

    model_pts, norm_factors, corner_pts = get_all_objs(obj_file_list=obj_file_list)
    return model_pts, norm_factors, corner_pts


def get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s):
    l = l_s / 2
    w = w_s / 2
    h = h_s / 2
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] += x_center
    corners_3d[1, :] += y_center
    corners_3d[2, :] += z_center
    return np.transpose(corners_3d)


def get_slider_button_pose(inst_name, pcs, scene, urdf_ins, link_pose_dict, joint_pose_dict, meta):

    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_info = urdf_ins['joint']
    link_name = inst_name.split(':')[0]
    if link_name == 'base':
        link_id = 0
    else:
        link_id = int(link_name.split('_')[-1]) + 1

    joint_id_list = []
    last_link_id = link_id
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links'] - 1):
            if joint_info['child'][
                    i] == last_link_id:  # !!! the joint's index should be the same as its child's index - 1
                joint_id_list.append(i)
                if joint_info['parent'][i] == 0:
                    end_flag = True
                    break
                else:
                    last_link_id = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    if len(joint_id_list) == 1:  # kinematic chain depth = 1
        joint_id = joint_id_list[0]
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        joint_qpos = joint_qpos_list[joint_id]  # !!! the joint's index should be the same as its child's index - 1

        # 获得当前button对应轴的xyz点和方向向量
        axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
        axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
        axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

        # 读取part对应的.obj文件
        obj_list = urdf_ins['obj_name'][link_id]
        part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
        part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                             )  # 从obj空间转移到sapien world space (利用base处的rpy？)

        # 以axis方向向量的反向为 canon space +z，world space -z且与方向向量垂直为canon space +y，叉乘得canon space +x
        canon_in_world_z = (-1) * axis_direction_vector
        canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)
        # 判断part的z轴是否指向world space的+z或-z，都不是则取重力方向world space -z相关方向为+y
        if abs(canon_in_world_z[0, 2] - 1.0) > 1e-5 and abs(canon_in_world_z[0, 2] + 1.0) > 1e-5:
            canon_in_world_x = np.cross(np.array([0, 0, -1]), canon_in_world_z)
            canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
            canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
            canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
        # part的z轴指向world space的+z，则取world space +y为+y
        elif abs(canon_in_world_z[0, 2] - 1.0) <= 1e-5:
            canon_in_world_x = np.cross(np.array([0, 1, 0]), canon_in_world_z)
            canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
            canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
            canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
        # part的z轴指向world space的-z，则取world space -y为+y
        else:
            canon_in_world_x = np.cross(np.array([0, -1, 0]), canon_in_world_z)
            canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
            canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
            canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)

        # 得到sapien world space到canon space的rotation matrix
        canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
        world2canon = canon2world.T

        # 转换part到canon space，得到tight bbox和pose
        part_conon_pts = part_obj_pts @ world2canon
        axis_canon = canon_in_world_z @ world2canon
        # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
        pcs_in_original = pcs_world - (axis_direction_vector * joint_qpos)
        pcs_original_canon = pcs_in_original @ world2canon
        part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

        x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
        y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
        z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
        l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
        w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
        h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

        center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
        tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
        tight_bbox_world = tight_bbox_canon @ canon2world

        # 恢复转轴在canon space内的位置
        # new_z = canon_in_world_z @ world2canon
        # new_y = canon_in_world_y @ world2canon
        # new_x = canon_in_world_x @ world2canon

        # 恢复part在sapien里的移动，通过joint的qpos
        center_t_moved = center_t + (axis_direction_vector * joint_qpos)
        tight_bbox_world_moved = tight_bbox_world + (axis_direction_vector * joint_qpos)
        x_t = center_t_moved[0, 0]
        y_t = center_t_moved[0, 1]
        z_t = center_t_moved[0, 2]

        R = canon2world  # shape (3,3)
        T = center_t_moved.reshape(3, )
        S = np.array([l_s, w_s, h_s]).reshape(3, )

        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(part_conon_pts)
        # bbox1 = draw_bbox(tight_bbox_canon, 5)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(part_obj_pts)
        # bbox2 = draw_bbox(tight_bbox_world, 7)
        # pcd3 = o3d.geometry.PointCloud()
        # pcd3.points = o3d.utility.Vector3dVector(pcs_world)
        # bbox3 = draw_bbox(tight_bbox_world_moved, 9)
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcd1, bbox1, pcd3, bbox3, coord_frame])

        pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
        if pc_in_bbox.shape[0] != pcs_world.shape[0]:
            print('part points: ', pcs_world.shape[0])
            print('points in box: ', pc_in_bbox.shape[0])
            print('Error! tight bbox failed: not all points are in the tight bbox!')
            exit(-1)

        return (tight_bbox_world_moved, (R, T, S))

        # axis_vector = joint_pose_dict[joint_id]['axis']
        # print(axis_start_point)
        # print(axis_vector)
        # line_z = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
        #                    new_z.reshape(3, ).tolist())
        # line_x = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
        #                    new_x.reshape(3, ).tolist())
        # line_y = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
        #                    new_y.reshape(3, ).tolist())
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcs_world)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(np.array(line_z))
        # pcd3 = o3d.geometry.PointCloud()
        # pcd3.points = o3d.utility.Vector3dVector(part_obj_pts)
        # pcd4 = o3d.geometry.PointCloud()
        # pcd4.points = o3d.utility.Vector3dVector(np.array(line_x))
        # pcd5 = o3d.geometry.PointCloud()
        # pcd5.points = o3d.utility.Vector3dVector(np.array(line_y))
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcd, pcd2, pcd3, pcd4, pcd5, coord_frame])

        # return tight_bbox_world_moved

    elif len(joint_id_list) == 2:  # kinematic chain depth = 2
        child_joint_id = joint_id_list[0]
        # 获得当前button对应轴的xyz点和方向向量 轴在world space中，叠加了parent joint的运动，需要先剔除
        child_axis_start_point = np.array(joint_pose_dict[child_joint_id]['xyz']).reshape(1, 3)
        child_axis_direction_vector = np.array(joint_pose_dict[child_joint_id]['axis']).reshape(1, 3)
        child_axis_direction_vector = child_axis_direction_vector / np.linalg.norm(child_axis_direction_vector)

        parent_joint_id = joint_id_list[1]
        # 获得当前button的parent joint对应轴的xyz点和方向向量 轴在world space中
        parent_axis_start_point = np.array(joint_pose_dict[parent_joint_id]['xyz']).reshape(1, 3)
        parent_axis_direction_vector = np.array(joint_pose_dict[parent_joint_id]['axis']).reshape(1, 3)
        parent_axis_direction_vector = parent_axis_direction_vector / np.linalg.norm(parent_axis_direction_vector)

        # 获得joint的qpos
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        child_joint_qpos = joint_qpos_list[
            child_joint_id]  # !!! the joint's index should be the same as its child's index - 1
        parent_joint_qpos = joint_qpos_list[parent_joint_id]

        parent_joint_type = joint_info['type'][parent_joint_id]
        if parent_joint_type == 'prismatic':
            child_axis_start_point_static = child_axis_start_point - (parent_axis_direction_vector * parent_joint_qpos)
            child_axis_direction_vector_static = child_axis_direction_vector
            child_axis_direction_vector_static = child_axis_direction_vector_static / np.linalg.norm(
                child_axis_direction_vector_static)

            # 读取part对应的.obj文件
            obj_list = urdf_ins['obj_name'][link_id]
            part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
            part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                                 )  # 从obj空间转移到sapien world space (利用base处的rpy？)

            # 以axis方向向量的反向为 canon space +z，world space -z且与方向向量垂直为canon space +y，叉乘得canon space +x
            canon_in_world_z = (-1) * child_axis_direction_vector_static
            canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)
            # 判断part的z轴是否指向world space的+z或-z，都不是则取重力方向world space -z相关方向为+y
            if abs(canon_in_world_z[0, 2] - 1.0) > 1e-5 and abs(canon_in_world_z[0, 2] + 1.0) > 1e-5:
                canon_in_world_x = np.cross(np.array([0, 0, -1]), canon_in_world_z)
                canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
                canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
                canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
            # part的z轴指向world space的+z，则取world space +y为+y
            elif abs(canon_in_world_z[0, 2] - 1.0) <= 1e-5:
                canon_in_world_x = np.cross(np.array([0, 1, 0]), canon_in_world_z)
                canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
                canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
                canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
            # part的z轴指向world space的-z，则取world space -y为+y
            else:
                canon_in_world_x = np.cross(np.array([0, -1, 0]), canon_in_world_z)
                canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
                canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
                canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)

            # 得到sapien world space到canon space的rotation matrix
            canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
            world2canon = canon2world.T

            # 转换part到canon space，得到tight bbox和pose
            part_conon_pts = part_obj_pts @ world2canon
            axis_canon = canon_in_world_z @ world2canon
            # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
            pcs_in_original = pcs_world - (child_axis_direction_vector * child_joint_qpos)
            pcs_in_original = pcs_in_original - (parent_axis_direction_vector * parent_joint_qpos)
            pcs_original_canon = pcs_in_original @ world2canon
            part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

            x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
            y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
            z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
            l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
            w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
            h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

            center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
            tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
            tight_bbox_world = tight_bbox_canon @ canon2world

            # 恢复part在sapien里的移动，通过joint的qpos
            center_t_moved = center_t + (parent_axis_direction_vector * parent_joint_qpos)
            center_t_moved = center_t_moved + (child_axis_direction_vector * child_joint_qpos)
            tight_bbox_world_moved = tight_bbox_world + (parent_axis_direction_vector * parent_joint_qpos)
            tight_bbox_world_moved = tight_bbox_world_moved + (child_axis_direction_vector * child_joint_qpos)
            x_t = center_t_moved[0, 0]
            y_t = center_t_moved[0, 1]
            z_t = center_t_moved[0, 2]

            R = canon2world  # shape (3,3)
            T = center_t_moved.reshape(3, )
            S = np.array([l_s, w_s, h_s]).reshape(3, )

            pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
            if pc_in_bbox.shape[0] != pcs_world.shape[0]:
                print('part points: ', pcs_world.shape[0])
                print('points in box: ', pc_in_bbox.shape[0])
                print('Error! tight bbox failed: not all points are in the tight bbox!')
                exit(-1)

            return (tight_bbox_world_moved, (R, T, S))

        elif parent_joint_type == 'revolute':
            # 恢复button axis在static的位置
            parent_rotation_theta = parent_joint_qpos - joint_info['limit'][parent_joint_id][0]
            child_axis_end_point = child_axis_start_point + child_axis_direction_vector * 1
            world2static = (t.axangle2mat(
                parent_axis_direction_vector.reshape(-1).tolist(), parent_rotation_theta * -1)).T
            static2world = world2static.T
            child_axis_start_point_static = (child_axis_start_point -
                                             parent_axis_start_point) @ world2static + parent_axis_start_point
            child_axis_end_point_static = (child_axis_end_point -
                                           parent_axis_start_point) @ world2static + parent_axis_start_point
            child_axis_direction_vector_static = child_axis_end_point_static - child_axis_start_point_static
            child_axis_direction_vector_static = child_axis_direction_vector_static / np.linalg.norm(
                child_axis_direction_vector_static)

            # 读取part对应的.obj文件
            obj_list = urdf_ins['obj_name'][link_id]
            part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
            part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T)

            # 以axis方向向量的反向为 canon space +z，world space -z且与方向向量垂直为canon space +y，叉乘得canon space +x
            canon_in_world_z = (-1) * child_axis_direction_vector_static
            canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)
            # 判断part的z轴是否指向world space的+z或-z，都不是则取重力方向world space -z相关方向为+y
            if abs(canon_in_world_z[0, 2] - 1.0) > 1e-5 and abs(canon_in_world_z[0, 2] + 1.0) > 1e-5:
                canon_in_world_x = np.cross(np.array([0, 0, -1]), canon_in_world_z)
                canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
                canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
                canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
            # part的z轴指向world space的+z，则取world space +y为+y
            elif abs(canon_in_world_z[0, 2] - 1.0) <= 1e-5:
                canon_in_world_x = np.cross(np.array([0, 1, 0]), canon_in_world_z)
                canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
                canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
                canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
            # part的z轴指向world space的-z，则取world space -y为+y
            else:
                canon_in_world_x = np.cross(np.array([0, -1, 0]), canon_in_world_z)
                canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
                canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
                canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)

            # 得到sapien world space到canon space的rotation matrix
            canon_in_static = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
            static2parent = (t.axangle2mat(parent_axis_direction_vector.reshape(-1).tolist(), parent_rotation_theta)).T
            canon2world = canon_in_static @ static2parent
            world2canon = canon2world.T

            # 转换part到canon space，得到tight bbox和pose
            obj2parent = (t.axangle2mat(parent_axis_direction_vector.reshape(-1).tolist(), parent_joint_qpos)).T
            part_obj_pts = (part_obj_pts - parent_axis_start_point) @ obj2parent + parent_axis_start_point
            part_conon_pts = part_obj_pts @ world2canon

            # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
            pcs_in_original = pcs_world - (child_axis_direction_vector * child_joint_qpos)
            pcs_original_canon = pcs_in_original @ world2canon
            part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

            x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
            y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
            z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
            l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
            w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
            h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

            center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
            tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
            tight_bbox_world = tight_bbox_canon @ canon2world

            # 恢复part在sapien里的移动，通过joint的qpos
            center_t_moved = center_t + (child_axis_direction_vector * child_joint_qpos)
            tight_bbox_world_moved = tight_bbox_world + (child_axis_direction_vector * child_joint_qpos)
            x_t = center_t_moved[0, 0]
            y_t = center_t_moved[0, 1]
            z_t = center_t_moved[0, 2]

            R = canon2world  # shape (3,3)
            T = center_t_moved.reshape(3, )
            S = np.array([l_s, w_s, h_s]).reshape(3, )

            # line_z = draw_line(parent_axis_start_point[0, 0], parent_axis_start_point[0, 1], parent_axis_start_point[0,
            #                                                                                                          2],
            #                    parent_axis_direction_vector.reshape(3, ).tolist())

            # pcd1 = o3d.geometry.PointCloud()
            # pcd1.points = o3d.utility.Vector3dVector(part_obj_pts)
            # bbox1 = draw_bbox(tight_bbox_canon, 5)
            # pcd2 = o3d.geometry.PointCloud()
            # pcd2.points = o3d.utility.Vector3dVector(part_obj_pts_0)
            # bbox2 = draw_bbox(tight_bbox_world, 7)
            # pcd3 = o3d.geometry.PointCloud()
            # pcd3.points = o3d.utility.Vector3dVector(pcs_world)
            # bbox3 = draw_bbox(tight_bbox_world_moved, 9)
            # pcd4 = o3d.geometry.PointCloud()
            # pcd4.points = o3d.utility.Vector3dVector(np.array(line_z))
            # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, pcd4, coord_frame])

            pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
            if pc_in_bbox.shape[0] != pcs_world.shape[0]:
                print('part points: ', pcs_world.shape[0])
                print('points in box: ', pc_in_bbox.shape[0])
                print('Error! tight bbox failed: not all points are in the tight bbox!')
                exit(-1)

            return (tight_bbox_world_moved, (R, T, S))

        else:
            print('Error! not support this father joint type!')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(parent_joint_type)
            exit(-1)

    else:
        print('Error! for slider button, kinematic chain depth is greater than 2! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)


def get_hinge_door_pose(inst_name, pcs, scene, urdf_ins, link_pose_dict, joint_pose_dict, meta):
    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_info = urdf_ins['joint']
    link_name = inst_name.split(':')[0]
    if link_name == 'base':
        link_id = 0
    else:
        link_id = int(link_name.split('_')[-1]) + 1

    joint_id_list = []
    last_link_id = link_id
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links'] - 1):
            if joint_info['child'][
                    i] == last_link_id:  # !!! the joint's index should be the same as its child's index - 1
                joint_id_list.append(i)
                if joint_info['parent'][i] == 0:
                    end_flag = True
                    break
                else:
                    last_link_id = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    if len(joint_id_list) == 1:  # kinematic chain depth = 1
        joint_id = joint_id_list[0]
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        joint_qpos = joint_qpos_list[joint_id]  # !!! the joint's index should be the same as its child's index - 1

        # 获得当前button对应轴的xyz点和方向向量
        axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
        axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
        axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

        # 读取part对应的.obj文件
        obj_list = urdf_ins['obj_name'][link_id]
        part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
        part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                             )  # 从obj空间转移到sapien world space (利用base处的rpy？)
        obj2world = t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), joint_qpos).T
        part_obj_pts = (part_obj_pts - axis_start_point) @ obj2world + axis_start_point

        # 以axis方向向量为 canon space +x，门的正面朝向为 canon space +z，叉乘得从转轴指向门对边的方向为 canon space +y
        # ! 假设转动方向符合右手定则，门在static状态下正面朝向world space的-x方向
        canon_in_world_x = axis_direction_vector
        canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
        rotation_theta = joint_qpos - joint_info['limit'][joint_id][0]
        canon_in_world_z = np.cross(canon_in_world_x, np.array([-1, 0, 0])) * math.sin(rotation_theta) + np.array(
            [-1, 0, 0]) * math.cos(rotation_theta)
        canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)
        canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
        canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)

        # 得到sapien world space到canon space的rotation matrix
        canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
        world2canon = canon2world.T

        # 转换part到canon space，得到tight bbox和pose
        part_conon_pts = part_obj_pts @ world2canon
        axis_canon = canon_in_world_x @ world2canon
        # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
        pcs_in_original = pcs_world
        pcs_original_canon = pcs_in_original @ world2canon
        part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

        x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
        y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
        z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
        l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
        w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
        h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

        center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
        tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
        tight_bbox_world = tight_bbox_canon @ canon2world

        # 恢复转轴在canon space内的位置
        # new_z = canon_in_world_z @ world2canon
        # new_y = canon_in_world_y @ world2canon
        # new_x = canon_in_world_x @ world2canon

        # 恢复part在sapien里的移动，通过joint的qpos
        center_t_moved = center_t
        tight_bbox_world_moved = tight_bbox_world
        x_t = center_t_moved[0, 0]
        y_t = center_t_moved[0, 1]
        z_t = center_t_moved[0, 2]

        R = canon2world  # shape (3,3)
        T = center_t_moved.reshape(3, )
        S = np.array([l_s, w_s, h_s]).reshape(3, )

        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(part_conon_pts)
        # bbox1 = draw_bbox(tight_bbox_canon, 5)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(part_obj_pts)
        # bbox2 = draw_bbox(tight_bbox_world, 7)
        # pcd3 = o3d.geometry.PointCloud()
        # pcd3.points = o3d.utility.Vector3dVector(pcs_world)
        # bbox3 = draw_bbox(tight_bbox_world_moved, 9)
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcd1, bbox1, pcd3, bbox3, coord_frame])

        # pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
        # if pc_in_bbox.shape[0] != pcs_world.shape[0]:
        #     print('part points: ', pcs_world.shape[0])
        #     print('points in box: ', pc_in_bbox.shape[0])
        #     print('Error! tight bbox failed: not all points are in the tight bbox!')
        #     exit(-1)

        return (tight_bbox_world_moved, (R, T, S))

    else:
        print('Error! for hinge door, kinematic chain depth is greater than 1! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)


def get_slider_drawer_pose(inst_name, pcs, scene, urdf_ins, link_pose_dict, joint_pose_dict, meta):
    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_info = urdf_ins['joint']
    link_name = inst_name.split(':')[0]
    if link_name == 'base':
        link_id = 0
    else:
        link_id = int(link_name.split('_')[-1]) + 1

    joint_id_list = []
    last_link_id = link_id
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links'] - 1):
            if joint_info['child'][i] == last_link_id:  
                # !!! the joint's index should be the same as its child's index - 1
                joint_id_list.append(i)
                if joint_info['parent'][i] == 0:
                    end_flag = True
                    break
                else:
                    last_link_id = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    if len(joint_id_list) == 1:  # kinematic chain depth = 1
        joint_id = joint_id_list[0]
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        joint_qpos = joint_qpos_list[joint_id]  # !!! the joint's index should be the same as its child's index - 1

        # 获得当前button对应轴的xyz点和方向向量
        axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
        axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
        axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

        # 读取part对应的.obj文件
        obj_list = urdf_ins['obj_name'][link_id]
        part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
        part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                             )  # 从obj空间转移到sapien world space (利用base处的rpy？)

        # 以axis方向向量为 canon space +x，world space +z为 canon space +z，叉乘得 canon space +y
        canon_in_world_x = axis_direction_vector
        canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
        canon_in_world_y = np.cross(np.array([0, 0, 1]), canon_in_world_x)
        canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
        canon_in_world_z = np.cross(canon_in_world_x, canon_in_world_y)
        canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)

        # 得到sapien world space到canon space的rotation matrix
        canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
        world2canon = canon2world.T

        # 转换part到canon space，得到tight bbox和pose
        part_conon_pts = part_obj_pts @ world2canon
        axis_canon = canon_in_world_x @ world2canon
        # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
        pcs_in_original = pcs_world - (axis_direction_vector * joint_qpos)
        pcs_original_canon = pcs_in_original @ world2canon
        part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

        x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
        y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
        z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
        l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
        w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
        h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

        center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
        tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
        tight_bbox_world = tight_bbox_canon @ canon2world

        # 恢复转轴在canon space内的位置
        # new_z = canon_in_world_z @ world2canon
        # new_y = canon_in_world_y @ world2canon
        # new_x = canon_in_world_x @ world2canon

        # 恢复part在sapien里的移动，通过joint的qpos
        center_t_moved = center_t + (axis_direction_vector * joint_qpos)
        tight_bbox_world_moved = tight_bbox_world + (axis_direction_vector * joint_qpos)
        x_t = center_t_moved[0, 0]
        y_t = center_t_moved[0, 1]
        z_t = center_t_moved[0, 2]

        R = canon2world  # shape (3,3)
        T = center_t_moved.reshape(3, )
        S = np.array([l_s, w_s, h_s]).reshape(3, )

        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(part_conon_pts)
        # bbox1 = draw_bbox(tight_bbox_canon, 5)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(part_obj_pts)
        # bbox2 = draw_bbox(tight_bbox_world, 7)
        # pcd3 = o3d.geometry.PointCloud()
        # pcd3.points = o3d.utility.Vector3dVector(pcs_world)
        # bbox3 = draw_bbox(tight_bbox_world_moved, 9)
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcd1, bbox1, pcd3, bbox3, coord_frame])

        pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
        if pc_in_bbox.shape[0] != pcs_world.shape[0]:
            print('part points: ', pcs_world.shape[0])
            print('points in box: ', pc_in_bbox.shape[0])
            print('Error! tight bbox failed: not all points are in the tight bbox!')
            exit(-1)

        return (tight_bbox_world_moved, (R, T, S))

    else:
        print('Error! for slider drawer, kinematic chain depth is greater than 1! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)


def get_hinge_lid_pose(inst_name, pcs, scene, urdf_ins, link_pose_dict, joint_pose_dict, meta):
    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_info = urdf_ins['joint']
    link_name = inst_name.split(':')[0]
    if link_name == 'base':
        link_id = 0
    else:
        link_id = int(link_name.split('_')[-1]) + 1

    joint_id_list = []
    last_link_id = link_id
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links'] - 1):
            if joint_info['child'][
                    i] == last_link_id:  # !!! the joint's index should be the same as its child's index - 1
                joint_id_list.append(i)
                if joint_info['parent'][i] == 0:
                    end_flag = True
                    break
                else:
                    last_link_id = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    if len(joint_id_list) == 1:  # kinematic chain depth = 1
        joint_id = joint_id_list[0]
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        joint_qpos = joint_qpos_list[joint_id]  # !!! the joint's index should be the same as its child's index - 1

        # 获得当前button对应轴的xyz点和方向向量
        axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
        axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
        axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

        # 读取part对应的.obj文件
        obj_list = urdf_ins['obj_name'][link_id]
        part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
        part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                             )  # 从obj空间转移到sapien world space (利用base处的rpy？)
        obj2world = t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), joint_qpos).T
        part_obj_pts = (part_obj_pts - axis_start_point) @ obj2world + axis_start_point

        # 以axis方向向量为 canon space +x，lid的正面朝向为 canon space +z，叉乘得从转轴指向lid对边的方向为 canon space +y
        # ! 假设转动方向符合右手定则，lid在static状态下正面朝向world space的-x方向
        canon_in_world_x = axis_direction_vector
        canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
        rotation_theta = joint_qpos - joint_info['limit'][joint_id][0]
        canon_in_world_z = np.cross(canon_in_world_x, np.array([0, 0, 1])) * math.sin(rotation_theta) + np.array(
            [0, 0, 1]) * math.cos(rotation_theta)
        canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)
        canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
        canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)

        # 得到sapien world space到canon space的rotation matrix
        canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
        world2canon = canon2world.T

        # 转换part到canon space，得到tight bbox和pose
        part_conon_pts = part_obj_pts @ world2canon
        axis_canon = canon_in_world_x @ world2canon
        # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
        pcs_in_original = pcs_world
        pcs_original_canon = pcs_in_original @ world2canon
        part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

        x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
        y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
        z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
        l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
        w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
        h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

        center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
        tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
        tight_bbox_world = tight_bbox_canon @ canon2world

        # 恢复转轴在canon space内的位置
        # new_z = canon_in_world_z @ world2canon
        # new_y = canon_in_world_y @ world2canon
        # new_x = canon_in_world_x @ world2canon

        # 恢复part在sapien里的移动，通过joint的qpos
        center_t_moved = center_t
        tight_bbox_world_moved = tight_bbox_world
        x_t = center_t_moved[0, 0]
        y_t = center_t_moved[0, 1]
        z_t = center_t_moved[0, 2]

        R = canon2world  # shape (3,3)
        T = center_t_moved.reshape(3, )
        S = np.array([l_s, w_s, h_s]).reshape(3, )

        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(part_conon_pts)
        # bbox1 = draw_bbox(tight_bbox_canon, 5)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(part_obj_pts)
        # bbox2 = draw_bbox(tight_bbox_world, 7)
        # pcd3 = o3d.geometry.PointCloud()
        # pcd3.points = o3d.utility.Vector3dVector(pcs_world)
        # bbox3 = draw_bbox(tight_bbox_world_moved, 9)
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcd1, bbox1, pcd3, bbox3, coord_frame])

        # pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
        # if pc_in_bbox.shape[0] != pcs_world.shape[0]:
        #     print('part points: ', pcs_world.shape[0])
        #     print('points in box: ', pc_in_bbox.shape[0])
        #     print('Error! tight bbox failed: not all points are in the tight bbox!')
        #     print(_.shape[0])
        #     print(pcs_world.shape[0])
        #     error_point = 0
        #     for i in range(_.shape[0]):
        #         if not _[i]:
        #             error_point = i
        #     print(pcs_world[error_point])
        #     print(pcs_original_canon[error_point] @ canon2world)
        #     print(world2canon @ canon2world)
        #     exit(-1)

        return (tight_bbox_world_moved, (R, T, S))

    else:
        print('Error! for hinge lid, kinematic chain depth is greater than 1! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)


def get_slider_lid_pose(inst_name, pcs, scene, urdf_ins, link_pose_dict, joint_pose_dict, meta):
    object_category = meta['category']

    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_info = urdf_ins['joint']
    link_name = inst_name.split(':')[0]
    if link_name == 'base':
        link_id = 0
    else:
        link_id = int(link_name.split('_')[-1]) + 1

    joint_id_list = []
    last_link_id = link_id
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links'] - 1):
            if joint_info['child'][
                    i] == last_link_id:  # !!! the joint's index should be the same as its child's index - 1
                joint_id_list.append(i)
                if joint_info['parent'][i] == 0:
                    end_flag = True
                    break
                else:
                    last_link_id = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    if len(joint_id_list) == 1:  # kinematic chain depth = 1
        joint_id = joint_id_list[0]
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        joint_qpos = joint_qpos_list[joint_id]  # !!! the joint's index should be the same as its child's index - 1

        # 获得当前button对应轴的xyz点和方向向量
        axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
        axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
        axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

        # 读取part对应的.obj文件
        obj_list = urdf_ins['obj_name'][link_id]
        part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
        part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                             )  # 从obj空间转移到sapien world space (利用base处的rpy？)

        # 以axis方向向量为 canon space +z，world canon space -x 为canon space +x，叉乘得canon space +y
        canon_in_world_z = axis_direction_vector
        canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)
        canon_in_world_y = np.cross(canon_in_world_z, np.array([-1, 0, 0]))
        canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
        canon_in_world_x = np.cross(canon_in_world_y, canon_in_world_z)
        canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)

        # 得到sapien world space到canon space的rotation matrix
        canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
        world2canon = canon2world.T

        # 转换part到canon space，得到tight bbox和pose
        part_conon_pts = part_obj_pts @ world2canon
        axis_canon = canon_in_world_x @ world2canon
        # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
        pcs_in_original = pcs_world - (axis_direction_vector * joint_qpos)
        pcs_original_canon = pcs_in_original @ world2canon
        part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

        x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
        y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
        z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
        l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
        w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
        h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

        center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
        tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
        tight_bbox_world = tight_bbox_canon @ canon2world

        # 恢复转轴在canon space内的位置
        # new_z = canon_in_world_z @ world2canon
        # new_y = canon_in_world_y @ world2canon
        # new_x = canon_in_world_x @ world2canon

        # 恢复part在sapien里的移动，通过joint的qpos
        center_t_moved = center_t + (axis_direction_vector * joint_qpos)
        tight_bbox_world_moved = tight_bbox_world + (axis_direction_vector * joint_qpos)
        x_t = center_t_moved[0, 0]
        y_t = center_t_moved[0, 1]
        z_t = center_t_moved[0, 2]

        R = canon2world  # shape (3,3)
        T = center_t_moved.reshape(3, )
        S = np.array([l_s, w_s, h_s]).reshape(3, )

        pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
        if pc_in_bbox.shape[0] != pcs_world.shape[0]:
            print('part points: ', pcs_world.shape[0])
            print('points in box: ', pc_in_bbox.shape[0])
            print('Error! tight bbox failed: not all points are in the tight bbox!')
            exit(-1)

        return (tight_bbox_world_moved, (R, T, S))

    elif len(joint_id_list) == 2:  # kinematic chain depth = 2
        if object_category == 'Camera':  # 横着的lid，单标
            joint_id = joint_id_list[0]
            joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
            joint_qpos = joint_qpos_list[joint_id]  # !!! the joint's index should be the same as its child's index - 1

            # 获得当前button对应轴的xyz点和方向向量
            axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
            axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
            axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

            # 读取part对应的.obj文件
            obj_list = urdf_ins['obj_name'][link_id]
            part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
            part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                                 )  # 从obj空间转移到sapien world space (利用base处的rpy？)

            # 以axis方向向量为 canon space +z，world canon space +z为canon space +y，叉乘得 canon space +x
            canon_in_world_z = axis_direction_vector
            canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)
            canon_in_world_x = np.cross(np.array([0, 0, 1]), canon_in_world_z)
            canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)
            canon_in_world_y = np.cross(canon_in_world_z, canon_in_world_x)
            canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)

            # 得到sapien world space到canon space的rotation matrix
            canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
            world2canon = canon2world.T

            # 转换part到canon space，得到tight bbox和pose
            part_conon_pts = part_obj_pts @ world2canon
            axis_canon = canon_in_world_x @ world2canon
            # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
            pcs_in_original = pcs_world - (axis_direction_vector * joint_qpos)
            pcs_original_canon = pcs_in_original @ world2canon
            part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

            x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
            y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
            z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
            l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
            w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
            h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

            center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
            tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
            tight_bbox_world = tight_bbox_canon @ canon2world

            # 恢复转轴在canon space内的位置
            # new_z = canon_in_world_z @ world2canon
            # new_y = canon_in_world_y @ world2canon
            # new_x = canon_in_world_x @ world2canon

            # 恢复part在sapien里的移动，通过joint的qpos
            center_t_moved = center_t + (axis_direction_vector * joint_qpos)
            tight_bbox_world_moved = tight_bbox_world + (axis_direction_vector * joint_qpos)
            x_t = center_t_moved[0, 0]
            y_t = center_t_moved[0, 1]
            z_t = center_t_moved[0, 2]

            R = canon2world  # shape (3,3)
            T = center_t_moved.reshape(3, )
            S = np.array([l_s, w_s, h_s]).reshape(3, )

            pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
            if pc_in_bbox.shape[0] != pcs_world.shape[0]:
                print('part points: ', pcs_world.shape[0])
                print('points in box: ', pc_in_bbox.shape[0])
                print('Error! tight bbox failed: not all points are in the tight bbox!')
                exit(-1)

            return (tight_bbox_world_moved, (R, T, S))

        else:  # CoffeeMachine
            child_joint_id = joint_id_list[0]
            # 获得当前button对应轴的xyz点和方向向量 轴在world space中，叠加了parent joint的运动，需要先剔除
            child_axis_start_point = np.array(joint_pose_dict[child_joint_id]['xyz']).reshape(1, 3)
            child_axis_direction_vector = np.array(joint_pose_dict[child_joint_id]['axis']).reshape(1, 3)
            child_axis_direction_vector = child_axis_direction_vector / np.linalg.norm(child_axis_direction_vector)

            parent_joint_id = joint_id_list[1]
            # 获得当前button的parent joint对应轴的xyz点和方向向量 轴在world space中
            parent_axis_start_point = np.array(joint_pose_dict[parent_joint_id]['xyz']).reshape(1, 3)
            parent_axis_direction_vector = np.array(joint_pose_dict[parent_joint_id]['axis']).reshape(1, 3)
            parent_axis_direction_vector = parent_axis_direction_vector / np.linalg.norm(parent_axis_direction_vector)

            # 获得joint的qpos
            joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
            child_joint_qpos = joint_qpos_list[
                child_joint_id]  # !!! the joint's index should be the same as its child's index - 1
            parent_joint_qpos = joint_qpos_list[parent_joint_id]

            parent_joint_type = joint_info['type'][parent_joint_id]
            if parent_joint_type == 'prismatic':
                child_axis_start_point_static = child_axis_start_point - (parent_axis_direction_vector *
                                                                          parent_joint_qpos)
                child_axis_direction_vector_static = child_axis_direction_vector
                child_axis_direction_vector_static = child_axis_direction_vector_static / np.linalg.norm(
                    child_axis_direction_vector_static)

                # 读取part对应的.obj文件
                obj_list = urdf_ins['obj_name'][link_id]
                part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
                part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                                     )  # 从obj空间转移到sapien world space (利用base处的rpy？)

                # 以axis方向向量为 canon space +z，world canon space -x 为canon space +x，叉乘得canon space +y
                canon_in_world_z = child_axis_direction_vector
                canon_in_world_z = canon_in_world_z / np.linalg.norm(canon_in_world_z)
                canon_in_world_y = np.cross(canon_in_world_z, np.array([-1, 0, 0]))
                canon_in_world_y = canon_in_world_y / np.linalg.norm(canon_in_world_y)
                canon_in_world_x = np.cross(canon_in_world_y, canon_in_world_z)
                canon_in_world_x = canon_in_world_x / np.linalg.norm(canon_in_world_x)

                # 得到sapien world space到canon space的rotation matrix
                canon2world = np.vstack((canon_in_world_x, canon_in_world_y, canon_in_world_z))
                world2canon = canon2world.T

                # 转换part到canon space，得到tight bbox和pose
                part_conon_pts = part_obj_pts @ world2canon
                axis_canon = canon_in_world_z @ world2canon
                # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
                pcs_in_original = pcs_world - (child_axis_direction_vector * child_joint_qpos)
                pcs_in_original = pcs_in_original - (parent_axis_direction_vector * parent_joint_qpos)
                pcs_original_canon = pcs_in_original @ world2canon
                part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

                x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
                y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
                z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
                l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
                w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
                h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

                center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
                tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
                tight_bbox_world = tight_bbox_canon @ canon2world

                # 恢复part在sapien里的移动，通过joint的qpos
                center_t_moved = center_t + (parent_axis_direction_vector * parent_joint_qpos)
                center_t_moved = center_t_moved + (child_axis_direction_vector * child_joint_qpos)
                tight_bbox_world_moved = tight_bbox_world + (parent_axis_direction_vector * parent_joint_qpos)
                tight_bbox_world_moved = tight_bbox_world_moved + (child_axis_direction_vector * child_joint_qpos)
                x_t = center_t_moved[0, 0]
                y_t = center_t_moved[0, 1]
                z_t = center_t_moved[0, 2]

                R = canon2world  # shape (3,3)
                T = center_t_moved.reshape(3, )
                S = np.array([l_s, w_s, h_s]).reshape(3, )

                pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
                if pc_in_bbox.shape[0] != pcs_world.shape[0]:
                    print('part points: ', pcs_world.shape[0])
                    print('points in box: ', pc_in_bbox.shape[0])
                    print('Error! tight bbox failed: not all points are in the tight bbox!')
                    exit(-1)

                return (tight_bbox_world_moved, (R, T, S))

            else:
                print('Error! not supported joint type')
                print(meta['category'], meta['model_id'])
                print(inst_name)
                print(parent_joint_type)
                exit(-1)

    else:
        print('Error! for slider lid, kinematic chain depth is greater than 1! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)


def get_hinge_knob_pose(inst_name, pcs, scene, urdf_ins, link_pose_dict, joint_pose_dict, meta):
    object_category = meta['category']

    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_info = urdf_ins['joint']
    link_name = inst_name.split(':')[0]
    if link_name == 'base':
        link_id = 0
    else:
        link_id = int(link_name.split('_')[-1]) + 1

    joint_id_list = []
    last_link_id = link_id
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links'] - 1):
            if joint_info['child'][
                    i] == last_link_id:  # !!! the joint's index should be the same as its child's index - 1
                joint_id_list.append(i)
                if joint_info['parent'][i] == 0:
                    end_flag = True
                    break
                else:
                    last_link_id = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    if len(joint_id_list) == 1:  # kinematic chain depth = 1
        joint_id = joint_id_list[0]
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        joint_qpos = joint_qpos_list[joint_id]  # !!! the joint's index should be the same as its child's index - 1

        # 获得当前button对应轴的xyz点和方向向量
        axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
        axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
        axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

        # 读取part对应的.obj文件
        obj_list = urdf_ins['obj_name'][link_id]
        part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
        part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                             )  # 从obj空间转移到sapien world space (利用base处的rpy？)

        # 以axis方向向量的反向为 canon space +z，world space -z且与方向向量垂直为canon space +y，叉乘得canon space +x
        canon_in_static_z = axis_direction_vector
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        # 判断part的z轴是否指向world space的+z或-z，都不是则取重力方向world space -z相关方向为+y
        if abs(canon_in_static_z[0, 2] - 1.0) > 1e-5 and abs(canon_in_static_z[0, 2] + 1.0) > 1e-5:
            canon_in_static_x = np.cross(np.array([0, 0, -1]), canon_in_static_z)
            canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
            canon_in_static_y = np.cross(canon_in_static_z, canon_in_static_x)
            canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
        # part的z轴指向world space的+z，则取world space +y为+y
        elif abs(canon_in_static_z[0, 2] - 1.0) <= 1e-5:
            canon_in_static_x = np.cross(np.array([0, 1, 0]), canon_in_static_z)
            canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
            canon_in_static_y = np.cross(canon_in_static_z, canon_in_static_x)
            canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
        # part的z轴指向world space的-z，则取world space -y为+y
        else:
            canon_in_static_x = np.cross(np.array([0, -1, 0]), canon_in_static_z)
            canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
            canon_in_static_y = np.cross(canon_in_static_z, canon_in_static_x)
            canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)

        # 得到sapien world space到canon space的rotation matrix
        canon_in_static = np.vstack((canon_in_static_x, canon_in_static_y, canon_in_static_z))
        rotation_theta = joint_qpos - joint_info['limit'][joint_id][0]
        static2world = (t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), rotation_theta)).T
        canon2world = canon_in_static @ static2world
        world2canon = canon2world.T

        # 转换part到canon space，得到tight bbox和pose
        obj2world = (t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), joint_qpos)).T
        part_obj_pts = (part_obj_pts - axis_start_point) @ obj2world + axis_start_point
        part_conon_pts = part_obj_pts @ world2canon
        axis_canon = canon2world[2, :] @ world2canon
        # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
        pcs_in_original = pcs_world
        pcs_original_canon = pcs_in_original @ world2canon
        part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

        x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
        y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
        z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
        l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
        w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
        h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

        center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
        tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
        tight_bbox_world = tight_bbox_canon @ canon2world

        # 恢复转轴在canon space内的位置
        # new_z = canon_in_world_z @ world2canon
        # new_y = canon_in_world_y @ world2canon
        # new_x = canon_in_world_x @ world2canon

        # 恢复part在sapien里的移动，通过joint的qpos
        center_t_moved = center_t
        tight_bbox_world_moved = tight_bbox_world
        x_t = center_t_moved[0, 0]
        y_t = center_t_moved[0, 1]
        z_t = center_t_moved[0, 2]

        R = canon2world  # shape (3,3)
        T = center_t_moved.reshape(3, )
        S = np.array([l_s, w_s, h_s]).reshape(3, )

        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(part_conon_pts)
        # bbox1 = draw_bbox(tight_bbox_canon, 5)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(part_obj_pts)
        # bbox2 = draw_bbox(tight_bbox_world, 7)
        # pcd3 = o3d.geometry.PointCloud()
        # pcd3.points = o3d.utility.Vector3dVector(pcs_world)
        # bbox3 = draw_bbox(tight_bbox_world_moved, 9)
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcd1, bbox1, pcd3, bbox3, coord_frame])

        pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
        if pc_in_bbox.shape[0] != pcs_world.shape[0]:
            print('part points: ', pcs_world.shape[0])
            print('points in box: ', pc_in_bbox.shape[0])
            print('Error! tight bbox failed: not all points are in the tight bbox!')
            exit(-1)

        return (tight_bbox_world_moved, (R, T, S))

        # axis_vector = joint_pose_dict[joint_id]['axis']
        # print(axis_start_point)
        # print(axis_vector)
        # line_z = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
        #                    new_z.reshape(3, ).tolist())
        # line_x = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
        #                    new_x.reshape(3, ).tolist())
        # line_y = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
        #                    new_y.reshape(3, ).tolist())
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcs_world)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(np.array(line_z))
        # pcd3 = o3d.geometry.PointCloud()
        # pcd3.points = o3d.utility.Vector3dVector(part_obj_pts)
        # pcd4 = o3d.geometry.PointCloud()
        # pcd4.points = o3d.utility.Vector3dVector(np.array(line_x))
        # pcd5 = o3d.geometry.PointCloud()
        # pcd5.points = o3d.utility.Vector3dVector(np.array(line_y))
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcd, pcd2, pcd3, pcd4, pcd5, coord_frame])

        # return tight_bbox_world_moved

    elif len(joint_id_list) == 2:  # kinematic chain depth = 2
        child_joint_id = joint_id_list[0]
        # 获得当前button对应轴的xyz点和方向向量 轴在world space中，叠加了parent joint的运动，需要先剔除
        child_axis_start_point = np.array(joint_pose_dict[child_joint_id]['xyz']).reshape(1, 3)
        child_axis_direction_vector = np.array(joint_pose_dict[child_joint_id]['axis']).reshape(1, 3)
        child_axis_direction_vector = child_axis_direction_vector / np.linalg.norm(child_axis_direction_vector)

        parent_joint_id = joint_id_list[1]
        # 获得当前button的parent joint对应轴的xyz点和方向向量 轴在world space中
        parent_axis_start_point = np.array(joint_pose_dict[parent_joint_id]['xyz']).reshape(1, 3)
        parent_axis_direction_vector = np.array(joint_pose_dict[parent_joint_id]['axis']).reshape(1, 3)
        parent_axis_direction_vector = parent_axis_direction_vector / np.linalg.norm(parent_axis_direction_vector)

        # 获得joint的qpos
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        child_joint_qpos = joint_qpos_list[
            child_joint_id]  # !!! the joint's index should be the same as its child's index - 1
        parent_joint_qpos = joint_qpos_list[parent_joint_id]

        parent_joint_type = joint_info['type'][parent_joint_id]
        if parent_joint_type == 'revolute':
            # 恢复knob axis在static的位置
            parent_rotation_theta = parent_joint_qpos - joint_info['limit'][parent_joint_id][0]
            child_axis_end_point = child_axis_start_point + child_axis_direction_vector * 1
            world2static = (t.axangle2mat(
                parent_axis_direction_vector.reshape(-1).tolist(), parent_rotation_theta * -1)).T
            static2world = world2static.T
            child_axis_start_point_static = (child_axis_start_point -
                                             parent_axis_start_point) @ world2static + parent_axis_start_point
            child_axis_end_point_static = (child_axis_end_point -
                                           parent_axis_start_point) @ world2static + parent_axis_start_point
            child_axis_direction_vector_static = child_axis_end_point_static - child_axis_start_point_static
            child_axis_direction_vector_static = child_axis_direction_vector_static / np.linalg.norm(
                child_axis_direction_vector_static)

            # 读取part对应的.obj文件
            obj_list = urdf_ins['obj_name'][link_id]
            part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
            part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T)

            # 以axis方向向量的反向为 canon space +z，world space -z且与方向向量垂直为canon space +y，叉乘得canon space +x
            canon_in_static_z = child_axis_direction_vector_static
            canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
            # 判断part的z轴是否指向world space的+z或-z，都不是则取重力方向world space -z相关方向为+y
            if abs(canon_in_static_z[0, 2] - 1.0) > 1e-5 and abs(canon_in_static_z[0, 2] + 1.0) > 1e-5:
                canon_in_static_x = np.cross(np.array([0, 0, -1]), canon_in_static_z)
                canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
                canon_in_static_y = np.cross(canon_in_static_z, canon_in_static_x)
                canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
            # part的z轴指向world space的+z，则取world space +y为+y
            elif abs(canon_in_static_z[0, 2] - 1.0) <= 1e-5:
                canon_in_static_x = np.cross(np.array([0, 1, 0]), canon_in_static_z)
                canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
                canon_in_static_y = np.cross(canon_in_static_z, canon_in_static_x)
                canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
            # part的z轴指向world space的-z，则取world space -y为+y
            else:
                canon_in_static_x = np.cross(np.array([0, -1, 0]), canon_in_static_z)
                canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
                canon_in_static_y = np.cross(canon_in_static_z, canon_in_static_x)
                canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)

            # 得到sapien world space到canon space的rotation matrix
            canon_in_static = np.vstack((canon_in_static_x, canon_in_static_y, canon_in_static_z))
            static2parent = (t.axangle2mat(parent_axis_direction_vector.reshape(-1).tolist(), parent_rotation_theta)).T
            parent2world = (t.axangle2mat(child_axis_direction_vector.reshape(-1).tolist(), child_joint_qpos)).T
            canon2world = canon_in_static @ static2parent @ parent2world
            world2canon = canon2world.T

            # 转换part到canon space，得到tight bbox和pose
            obj2parent = (t.axangle2mat(parent_axis_direction_vector.reshape(-1).tolist(), parent_joint_qpos)).T
            part_obj_pts = (part_obj_pts - parent_axis_start_point) @ obj2parent + parent_axis_start_point
            part_obj_pts = (part_obj_pts - child_axis_start_point) @ parent2world + child_axis_start_point
            part_conon_pts = part_obj_pts @ world2canon
            axis_canon = canon2world[2, :] @ world2canon
            # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
            pcs_in_original = pcs_world
            pcs_original_canon = pcs_in_original @ world2canon
            part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

            x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
            y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
            z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
            l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
            w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
            h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

            center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
            tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
            tight_bbox_world = tight_bbox_canon @ canon2world

            # 恢复part在sapien里的移动，通过joint的qpos
            center_t_moved = center_t
            tight_bbox_world_moved = tight_bbox_world
            x_t = center_t_moved[0, 0]
            y_t = center_t_moved[0, 1]
            z_t = center_t_moved[0, 2]

            # axis_vector = joint_pose_dict[joint_id]['axis']
            # print(axis_start_point)
            # print(axis_vector)
            # line_z = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
            #                    new_z.reshape(3, ).tolist())
            # line_x = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
            #                    new_x.reshape(3, ).tolist())
            # line_y = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
            #                    new_y.reshape(3, ).tolist())
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pcs_world)
            # pcd2 = o3d.geometry.PointCloud()
            # pcd2.points = o3d.utility.Vector3dVector(tight_bbox_canon)
            # obj_list = urdf_ins['obj_name'][link_id]
            # part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
            # part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T)
            # pcd3 = o3d.geometry.PointCloud()
            # pcd3.points = o3d.utility.Vector3dVector(part_obj_pts)
            # pcd4 = o3d.geometry.PointCloud()
            # pcd4.points = o3d.utility.Vector3dVector(tight_bbox_world)
            # pcd5 = o3d.geometry.PointCloud()
            # pcd5.points = o3d.utility.Vector3dVector(np.array(center_t))
            # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # o3d.visualization.draw_geometries([pcd4, pcd5, coord_frame])

            R = canon2world  # shape (3,3)
            T = center_t_moved.reshape(3, )
            S = np.array([l_s, w_s, h_s]).reshape(3, )

            pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
            if pc_in_bbox.shape[0] != pcs_world.shape[0]:
                print('part points: ', pcs_world.shape[0])
                print('points in box: ', pc_in_bbox.shape[0])
                print('Error! tight bbox failed: not all points are in the tight bbox!')
                exit(-1)

            return (tight_bbox_world_moved, (R, T, S))

        else:
            print('Error! hinge knob not support this father joint type!')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(parent_joint_type)
            exit(-1)

    else:
        print('Error! for hinge knob, kinematic chain depth is greater than 2! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)


# TODO 没有修改，不能用来获得NOCS的三个参数
def get_fixed_handle_pose_old(inst_name, pcs, scene, urdf_ins, link_pose_dict, joint_pose_dict, anno_config, meta):
    fixed_handle_list = urdf_ins['fixed_handle_ins']
    handle_objs = None
    link_name = inst_name.split('/')[0].split(':')[0]
    if link_name == 'base':
        parent_link_id = 0
    else:
        parent_link_id = int(link_name.split('_')[1]) + 1
    for handle_tuple in fixed_handle_list[parent_link_id]:
        _, part_name, objs = handle_tuple
        if part_name == inst_name:
            handle_objs = objs
            break
    assert handle_objs != None, "Error! not find match fixed handle objs!"

    object_category = meta['category']
    model_id = meta['model_id']
    anno_scheme_index = anno_config[object_category][str(model_id)]

    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_info = urdf_ins['joint']
    link_name = inst_name.split(':')[0]

    joint_id_list = []
    last_link_id = parent_link_id
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links'] - 1):
            if joint_info['child'][
                    i] == last_link_id:  # !!! the joint's index should be the same as its child's index - 1
                joint_id_list.append(i)
                if joint_info['parent'][i] == 0:
                    end_flag = True
                    break
                else:
                    last_link_id = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    if len(joint_id_list) == 0:
        pass
    elif len(joint_id_list) == 1:
        parent_joint_id = joint_id_list[0]
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        joint_qpos = joint_qpos_list[
            parent_joint_id]  # !!! the joint's index should be the same as its child's index - 1

        # 获得当前button对应轴的xyz点和方向向量
        axis_start_point = np.array(joint_pose_dict[parent_joint_id]['xyz']).reshape(1, 3)
        axis_direction_vector = np.array(joint_pose_dict[parent_joint_id]['axis']).reshape(1, 3)
        axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

        # 读取part对应的.obj文件
        obj_list = handle_objs
        part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
        part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                             )  # 从obj空间转移到sapien world space (利用base处的rpy？)

        if object_category == 'StorageFurniture' and anno_scheme_index == 2:
            if joint_info['type'][parent_joint_id] == 'revolute':
                tag = 'line'
                # 设置static下的pose，再旋转到world得到world下的pose
                # 横着的、把手朝-x方向的handle，以world space -x方向向量为 canon space +z，world space +z方向向量为 canon space +y，叉乘得 canon space +x(沿长边)
                canon_in_static_z = np.array([-1, 0, 0])
                canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
                canon_in_static_y = np.array([0, 0, 1])
                canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
                canon_in_static_x = np.cross(canon_in_static_y, canon_in_static_z)
                canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)

                # 得到sapien world space到canon space的rotation matrix
                canon_in_static = np.vstack((canon_in_static_x, canon_in_static_y, canon_in_static_z))
                rotation_theta = joint_qpos - joint_info['limit'][parent_joint_id][0]
                static2world = (t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), rotation_theta)).T
                canon2world = canon_in_static @ static2world
                world2canon = canon2world.T

                # line_z = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
                #                    canon2world[2].reshape(3, ).tolist())
                # line_x = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
                #                    canon2world[0].reshape(3, ).tolist())
                # line_y = draw_line(axis_start_point[0, 0], axis_start_point[0, 1], axis_start_point[0, 2],
                #                    canon2world[1].reshape(3, ).tolist())
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pcs_world)
                # pcd2 = o3d.geometry.PointCloud()
                # pcd2.points = o3d.utility.Vector3dVector(np.array(line_z))
                # pcd3 = o3d.geometry.PointCloud()
                # pcd3.points = o3d.utility.Vector3dVector(part_obj_pts)
                # pcd4 = o3d.geometry.PointCloud()
                # pcd4.points = o3d.utility.Vector3dVector(np.array(line_x))
                # pcd5 = o3d.geometry.PointCloud()
                # pcd5.points = o3d.utility.Vector3dVector(np.array(line_y))
                # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
                # o3d.visualization.draw_geometries([pcd, pcd2, pcd3, pcd4, pcd5, coord_frame])

                # 转换part到canon space，得到tight bbox和pose
                obj2world = (t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), joint_qpos)).T
                part_obj_pts = (part_obj_pts - axis_start_point) @ obj2world + axis_start_point
                part_conon_pts = part_obj_pts @ world2canon
                axis_canon = canon2world[2, :] @ world2canon
                # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题

                pcs_in_original = pcs_world
                pcs_original_canon = pcs_in_original @ world2canon
                part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

                x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
                y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
                z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
                l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
                w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
                h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

                center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
                tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
                tight_bbox_world = tight_bbox_canon @ canon2world

                # 恢复part在sapien里的移动，通过joint的qpos
                center_t_moved = center_t
                tight_bbox_world_moved = tight_bbox_world
                x_t = center_t_moved[0, 0]
                y_t = center_t_moved[0, 1]
                z_t = center_t_moved[0, 2]

                r = canon2world  # shape (3,3)

                pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
                if pc_in_bbox.shape[0] != pcs_world.shape[0]:
                    print('part points: ', pcs_world.shape[0])
                    print('points in box: ', pc_in_bbox.shape[0])
                    print('Error! tight bbox failed: not all points are in the tight bbox!')
                    exit(-1)

                return tight_bbox_world_moved, tag

            elif joint_info['type'][parent_joint_id] == 'prismatic':
                tag = 'line'
                # 设置static下的pose，再旋转到world得到world下的pose
                # 横着的、把手朝-x方向的handle，以world space -x方向向量为 canon space +z，world space +z方向向量为 canon space +y，叉乘得 canon space +x(沿长边)
                canon_in_static_z = np.array([-1, 0, 0])
                canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
                canon_in_static_y = np.array([0, 0, 1])
                canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
                canon_in_static_x = np.cross(canon_in_static_y, canon_in_static_z)
                canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)

                # 得到sapien world space到canon space的rotation matrix
                canon_in_static = np.vstack((canon_in_static_x, canon_in_static_y, canon_in_static_z))
                canon2world = canon_in_static
                world2canon = canon2world.T

                # 转换part到canon space，得到tight bbox和pose
                part_conon_pts = part_obj_pts @ world2canon
                axis_canon = canon2world[2, :] @ world2canon
                # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题

                pcs_in_original = pcs_world - (axis_direction_vector * joint_qpos)
                pcs_original_canon = pcs_in_original @ world2canon
                part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

                x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
                y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
                z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
                l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
                w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
                h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

                center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
                tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
                tight_bbox_world = tight_bbox_canon @ canon2world

                # 恢复转轴在canon space内的位置
                # new_z = canon_in_world_z @ world2canon
                # new_y = canon_in_world_y @ world2canon
                # new_x = canon_in_world_x @ world2canon

                # 恢复part在sapien里的移动，通过joint的qpos
                center_t_moved = center_t + (axis_direction_vector * joint_qpos)
                tight_bbox_world_moved = tight_bbox_world + (axis_direction_vector * joint_qpos)
                x_t = center_t_moved[0, 0]
                y_t = center_t_moved[0, 1]
                z_t = center_t_moved[0, 2]

                r = canon2world  # shape (3,3)

                # pcd1 = o3d.geometry.PointCloud()
                # pcd1.points = o3d.utility.Vector3dVector(part_conon_pts)
                # bbox1 = draw_bbox(tight_bbox_canon, 5)
                # pcd2 = o3d.geometry.PointCloud()
                # pcd2.points = o3d.utility.Vector3dVector(part_obj_pts)
                # bbox2 = draw_bbox(tight_bbox_world, 7)
                # pcd3 = o3d.geometry.PointCloud()
                # pcd3.points = o3d.utility.Vector3dVector(pcs_world)
                # bbox3 = draw_bbox(tight_bbox_world_moved, 9)
                # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
                # o3d.visualization.draw_geometries([pcd1, pcd2, bbox1, pcd3, bbox3, coord_frame])

                pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
                if pc_in_bbox.shape[0] != pcs_world.shape[0]:
                    print('part points: ', pcs_world.shape[0])
                    print('points in box: ', pc_in_bbox.shape[0])
                    print('Error! tight bbox failed: not all points are in the tight bbox!')
                    exit(-1)

                return tight_bbox_world_moved, tag

            else:
                print('Error! not supported joint type')
                print(meta['category'], meta['model_id'])
                print(inst_name)
                print(joint_info['type'][parent_joint_id])
                exit(-1)

    else:
        print('Error! for fixed handle, kinematic chain depth is greater than 1! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)

    return None, None


def get_y_crd_sign(pcs_world):
    N = pcs_world.shape[0]
    y = pcs_world[:, 1]
    res = np.sum(y > 0) / N
    if res == 1:
        return 1
    elif res == 0:
        return -1
    else:
        print('Error! part point cloud are not in one side of xOz plane!')
        print(res)
        exit(-1)


def get_canon_in_static_frame(frame_num):
    if frame_num == 1:
        canon_in_static_z = np.array([-1, 0, 0])
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        canon_in_static_y = np.array([0, 0, 1])
        canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
        canon_in_static_x = np.cross(canon_in_static_y, canon_in_static_z)
        canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
    elif frame_num == 2:
        canon_in_static_z = np.array([-1, 0, 0])
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        canon_in_static_x = np.array([0, 0, 1])
        canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
        canon_in_static_y = np.cross(canon_in_static_z, canon_in_static_x)
        canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
    elif frame_num == 3:
        canon_in_static_y = np.array([-1, 0, 0])
        canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
        canon_in_static_z = np.array([0, 0, 1])
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        canon_in_static_x = np.cross(canon_in_static_y, canon_in_static_z)
        canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
    elif frame_num == 4:
        canon_in_static_z = np.array([-1, 0, 0])
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        canon_in_static_y = np.array([0, 0, 1])
        canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
        canon_in_static_x = np.cross(canon_in_static_y, canon_in_static_z)
        canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
    elif frame_num == 5:
        canon_in_static_x = np.array([-1, 0, 0])
        canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
        canon_in_static_z = np.array([0, 0, 1])
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        canon_in_static_y = np.cross(canon_in_static_z, canon_in_static_x)
        canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
    elif frame_num == 6:
        canon_in_static_z = np.array([0, 1, 0])
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        canon_in_static_y = np.array([0, 0, 1])
        canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
        canon_in_static_x = np.cross(canon_in_static_y, canon_in_static_z)
        canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
    elif frame_num == 7:
        canon_in_static_z = np.array([0, -1, 0])
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        canon_in_static_y = np.array([0, 0, 1])
        canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
        canon_in_static_x = np.cross(canon_in_static_y, canon_in_static_z)
        canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
    elif frame_num == 8:
        canon_in_static_z = np.array([0, 1, 0])
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        canon_in_static_x = np.array([0, 0, 1])
        canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
        canon_in_static_y = np.cross(canon_in_static_z, canon_in_static_x)
        canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
    elif frame_num == 9:
        canon_in_static_z = np.array([0, -1, 0])
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        canon_in_static_x = np.array([0, 0, 1])
        canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
        canon_in_static_y = np.cross(canon_in_static_z, canon_in_static_x)
        canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
    elif frame_num == 10:
        canon_in_static_y = np.array([0, -1, 0])
        canon_in_static_y = canon_in_static_y / np.linalg.norm(canon_in_static_y)
        canon_in_static_z = np.array([0, 0, 1])
        canon_in_static_z = canon_in_static_z / np.linalg.norm(canon_in_static_z)
        canon_in_static_x = np.cross(canon_in_static_y, canon_in_static_z)
        canon_in_static_x = canon_in_static_x / np.linalg.norm(canon_in_static_x)
    else:
        print('Error! wrong frame number!')
        print(frame_num)
        exit(-1)

    canon_in_static = np.vstack((canon_in_static_x, canon_in_static_y, canon_in_static_z))
    return canon_in_static


def get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, frame_num, meta):
    # get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
    joint_qpos = joint_qpos_list[parent_joint_id]  # !!! the joint's index should be the same as its child's index - 1

    # 获得当前button对应轴的xyz点和方向向量
    axis_start_point = np.array(joint_pose_dict[parent_joint_id]['xyz']).reshape(1, 3)
    axis_direction_vector = np.array(joint_pose_dict[parent_joint_id]['axis']).reshape(1, 3)
    axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

    # 读取part对应的.obj文件
    obj_list = handle_objs
    part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
    part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                         )  # 从obj空间转移到sapien world space (利用base处的rpy？)

    # 设置static下的pose，prismatic下直接得到world下的pose
    canon_in_static = get_canon_in_static_frame(frame_num)
    canon2world = canon_in_static
    world2canon = canon2world.T

    # 转换part到canon space，得到tight bbox和pose
    part_conon_pts = part_obj_pts @ world2canon

    # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
    pcs_in_original = pcs_world - (axis_direction_vector * joint_qpos)
    pcs_original_canon = pcs_in_original @ world2canon
    part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

    x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
    y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
    z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
    l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
    w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
    h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

    center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
    tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
    tight_bbox_world = tight_bbox_canon @ canon2world

    # 恢复part在sapien里的移动，通过joint的qpos
    center_t_moved = center_t + (axis_direction_vector * joint_qpos)
    tight_bbox_world_moved = tight_bbox_world + (axis_direction_vector * joint_qpos)
    x_t = center_t_moved[0, 0]
    y_t = center_t_moved[0, 1]
    z_t = center_t_moved[0, 2]

    R = canon2world  # shape (3,3)
    T = center_t_moved.reshape(3, )
    S = np.array([l_s, w_s, h_s]).reshape(3, )

    pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
    if pc_in_bbox.shape[0] != pcs_world.shape[0]:
        print('part points: ', pcs_world.shape[0])
        print('points in box: ', pc_in_bbox.shape[0])
        print('Error! tight bbox failed: not all points are in the tight bbox!')
        exit(-1)

    return (tight_bbox_world_moved, (R, T, S))


def get_prismatic_based_handle_depth2(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id,
                                      grandparent_joint_id, frame_num, meta):
    # get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
    parent_joint_qpos = joint_qpos_list[
        parent_joint_id]  # !!! the joint's index should be the same as its child's index - 1
    grandparent_joint_qpos = joint_qpos_list[grandparent_joint_id]

    # 获得当前button对应轴的xyz点和方向向量
    parent_axis_start_point = np.array(joint_pose_dict[parent_joint_id]['xyz']).reshape(1, 3)
    parent_axis_direction_vector = np.array(joint_pose_dict[parent_joint_id]['axis']).reshape(1, 3)
    parent_axis_direction_vector = parent_axis_direction_vector / np.linalg.norm(parent_axis_direction_vector)

    grandparent_axis_start_point = np.array(joint_pose_dict[grandparent_joint_id]['xyz']).reshape(1, 3)
    grandparent_axis_direction_vector = np.array(joint_pose_dict[grandparent_joint_id]['axis']).reshape(1, 3)
    grandparent_axis_direction_vector = grandparent_axis_direction_vector / np.linalg.norm(
        grandparent_axis_direction_vector)

    # 读取part对应的.obj文件
    obj_list = handle_objs
    part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
    part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                         )  # 从obj空间转移到sapien world space (利用base处的rpy？)

    # 设置static下的pose，prismatic下直接得到world下的pose
    canon_in_static = get_canon_in_static_frame(frame_num)
    canon2world = canon_in_static
    world2canon = canon2world.T

    # 转换part到canon space，得到tight bbox和pose
    part_conon_pts = part_obj_pts @ world2canon

    # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
    pcs_in_original = pcs_world - (parent_axis_direction_vector *
                                   parent_joint_qpos) - (grandparent_axis_direction_vector * grandparent_joint_qpos)
    pcs_original_canon = pcs_in_original @ world2canon
    part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

    x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
    y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
    z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
    l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
    w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
    h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

    center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
    tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
    tight_bbox_world = tight_bbox_canon @ canon2world

    # 恢复part在sapien里的移动，通过joint的qpos
    center_t_moved = center_t + (parent_axis_direction_vector *
                                 parent_joint_qpos) + (grandparent_axis_direction_vector * grandparent_joint_qpos)
    tight_bbox_world_moved = tight_bbox_world + (parent_axis_direction_vector * parent_joint_qpos) + (
        grandparent_axis_direction_vector * grandparent_joint_qpos)
    x_t = center_t_moved[0, 0]
    y_t = center_t_moved[0, 1]
    z_t = center_t_moved[0, 2]

    R = canon2world  # shape (3,3)
    T = center_t_moved.reshape(3, )
    S = np.array([l_s, w_s, h_s]).reshape(3, )

    pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
    if pc_in_bbox.shape[0] != pcs_world.shape[0]:
        print('part points: ', pcs_world.shape[0])
        print('points in box: ', pc_in_bbox.shape[0])
        print('Error! tight bbox failed: not all points are in the tight bbox!')
        exit(-1)

    return (tight_bbox_world_moved, (R, T, S))


def get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, frame_num, meta):
    # get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
    joint_qpos = joint_qpos_list[parent_joint_id]  # !!! the joint's index should be the same as its child's index - 1

    # 获得当前button对应轴的xyz点和方向向量
    axis_start_point = np.array(joint_pose_dict[parent_joint_id]['xyz']).reshape(1, 3)
    axis_direction_vector = np.array(joint_pose_dict[parent_joint_id]['axis']).reshape(1, 3)
    axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)

    # 读取part对应的.obj文件
    obj_list = handle_objs
    part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
    part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                         )  # 从obj空间转移到sapien world space (利用base处的rpy？)

    # 设置static下的pose，prismatic下直接得到world下的pose
    canon_in_static = get_canon_in_static_frame(frame_num)
    rotation_theta = joint_qpos - joint_info['limit'][parent_joint_id][0]
    static2world = (t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), rotation_theta)).T
    canon2world = canon_in_static @ static2world
    world2canon = canon2world.T

    # 转换part到canon space，得到tight bbox和pose
    obj2world = (t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), joint_qpos)).T
    part_obj_pts = (part_obj_pts - axis_start_point) @ obj2world + axis_start_point
    part_conon_pts = part_obj_pts @ world2canon

    # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
    pcs_in_original = pcs_world
    pcs_original_canon = pcs_in_original @ world2canon
    part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

    x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
    y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
    z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
    l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
    w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
    h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

    center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
    tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
    tight_bbox_world = tight_bbox_canon @ canon2world

    # 恢复part在sapien里的移动，通过joint的qpos
    center_t_moved = center_t
    tight_bbox_world_moved = tight_bbox_world
    x_t = center_t_moved[0, 0]
    y_t = center_t_moved[0, 1]
    z_t = center_t_moved[0, 2]

    R = canon2world  # shape (3,3)
    T = center_t_moved.reshape(3, )
    S = np.array([l_s, w_s, h_s]).reshape(3, )

    pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
    if pc_in_bbox.shape[0] != pcs_world.shape[0]:
        print('part points: ', pcs_world.shape[0])
        print('points in box: ', pc_in_bbox.shape[0])
        print('Error! tight bbox failed: not all points are in the tight bbox!')
        exit(-1)

    return (tight_bbox_world_moved, (R, T, S))


def get_free_handle(pcs, handle_objs, frame_num, meta):
    # get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    # 读取part对应的.obj文件
    obj_list = handle_objs
    part_obj_pts, _, _ = get_all_objs(obj_file_list=[obj_list])
    part_obj_pts = part_obj_pts[0][0] @ (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
                                         )  # 从obj空间转移到sapien world space (利用base处的rpy？)

    # 设置static下的pose，prismatic下直接得到world下的pose
    canon_in_static = get_canon_in_static_frame(frame_num)
    canon2world = canon_in_static
    world2canon = canon2world.T

    # 转换part到canon space，得到tight bbox和pose
    part_conon_pts = part_obj_pts @ world2canon

    # 在obj读出点的基础上叠加sapien采得的点云，以避免浮点误差带来的问题
    pcs_in_original = pcs_world
    pcs_original_canon = pcs_in_original @ world2canon
    part_conon_pts = np.vstack((part_conon_pts, pcs_original_canon))

    x_center = float(max(part_conon_pts[:, 0]) + min(part_conon_pts[:, 0])) / 2
    y_center = float(max(part_conon_pts[:, 1]) + min(part_conon_pts[:, 1])) / 2
    z_center = float(max(part_conon_pts[:, 2]) + min(part_conon_pts[:, 2])) / 2
    l_s = float(max(part_conon_pts[:, 0]) - min(part_conon_pts[:, 0])) * (1 + EPSILON_L)
    w_s = float(max(part_conon_pts[:, 1]) - min(part_conon_pts[:, 1])) * (1 + EPSILON_W)
    h_s = float(max(part_conon_pts[:, 2]) - min(part_conon_pts[:, 2])) * (1 + EPSILON_H)

    center_t = (np.array([x_center, y_center, z_center]).reshape(1, 3)) @ canon2world
    tight_bbox_canon = get_bbox_from_scale(x_center, y_center, z_center, l_s, w_s, h_s)
    tight_bbox_world = tight_bbox_canon @ canon2world

    # 恢复part在sapien里的移动，通过joint的qpos
    center_t_moved = center_t
    tight_bbox_world_moved = tight_bbox_world
    x_t = center_t_moved[0, 0]
    y_t = center_t_moved[0, 1]
    z_t = center_t_moved[0, 2]

    R = canon2world  # shape (3,3)
    T = center_t_moved.reshape(3, )
    S = np.array([l_s, w_s, h_s]).reshape(3, )

    pc_in_bbox, _ = extract_pc_in_box3d(pcs_world, tight_bbox_world_moved)
    if pc_in_bbox.shape[0] != pcs_world.shape[0]:
        print('part points: ', pcs_world.shape[0])
        print('points in box: ', pc_in_bbox.shape[0])
        print('Error! tight bbox failed: not all points are in the tight bbox!')
        exit(-1)

    return (tight_bbox_world_moved, (R, T, S))


def get_fixed_handle_pose(inst_name, pcs, scene, urdf_ins, link_pose_dict, joint_pose_dict, anno_config, meta):
    fixed_handle_list = urdf_ins['fixed_handle_ins']
    joint_info = urdf_ins['joint']
    handle_objs = None
    link_name = inst_name.split('/')[0].split(':')[0]
    if link_name == 'base':
        parent_link_id = 0
    else:
        parent_link_id = int(link_name.split('_')[1]) + 1
    for handle_tuple in fixed_handle_list[parent_link_id]:
        _, part_name, objs = handle_tuple
        if part_name == inst_name:
            handle_objs = objs
            break
    assert handle_objs != None, "Error! not find match fixed handle objs!"

    object_category = meta['category']
    model_id = meta['model_id']
    anno_scheme_index = FIXED_HANDLE_ANNO_SCHEME[object_category][anno_config[object_category][str(model_id)]]

    # for debugging: get the point cloud in world space
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl

    joint_id_list = []
    last_link_id = parent_link_id
    end_flag = False
    while (not end_flag):
        for i in range(urdf_ins['num_links'] - 1):
            if joint_info['child'][
                    i] == last_link_id:  # !!! the joint's index should be the same as its child's index - 1
                joint_id_list.append(i)
                if joint_info['parent'][i] == 0:
                    end_flag = True
                    break
                else:
                    last_link_id = joint_info['parent'][i]
                    break
    joint_id_list = joint_id_list[:-1]
    kinematic_chain_depth = len(joint_id_list)
    if kinematic_chain_depth >= 2 and object_category != 'CoffeeMachine':  # 只允许咖啡机depth>=2
        print('Error! for fixed handle, kinematic chain depth is greater than 1! don\'t support!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        exit(-1)

    if anno_scheme_index == 1:
        parent_joint_id = joint_id_list[0]
        parent_joint_type = joint_info['type'][parent_joint_id]
        if parent_joint_type == 'prismatic':
            tag = 'line'
            bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 1, meta)
            return bbox, tag
        elif parent_joint_type == 'revolute':
            tag = 'line'
            bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 2, meta)
            return bbox, tag
        else:
            print('Error! not supported joint type')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(joint_info['type'][parent_joint_id])
            exit(-1)

    elif anno_scheme_index == 2:
        parent_joint_id = joint_id_list[0]
        parent_joint_type = joint_info['type'][parent_joint_id]
        if parent_joint_type == 'prismatic':
            tag = 'line'
            bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 1, meta)
            return bbox, tag
        elif parent_joint_type == 'revolute':
            tag = 'line'
            bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 1, meta)
            return bbox, tag
        else:
            print('Error! not supported joint type')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(joint_info['type'][parent_joint_id])
            exit(-1)

    elif anno_scheme_index == 3:
        parent_joint_id = joint_id_list[0]
        parent_joint_type = joint_info['type'][parent_joint_id]
        if parent_joint_type == 'prismatic':
            tag = 'line'
            bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 1, meta)
            return bbox, tag
        elif parent_joint_type == 'revolute':
            tag = 'round'
            bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 4, meta)
            return bbox, tag
        else:
            print('Error! not supported joint type')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(joint_info['type'][parent_joint_id])
            exit(-1)

    elif anno_scheme_index == 4:
        parent_joint_id = joint_id_list[0]
        parent_joint_type = joint_info['type'][parent_joint_id]
        if parent_joint_type == 'prismatic':
            tag = 'line'
            bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 1, meta)
            return bbox, tag
        else:
            print('Error! not supported joint type')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(joint_info['type'][parent_joint_id])
            exit(-1)

    elif anno_scheme_index == 5:
        if kinematic_chain_depth == 0:
            tag = 'line'
            Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
            Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
            pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl
            if get_y_crd_sign(pcs_world) > 0:
                bbox = get_free_handle(pcs, handle_objs, 6, meta)
                return bbox, tag
            else:
                bbox = get_free_handle(pcs, handle_objs, 7, meta)
                return bbox, tag
        else:
            tag = 'round'
            parent_joint_id = joint_id_list[0]
            parent_joint_type = joint_info['type'][parent_joint_id]
            bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 5, meta)
            return bbox, tag

    elif anno_scheme_index == 6:  # 咖啡机的kinematic chain可能为2 对lid上的handle要特殊处理
        link_category = inst_name.split('/')[0].split(':')[1]
        if 'slider_lid' in link_category:  # depth = 2
            tag = 'round'
            parent_joint_id = joint_id_list[0]
            parent_joint_type = joint_info['type'][parent_joint_id]
            grandparent_joint_id = joint_id_list[1]
            grandparent_joint_type = joint_info['type'][grandparent_joint_id]
            bbox = get_prismatic_based_handle_depth2(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id,
                                                     grandparent_joint_id, 5, meta)
            return bbox, tag
        else:
            tag = 'line'
            parent_joint_id = joint_id_list[0]
            parent_joint_type = joint_info['type'][parent_joint_id]
            Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
            Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
            pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl
            if get_y_crd_sign(pcs_world) > 0:
                bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 8,
                                                  meta)
                return bbox, tag
            else:
                bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 9,
                                                  meta)
                return bbox, tag

    elif anno_scheme_index == 7:
        parent_joint_id = joint_id_list[0]
        parent_joint_type = joint_info['type'][parent_joint_id]
        if parent_joint_type == 'prismatic':
            tag = 'round'
            bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 4, meta)
            return bbox, tag
        elif parent_joint_type == 'revolute':
            tag = 'line'
            bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 2, meta)
            return bbox, tag
        else:
            print('Error! not supported joint type')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(joint_info['type'][parent_joint_id])
            exit(-1)

    elif anno_scheme_index == 8:
        link_category = inst_name.split('/')[0].split(':')[1]
        if 'slider_lid' in link_category:
            tag = 'line'
            parent_joint_id = joint_id_list[0]
            parent_joint_type = joint_info['type'][parent_joint_id]
            bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 3, meta)
            return bbox, tag
        else:
            tag = 'line'
            Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
            Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
            pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl
            if get_y_crd_sign(pcs_world) > 0:
                bbox = get_free_handle(pcs, handle_objs, 6, meta)
                return bbox, tag
            else:
                bbox = get_free_handle(pcs, handle_objs, 7, meta)
                return bbox, tag

    elif anno_scheme_index == 9:
        parent_joint_id = joint_id_list[0]
        parent_joint_type = joint_info['type'][parent_joint_id]
        if parent_joint_type == 'prismatic':
            tag = 'line'
            bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 2, meta)
            return bbox, tag
        else:
            print('Error! not supported joint type')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(joint_info['type'][parent_joint_id])
            exit(-1)

    elif anno_scheme_index == 10:
        if kinematic_chain_depth == 0:
            tag = 'line'
            Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
            Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
            pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl
            if get_y_crd_sign(pcs_world) > 0:
                bbox = get_free_handle(pcs, handle_objs, 6, meta)
                return bbox, tag
            else:
                bbox = get_free_handle(pcs, handle_objs, 7, meta)
                return bbox, tag
        else:
            tag = 'line'
            parent_joint_id = joint_id_list[0]
            parent_joint_type = joint_info['type'][parent_joint_id]
            Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
            Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
            pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl
            if get_y_crd_sign(pcs_world) > 0:
                bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 6,
                                                  meta)
                return bbox, tag
            else:
                bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 7,
                                                  meta)
                return bbox, tag

    elif anno_scheme_index == 11:
        parent_joint_id = joint_id_list[0]
        parent_joint_type = joint_info['type'][parent_joint_id]
        if parent_joint_type == 'prismatic':
            tag = 'round'
            bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 4, meta)
            return bbox, tag
        elif parent_joint_type == 'revolute':
            tag = 'round'
            bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 4, meta)
            return bbox, tag
        else:
            print('Error! not supported joint type')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(joint_info['type'][parent_joint_id])
            exit(-1)

    elif anno_scheme_index == 12:
        parent_joint_id = joint_id_list[0]
        parent_joint_type = joint_info['type'][parent_joint_id]
        if parent_joint_type == 'prismatic':
            tag = 'line'
            bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 2, meta)
            return bbox, tag
        elif parent_joint_type == 'revolute':
            tag = 'line'
            bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 2, meta)
            return bbox, tag
        else:
            print('Error! not supported joint type')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(joint_info['type'][parent_joint_id])
            exit(-1)

    elif anno_scheme_index == 13:
        if kinematic_chain_depth == 0:
            tag = 'line'
            Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
            Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
            pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl
            if get_y_crd_sign(pcs_world) > 0:
                bbox = get_free_handle(pcs, handle_objs, 6, meta)
                return bbox, tag
            else:
                bbox = get_free_handle(pcs, handle_objs, 7, meta)
                return bbox, tag
        else:
            tag = 'line'
            parent_joint_id = joint_id_list[0]
            parent_joint_type = joint_info['type'][parent_joint_id]
            Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
            Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
            pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl
            if get_y_crd_sign(pcs_world) > 0:
                bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 6,
                                                 meta)
                return bbox, tag
            else:
                bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 7,
                                                 meta)
                return bbox, tag

    elif anno_scheme_index == 14:
        if kinematic_chain_depth == 1:
            tag = 'line'
            parent_joint_id = joint_id_list[0]
            parent_joint_type = joint_info['type'][parent_joint_id]
            Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
            Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
            pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl
            if get_y_crd_sign(pcs_world) > 0:
                bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 10,
                                                 meta)
                return bbox, tag
            else:
                bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 10,
                                                 meta)
                return bbox, tag
        else:
            print('Error! not supported kinematic chain depth!')
            print(meta['category'], meta['model_id'])
            print(inst_name)
            print(kinematic_chain_depth)
            exit(-1)

    elif anno_scheme_index == 15:
        if kinematic_chain_depth == 1:
            tag = 'line'
            parent_joint_id = joint_id_list[0]
            parent_joint_type = joint_info['type'][parent_joint_id]
            bbox = get_revolute_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 3, meta)
            return bbox, tag

        else:
            tag = 'line'
            bbox = get_free_handle(pcs, handle_objs, 1, meta)
            return bbox, tag

    elif anno_scheme_index == 16:  # 咖啡机的kinematic chain可能为2 对lid上的handle要特殊处理
        link_category = inst_name.split('/')[0].split(':')[1]
        if 'slider_lid' in link_category:  # depth = 2
            tag = 'line'
            parent_joint_id = joint_id_list[0]
            parent_joint_type = joint_info['type'][parent_joint_id]
            grandparent_joint_id = joint_id_list[1]
            grandparent_joint_type = joint_info['type'][grandparent_joint_id]
            bbox = get_prismatic_based_handle_depth2(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id,
                                                     grandparent_joint_id, 3, meta)
            return bbox, tag
        else:
            tag = 'line'
            parent_joint_id = joint_id_list[0]
            parent_joint_type = joint_info['type'][parent_joint_id]
            Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
            Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
            pcs_world = pcs @ Rtilt_rot.T + Rtilt_trl
            if get_y_crd_sign(pcs_world) > 0:
                bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 8,
                                                  meta)
                return bbox, tag
            else:
                bbox = get_prismatic_based_handle(pcs, joint_info, joint_pose_dict, handle_objs, parent_joint_id, 9,
                                                  meta)
                return bbox, tag

    else:
        print('Error! wrong anno scheme index!')
        print(meta['category'], meta['model_id'])
        print(inst_name)
        print(anno_scheme_index)
        exit(-1)


def get_urdf_mobility(inpath, verbose=False):
    if not inpath.endswith(".urdf"):
        urdf_name = inpath + "/mobility_relabel.urdf"
    else:
        urdf_name = inpath
        inpath = '/'.join(inpath.split('/')[:-1])

    urdf_ins = {}
    tree_urdf = ET.parse(urdf_name)
    num_real_links = len(tree_urdf.findall('link'))
    num_real_joints = len(tree_urdf.findall('joint'))
    print(num_real_joints)
    root_urdf = tree_urdf.getroot()
    rpy_xyz = {}
    list_xyz = [None] * num_real_links
    list_rpy = [None] * num_real_links
    list_obj = [None] * num_real_links
    # ['obj'] ['link/joint']['xyz/rpy'] [0, 1, 2, 3, 4]
    num_links = 0
    handle_parent_index = []
    handle_index = []
    handle_names = []
    
    for link in root_urdf.iter('link'):
        num_links += 1
        if link.attrib['name'] == 'base':
            index_link = 0
        elif "handle" in link.attrib['name']:
            handle_parent_index.append(index_link)
            if len(link.attrib['name'].split("_")) == 3:
                handle_index.append(int(link.attrib['name'].split("_")[-1]) + 1)
            continue
        else:
            index_link = int(link.attrib['name'].split('_')[1]) + 1  # since the name is base, link_0, link_1
        list_xyz[index_link] = []
        list_rpy[index_link] = []
        list_obj[index_link] = []
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

    rpy_xyz = {}
    list_type = [None] * (num_real_links - 1)
    list_parent = [None] * (num_real_links - 1)
    list_child = [None] * (num_real_links - 1)
    list_xyz = [None] * (num_real_links - 1)
    list_rpy = [None] * (num_real_links - 1)
    list_axis = [None] * (num_real_links - 1)
    list_limit = [[0, 0]] * (num_real_links - 1)
    # here we still have to read the URDF file
    for joint in root_urdf.iter('joint'):
        """
        joint_index = int(joint.attrib['name'].split('_')[1])
        list_type[joint_index] = joint.attrib['type']
        """

        for child in joint.iter('child'):
            link_name = child.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            joint_index = link_index - 1  # !!! the joint's index should be the same as its child's index - 1
            list_child[joint_index] = link_index

        list_type[joint_index] = joint.attrib['type']

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

    urdf_ins['joint'] = rpy_xyz
    urdf_ins['num_links'] = num_real_links
    if verbose:
        for j, pos in enumerate(urdf_ins['link']['xyz']):
            if len(pos) > 3:
                print('link {} xyz: '.format(j), pos[0])
            else:
                print('link {} xyz: '.format(j), pos)
        for j, orient in enumerate(urdf_ins['link']['rpy']):
            if len(orient) > 3:
                print('link {} rpy: '.format(j), orient[0])
            else:
                print('link {} rpy: '.format(j), orient)
        # for joint
        for j, pos in enumerate(urdf_ins['joint']['xyz']):
            print('joint {} xyz: '.format(j), pos)
        for j, orient in enumerate(urdf_ins['joint']['rpy']):
            print('joint {} rpy: '.format(j), orient)
        for j, orient in enumerate(urdf_ins['joint']['axis']):
            print('joint {} axis: '.format(j), orient)
        for j, child in enumerate(urdf_ins['joint']['child']):
            print('joint {} has child link: '.format(j), child)
        for j, parent in enumerate(urdf_ins['joint']['parent']):
            print('joint {} has parent link: '.format(j), parent)

    return urdf_ins,