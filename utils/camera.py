import math 
import numpy as np
from scipy.spatial.transform import Rotation

def quaternionFromRotMat(rotation_matrix, mode):
    rotation_matrix = np.reshape(rotation_matrix, (1, 9))[0]
    w = math.sqrt(rotation_matrix[0]+rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    x = math.sqrt(rotation_matrix[0]-rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    y = math.sqrt(-rotation_matrix[0]+rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    z = math.sqrt(-rotation_matrix[0]-rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    a = [w,x,y,z]
    m = a.index(max(a))
    if m == 0:
        x = (rotation_matrix[7]-rotation_matrix[5])/(4*w)
        y = (rotation_matrix[2]-rotation_matrix[6])/(4*w)
        z = (rotation_matrix[3]-rotation_matrix[1])/(4*w)
    if m == 1:
        w = (rotation_matrix[7]-rotation_matrix[5])/(4*x)
        y = (rotation_matrix[1]+rotation_matrix[3])/(4*x)
        z = (rotation_matrix[6]+rotation_matrix[2])/(4*x)
    if m == 2:
        w = (rotation_matrix[2]-rotation_matrix[6])/(4*y)
        x = (rotation_matrix[1]+rotation_matrix[3])/(4*y)
        z = (rotation_matrix[5]+rotation_matrix[7])/(4*y)
    if m == 3:
        w = (rotation_matrix[3]-rotation_matrix[1])/(4*z)
        x = (rotation_matrix[6]+rotation_matrix[2])/(4*z)
        y = (rotation_matrix[5]+rotation_matrix[7])/(4*z)
    if mode=="wxyz":
        quaternion = (w,x,y,z)
    elif mode=="xyzw":
        quaternion = (x,y,z,w)
    return quaternion

def quaternionToRotation(q):
    x, y, z, w = q
    r00 = 1 - 2 * y ** 2 - 2 * z ** 2
    r01 = 2 * x * y + 2 * w * z
    r02 = 2 * x * z - 2 * w * y

    r10 = 2 * x * y - 2 * w * z
    r11 = 1 - 2 * x ** 2 - 2 * z ** 2
    r12 = 2 * y * z + 2 * w * x

    r20 = 2 * x * z + 2 * w * y
    r21 = 2 * y * z - 2 * w * x
    r22 = 1 - 2 * x ** 2 - 2 * y ** 2
    r = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return r

def getRTFromAToB(pointCloudA, pointCloudB):

    muA = np.mean(pointCloudA, axis=0)
    muB = np.mean(pointCloudB, axis=0)

    zeroMeanA = pointCloudA - muA
    zeroMeanB = pointCloudB - muB

    # 计算协方差矩阵
    covMat = np.matmul(np.transpose(zeroMeanA), zeroMeanB)
    U, S, Vt = np.linalg.svd(covMat)
    R = np.matmul(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T
    T = (-np.matmul(R, muA.T) + muB.T).reshape(3, 1)
    return R, T

def rotationAngleToMatrix(angle, axis):
    if axis == "x":
        transform_matrix = np.array([[1, 0, 0, 0], 
                                [0, math.cos(angle), -math.sin(angle), 0], 
                                [0, math.sin(angle), math.cos(angle), 0], 
                                [0, 0, 0, 1]])
    elif axis == "y":
        transform_matrix = np.array([[math.cos(angle), 0, math.sin(angle), 0], 
                                [0, 1, 0, 0], 
                                [-math.sin(angle), 0, math.cos(angle), 0], 
                                [0, 0, 0, 1]])
    elif axis == "z":
        transform_matrix = np.array([[math.cos(angle), -math.sin(angle), 0, 0],
                                [math.sin(angle), math.cos(angle), 0, 0],
                                [0,0,1,0],
                                [0,0,0,1]])
    else:
        print("rotationAngleToMatrix ERROR!")

    return transform_matrix

def add_noise_to_transformation_matrix(Rot, Trans, angle_std=2, translation_std=0.01):
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, angle_std) / 180 * np.pi
    Rot = Rotation.from_rotvec(angle * axis).as_matrix() @ Rot
    direction = np.random.rand(3)
    direction /= np.linalg.norm(direction)
    length = np.random.uniform(0, translation_std)
    Trans += direction.reshape(Trans.shape) * length
    return Rot, Trans

def gen_camera_pose(look_at, alpha_range_list, num_point_ver_list, num_point_hor, beta_range, r, is_camera_rand=False):
    ### q
    quat_src_list = []
    rot_src_list = []
    trans_src_list = []
    ### q
    quat_list = []
    rot_list = []
    trans_list = []
    pose_mat_list = []

    for alpha_range_id in range(len(alpha_range_list)):
        alpha_range = alpha_range_list[alpha_range_id]
        num_point_ver = num_point_ver_list[alpha_range_id]
        alpha = alpha_range[0]
        alpha_delta = (alpha_range[1]-alpha_range[0])/(num_point_ver-1)
        for i in range(num_point_ver):
            if i != 0:
                alpha = alpha + alpha_delta # 从x轴起绕z轴逆时针转
            # 确保alpha角是锐角，并根据alpha角大小确定象限，以确定x和y符号
            flag_x = 1
            flag_y = 1
            alpha1 = alpha
            if alpha < 0:
                alpha = alpha + 2 * math.pi
            if alpha > math.pi/2 and alpha <= math.pi:
                alpha1 = math.pi - alpha #alpha - math.pi/2
                flag_x = -1
                flag_y = 1
            elif alpha > math.pi and alpha <= math.pi*(3/2):
                alpha1 = alpha - math.pi #math.pi*(3/2) - alpha
                flag_x = -1
                flag_y = -1
            elif alpha > math.pi*(3/2):
                alpha1 = math.pi*2 - alpha #alpha - math.pi*(3/2)
                flag_x = 1
                flag_y = -1

            beta = beta_range[0]
            if num_point_hor > 1:
                beta_delta = (beta_range[1]-beta_range[0])/(num_point_hor-1)
            for j in range(num_point_hor):
                if j != 0:
                    beta = beta + beta_delta    # 从z轴起绕y轴逆时针转
                # 计算相机position
                x = flag_x * (r * math.sin(beta)) * math.cos(alpha1)
                y = flag_y * (r * math.sin(beta)) * math.sin(alpha1)
                z = r * math.cos(beta)
                position = np.array([x, y, z]) + look_at
                # lookat由外面传入
                look_at = look_at
                # up，暂定为一直向上
                up = np.array([0, 0, 1])

                vectorZ = - (look_at - position)/np.linalg.norm(look_at - position)
                vectorX = np.cross(up, vectorZ)/np.linalg.norm(np.cross(up, vectorZ))
                vectorY = np.cross(vectorZ, vectorX)/np.linalg.norm(np.cross(vectorX, vectorZ))

                # points in camera coordinates
                pointSensor= np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])

                # points in world coordinates
                pointWorld = np.array([position,
                                    position + vectorX,
                                    position + vectorY * 2,
                                    position + vectorZ * 3])

                # get R and T
                resR, resT = getRTFromAToB(pointSensor, pointWorld)

                # add noise
                if is_camera_rand:
                    resR, resT = add_noise_to_transformation_matrix(resR, resT)

                ### q
                # add to list
                resQ = quaternionFromRotMat(resR, "wxyz")
                quat_src_list.append(resQ)
                rot_src_list.append(resR)
                trans_src_list.append(resT)
                ### q

                # 由于issac相机初始pose沿x正方向看，需先绕y轴旋转90度，再绕z轴转90度
                angle = math.pi/2
                y_rot = rotationAngleToMatrix(angle, "y")[:3,:3]
                z_rot = rotationAngleToMatrix(angle, "z")[:3,:3]
                rot_mat_new = resR @ (z_rot @ y_rot)
                quat_new = quaternionFromRotMat(rot_mat_new, "xyzw")

                # add to list
                quat_list.append(quat_new)
                rot_list.append(rot_mat_new)
                trans_list.append(resT)

                x_transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                x_transform_inv = np.linalg.inv(x_transform)
                resR = resR @ x_transform_inv
                rt = np.concatenate([resR, resT.reshape(3, 1)], 1)
                pose_mat = np.concatenate([rt, [[0, 0, 0, 1]]], 0)

                pose_mat_list.append(pose_mat)

    # NOTE quat is not corresponde to pose_mat !!!!
    #return np.array(quat_list), np.array(rot_list), np.array(trans_list), np.array(pose_mat_list)
    ### q
    return np.array(quat_src_list), np.array(rot_src_list), np.array(trans_src_list), \
           np.array(quat_list), np.array(rot_list), np.array(trans_list), np.array(pose_mat_list)
    ### q