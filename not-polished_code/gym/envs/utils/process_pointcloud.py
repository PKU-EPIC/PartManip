
import torch
from pointnet2_ops import pointnet2_utils
# from perception.structures.point_cloud import PointCloud
import time
# @TimeCounter
def compute_point_cloud_state(task, depth_bar, asymm_obs=False, add_mask=None, return_img=False):

    torch.cuda.synchronize()
    s = time.time()
    task.gym.render_all_camera_sensors(task.sim)
    torch.cuda.synchronize()
    t = time.time()
    torch.cuda.synchronize()
    print("cam pointcloud", t - s)
    # import pdb
    # pdb.set_trace()
    task.gym.start_access_image_tensors(task.sim)
    point_clouds = torch.zeros((task.num_envs, task.pc_downsample_num, 7), device=task.device)
    camera_depth_tensor = torch.stack([torch.stack(i) for i in task.camera_depth_tensor_list])
    camera_rgb_tensor = torch.stack([torch.stack(i) for i in task.camera_rgb_tensor_list])
    camera_seg_tensor = torch.stack([torch.stack(i) for i in task.camera_seg_tensor_list])
    camera_view_matrix_inv = torch.stack([torch.stack(i) for i in task.camera_view_matrix_inv_list])
    camera_proj_matrix = torch.stack([torch.stack(i) for i in task.camera_proj_matrix_list])
    
    points0 ,valid0 = depth_image_to_point_cloud_GPU_batch(camera_depth_tensor[:,0], \
            camera_rgb_tensor[:,0], camera_seg_tensor[:,0], \
            camera_view_matrix_inv[:,0], camera_proj_matrix[:,0], \
            task.camera_u2, task.camera_v2, task.camera_props.width, task.camera_props.height, depth_bar, task.device, task.z_p_bar, task.z_n_bar)
    points1 ,valid1 = depth_image_to_point_cloud_GPU_batch(camera_depth_tensor[:,1], \
            camera_rgb_tensor[:,1], camera_seg_tensor[:,1], \
            camera_view_matrix_inv[:,1], camera_proj_matrix[:,1], \
            task.camera_u2, task.camera_v2, task.camera_props.width, task.camera_props.height, depth_bar, task.device, task.z_p_bar, task.z_n_bar)
    points2 ,valid2 = depth_image_to_point_cloud_GPU_batch(camera_depth_tensor[:,2], \
            camera_rgb_tensor[:,2], camera_seg_tensor[:,2], \
            camera_view_matrix_inv[:,2], camera_proj_matrix[:,2], \
            task.camera_u2, task.camera_v2, task.camera_props.width, task.camera_props.height, depth_bar, task.device, task.z_p_bar, task.z_n_bar)
    # import cv2
    # cv2.imwrite()
    points = torch.cat((points0, points1, points2), dim = 1)
    points[:,:,:3] -= task.env_origin.view(task.num_envs, 1, 3)
    valid = torch.cat((valid0, valid1, valid2), dim = 1)*(points[:, :, 0]>task.x_n_bar)*(points[:, :, 0]<task.x_p_bar)*(points[:, :, 1]<task.y_p_bar)*(points[:, :, 1]>task.y_n_bar)

    point_nums = valid.sum(1)
    now = 0
    total_per_env = valid.shape[1]
    points_list = []
    valid_points = points[valid]

    for point_num in point_nums:
        points_all = valid_points[now : now + point_num]
        random_ids = torch.randint(0,points_all.shape[0], (task.pre_random_sample_num,), device = task.device, dtype=torch.long)
        points_all_rnd = points_all[random_ids]
        points_list.append(points_all_rnd)
        now += point_num

    points_batch = torch.stack(points_list)
    # points_fps = points_batch
    points_fps = task.sample_points(points_batch, sample_num=task.pc_downsample_num, sample_method='furthest_batch')

    # print("get_point_cloud_total_time:",e - s)
    # print(points_fps[:,0,:3])
    # exit(123)
    # points_tensor = torch.stack(points_list)


    # for i in range(task.num_envs):
    #     img_0 = task.camera_rgb_tensor_list[i][0].cpu().numpy()
    #     img_1 = task.camera_rgb_tensor_list[i][1].cpu().numpy()
    #     img_2 = task.camera_rgb_tensor_list[i][2].cpu().numpy()
        # img_rgb_0_bgr = cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR)
        # img_rgb_1_bgr = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
        # img_rgb_2_bgr = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)
        # img_seg_0 = COLOR20[task.camera_seg_tensor_list[i][0].long().cpu().numpy()]
        # img_seg_1 = COLOR20[task.camera_seg_tensor_list[i][1].long().cpu().numpy()]
        # img_seg_2 = COLOR20[task.camera_seg_tensor_list[i][2].long().cpu().numpy()]
        # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/seg_{i}_0.png", img_seg_0)
        # cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/seg_{i}_1.png", img_seg_1)
    #     cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/seg_{i}_2.png", img_seg_2)
    #     cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/rgb_{i}_0.png", img_rgb_0_bgr)
    #     cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/rgb_{i}_1.png", img_rgb_1_bgr)
    #     cv2.imwrite(f"/data2/haoran/RL-Pose/PoseOrientedGym/assets/debug/rgb_{i}_2.png", img_rgb_2_bgr)
    # exit(123)
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_fps[0, :, :3].cpu().numpy())
    # pcd.colors = o3d.utility.Vector3dVector(points_fps[0, :, 3:6].cpu().numpy()/255.0)
    # task.pointCloudVisualizer.add_geometry(pcd)
    # while True :
    #     task.pointCloudVisualizer.update(pcd)

    task.gym.end_access_image_tensors(task.sim)

    if add_mask != None :

        (cabinet_door_relative_o,
        cabinet_door_relative_x,
        cabinet_door_relative_y,
        cabinet_door_relative_z,
        cabinet_door_x,
        cabinet_door_y,
        cabinet_door_z) = add_mask
    

        eps = 1e-2
        delta = point_clouds - cabinet_door_relative_o.view(task.num_envs, 1, 3)
        point_cloud_relative_x = (delta*(cabinet_door_relative_x.view(task.num_envs, 1, 3))).sum(dim=2)
        point_cloud_relative_y = (delta*(cabinet_door_relative_y.view(task.num_envs, 1, 3))).sum(dim=2)
        point_cloud_relative_z = (delta*(cabinet_door_relative_z.view(task.num_envs, 1, 3))).sum(dim=2)
        cabinet_door_x = cabinet_door_x.view(task.num_envs, 1)
        cabinet_door_y = cabinet_door_y.view(task.num_envs, 1)
        cabinet_door_z = cabinet_door_z.view(task.num_envs, 1)
        in_range = torch.where(
            (point_cloud_relative_x>=-eps) &
            (point_cloud_relative_x<=cabinet_door_x+eps) &
            (point_cloud_relative_y>=-eps) &
            (point_cloud_relative_y<=cabinet_door_y+eps) &
            (point_cloud_relative_z>=-eps) &
            (point_cloud_relative_z<=cabinet_door_z+eps), 1, 0)
        point_clouds = torch.cat((point_clouds, in_range.view(task.num_envs, -1, 1)), dim=2)

    if return_img:
        return points_fps, camera_rgb_tensor
    return points_fps

def sample_points(task, points, sample_num=1000, sample_method='random'):
    if sample_method == 'random':
        eff_points = points[points[:, 2]>0.04]
        eff_points_xyz = eff_points.contiguous()
        if eff_points.shape[0] < sample_num :
            eff_points = points[:, 0:3].contiguous()
        sampled_points = task.rand_row(eff_points, sample_num)
    elif sample_method == "furthest_batch":
        idx = pointnet2_utils.furthest_point_sample(points[:, :, :3].contiguous().cuda(), sample_num).long().to(task.device)
        idx = idx.view(*idx.shape, 1).repeat_interleave(points.shape[-1], dim=2)
        sampled_points = torch.gather(points, dim=1, index=idx)

    elif sample_method == 'furthest':
        eff_points = points[points[:, 2]>0.04]
        eff_points_xyz = eff_points.contiguous()
        if eff_points.shape[0] < sample_num :
            eff_points = points[:, 0:3].contiguous()
        sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points_xyz.reshape(1, *eff_points_xyz.shape), sample_num)
        sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
    return sampled_points

def PointCloud_format(pointcloud_fps):
    # print(pointcloud_fps.shape)
    PointCloudList_ = []
    mask_list_ = [[], []]
    for i in range(pointcloud_fps.shape[0]):
        PointCloudList_.append(pointcloud_fps[i][:,:6].cpu())
    mask_list_[0] = (pointcloud_fps[:, :, 6] == 102)
    mask_list_[1] = (pointcloud_fps[:, :, 6] == 103)

    return PointCloudList_, mask_list_

# @torch.jit.script
def depth_image_to_point_cloud_GPU(camera_depth_tensor, camera_rgb_tensor, camera_seg_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    # time1 = time.time()
    # print(u,v,width, height)
    # exit(123)
    depth_buffer = camera_depth_tensor.to(device)
    rgb_buffer = camera_rgb_tensor.to(device)
    seg_buffer = camera_seg_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = camera_view_matrix_inv

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv
    # print(rgb_buffer.shape)
    # print(seg_buffer.shape)
    R = rgb_buffer[...,0].view(-1)
    G = rgb_buffer[...,1].view(-1)
    B = rgb_buffer[...,2].view(-1)
    S = seg_buffer.view(-1)

    # exit(123)

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device), R, G, B, S))[:, valid]
    position = position.permute(1, 0)
    position[:,0:4] = position[:,0:4]@vinv
    print(position.shape)
    points = torch.cat((position[:, 0:3], position[:, 4:8]), dim = 1)

    return points

def depth_image_to_point_cloud_GPU_batch(
    camera_depth_tensor_batch, camera_rgb_tensor_batch, 
    camera_seg_tensor_batch, camera_view_matrix_inv_batch, 
    camera_proj_matrix_batch, u, v, width: float, height:float, 
    depth_bar:float, device:torch.device, 
    z_p_bar:float = 3.0,
    z_n_bar:float = 0.3,):
    # time1 = time.time()
    batch_num = camera_depth_tensor_batch.shape[0]
    depth_buffer_batch = camera_depth_tensor_batch.to(device)
    rgb_buffer_batch = camera_rgb_tensor_batch.to(device)/255.0
    seg_buffer_batch = camera_seg_tensor_batch.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv_batch = camera_view_matrix_inv_batch

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    
    proj_batch = camera_proj_matrix_batch
    fu_batch = 2/proj_batch[:, 0, 0]
    fv_batch = 2/proj_batch[:, 1, 1]

    centerU = width/2
    centerV = height/2

    Z_batch = depth_buffer_batch
    # print(u, centerU, width, Z_batch.shape, fu_batch.shape)
    X_batch = -(u.view(1, u.shape[-2], u.shape[-1])-centerU)/width * Z_batch * fu_batch.view(-1, 1, 1)
    Y_batch = (v.view(1, v.shape[-2], v.shape[-1])-centerV)/height * Z_batch * fv_batch.view(-1, 1, 1)
    # print(rgb_buffer.shape)
    # print(seg_buffer.shape)
    R_batch = rgb_buffer_batch[...,0].view(batch_num, 1, -1)
    G_batch = rgb_buffer_batch[...,1].view(batch_num, 1, -1)
    B_batch = rgb_buffer_batch[...,2].view(batch_num, 1, -1)
    S_batch = seg_buffer_batch.view(batch_num, 1, -1)

    # exit(123)
    valid_depth_batch = Z_batch.view(batch_num, -1) > -depth_bar
    # print(Z_batch.shape)
    Z_batch = Z_batch.view(batch_num, 1, -1)
    
    X_batch = X_batch.view(batch_num, 1, -1)
    Y_batch = Y_batch.view(batch_num, 1, -1)
    O_batch = torch.ones((X_batch.shape), device=device)

    position_batch = torch.cat((X_batch, Y_batch, Z_batch, O_batch, R_batch, G_batch, B_batch, S_batch), dim=1)
    position_batch = position_batch.permute(0, 2, 1)
    position_batch[...,0:4] = position_batch[...,0:4]@vinv_batch
    # print(position_batch.shape)
    # points_batch = torch.cat((position_batch[..., 0:3], position_batch[..., 4:8]), dim = 2)
    
    points_batch = position_batch[..., [0,1,2,4,5,6,7]]
    valid_z_p_batch = position_batch[...,2] < z_p_bar
    valid_z_n_batch = position_batch[...,2] > z_n_bar
    # valid_x_p_batch = position_batch[...,0] < x_p_bar
    # valid_x_n_batch = position_batch[...,0] > x_n_bar
    # valid_y_p_batch = position_batch[...,1] < y_p_bar
    # valid_y_n_batch = position_batch[...,1] > y_n_bar

    # print(valid_z_batch.shape)
    # print(valid_depth_batch.shape)
    valid_batch = valid_depth_batch*valid_z_p_batch*valid_z_n_batch
    # valid_batch = valid_depth_batch*valid_z_p_batch*valid_z_n_batch*valid_y_p_batch*valid_y_n_batch*valid_x_p_batch*valid_x_n_batch
    # print("valid", valid_batch.shape)
    return points_batch, valid_batch