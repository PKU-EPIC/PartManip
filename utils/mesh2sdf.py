'''
Paralelly compute SDF from mesh using pytorch/kaolin.
Define a volume first and query the SDF value of each voxel and finally save it.

'''

import numpy as np
import torch 
import torch.nn.functional as F
import trimesh 
from skimage import measure
from os.path import join as pjoin 
import os 

class TSDFfromMesh:
    def __init__(self, num_envs, size, resolution, device, parallel=True, debug=False, vox_origin=None):
        self.num_envs = num_envs
        self.parallel = parallel
        self.device = device
        self.debug = debug   # save pc to 'debug/'

        self.resolution = resolution
        self.size = size
        self.vox_size = self.size / self.resolution      
        self.sdf_trunc = 4 * self.vox_size
        self.vox_origin = torch.tensor([-0.25, -0.25, -0.0503], device=self.device) if vox_origin is None else vox_origin
        
        # define queried points in the workspace
        tmp = torch.arange(0, self.resolution)
        xv, yv, zv = torch.meshgrid(tmp, tmp, tmp)
        vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(self.device)
        self.vox_coords = vox_coords * self.vox_size + self.vox_origin     # [n, 3]
        self.point_num = self.vox_coords.shape[0]

        # NOTE Consider ground when initialization and no need to load ground!
        self.init_tsdf = torch.tensor(self.vox_coords[:, -1], device=self.device).unsqueeze(0).repeat(self.num_envs, 1) #[b,n]
        self.ground_tsdf = torch.tensor(self.vox_coords[:, -1], device=self.device).unsqueeze(0).repeat(self.num_envs, 1) #[b,n]

        self.pre_store_sdf_trunc = self.sdf_trunc 
        self.pre_store_sdf_voxel_size = 0.002 
        self.sdf_dict_list = []

        hand_base_path = './assets/franka_description'
        self.load_franka(hand_base_path)

        # load obj
        obj_mesh_path = './assets/objs/cube/cube.obj' 
        obj_sdf_path = './assets/objs/cube/sdf.npy'
        
        self.load_sdf(obj_sdf_path, obj_mesh_path)

        if self.parallel:
            self.merge_sdf_field()  # merge different sdf field for parallel
        return

    ### q
    def initialize_sdf(self, nerf_pred_tsdf):
        '''
        pred_tsdf: [b, n]
        '''
        self.init_tsdf = torch.tensor(nerf_pred_tsdf, device=self.device) * self.sdf_trunc
    ### q

    def load_sdf(self, sdf_path, mesh_path, preprocess_path=None):
        '''
        If preprocess_path is not None, Use manifoldplus to preprocess the mesh and save.
        The preprocess may avoid some problems in 'check_sign', but may also bring new problems.  
        '*.stl' file can't do preprocess.
        '''
        if os.path.exists(sdf_path):
            print(f'Find {sdf_path}! Loading pre-stored sdf...')
            sdf_dict = np.load(sdf_path, allow_pickle=True).item()
            print(sdf_dict['sdf'].shape)
        else:
            if preprocess_path is not None:
                self.preprocess_mesh(mesh_path, preprocess_path)
                mesh_path = preprocess_path
            sdf_dict = self.mesh2sdf(mesh_path)
            os.makedirs(os.path.dirname(sdf_path), exist_ok=True)
            np.save(sdf_path, sdf_dict)
        
        self.sdf_dict_list.append(sdf_dict)

        if self.debug:
            self.extract_surface_points_from_volume(sdf_dict['sdf'], f'debug/part{len(self.sdf_dict_list)}.txt')
            # self.visualize(sdf_dict['sdf'], sdf_dict['voxel_size'], sdf_dict['bbox_min'], save_path=f'debug/part{len(self.sdf_dict_list)}.txt')
        return 

    def query_tsdf(self, pose_R, pose_T):
        if self.parallel:
            return self.query_tsdf_parallel(pose_R, pose_T)
        else:
            return self.query_tsdf_naive(pose_R, pose_T)
    
    def query_tsdf_seperately(self, pose_R, pose_T):
        objspace_vox_coords = torch.bmm(self.vox_coords[None,...] - pose_T.unsqueeze(-2).reshape(-1,1,3), pose_R.reshape(-1,3,3))  # [b*m, n, 3]
        objspace_vox_coords = objspace_vox_coords.reshape(self.num_envs, self.part_num, self.point_num, 3)
        updated_tsdf = self.triplet_interpolation_query_parallel(objspace_vox_coords)    # [b,m,n]
        scene_tsdf = torch.cat((updated_tsdf[:,:-1,:],self.init_tsdf.unsqueeze(1)), dim=1)
        scene_tsdf = scene_tsdf.min(dim=1)[0]
        obj_tsdf = torch.cat((updated_tsdf[:,-1:,:],self.ground_tsdf.unsqueeze(1)), dim=1)
        obj_tsdf = obj_tsdf.min(dim=1)[0]

        scene_tsdf = torch.clamp(scene_tsdf / self.sdf_trunc, -1, 1)
        scene_tsdf = scene_tsdf.reshape(self.num_envs, self.resolution, self.resolution, self.resolution)
        
        obj_tsdf = torch.clamp(obj_tsdf / self.sdf_trunc, -1, 1)
        obj_tsdf = obj_tsdf.reshape(self.num_envs, self.resolution, self.resolution, self.resolution)

        if self.debug:
            scene_tsdf = scene_tsdf.cpu().numpy()
            obj_tsdf = obj_tsdf.cpu().numpy()
            for i in range(self.num_envs):
                self.extract_surface_points_from_volume(scene_tsdf[i], f'debug/scene{i}.txt')
                self.extract_surface_points_from_volume(obj_tsdf[i], f'debug/obj{i}.txt')
                # self.visualize(self.tsdf[i], self.vox_size, self.vox_origin.cpu().numpy(), f'debug/scene{i}.txt')
        return scene_tsdf, obj_tsdf 

    def query_tsdf_parallel(self, pose_R, pose_T):
        '''
        This is a little faster! 8 env 12 part: 0.032s
        pose_R: [b, m, 3, 3] b: env num, m: part num
        pose_T: [b, m, 3]
        '''
        objspace_vox_coords = torch.bmm(self.vox_coords[None,...] - pose_T.unsqueeze(-2).reshape(-1,1,3), pose_R.reshape(-1,3,3))  # [b*m, n, 3]
        objspace_vox_coords = objspace_vox_coords.reshape(self.num_envs, self.part_num, self.point_num, 3)
        updated_tsdf = self.triplet_interpolation_query_parallel(objspace_vox_coords)    # [b,m,n]
        tsdf = torch.cat((updated_tsdf,self.init_tsdf.unsqueeze(1)), dim=1)
        tsdf = tsdf.min(dim=1)[0]

        tsdf = torch.clamp(tsdf / self.sdf_trunc, -1, 1)
        tsdf = tsdf.reshape(self.num_envs, self.resolution, self.resolution, self.resolution)

        if self.debug:
            tsdf = tsdf.cpu().numpy()
            for i in range(self.num_envs):
                self.extract_surface_points_from_volume(tsdf[i], f'debug/scene{i}.txt')
                # self.visualize(self.tsdf[i], self.vox_size, self.vox_origin.cpu().numpy(), f'debug/scene{i}.txt')
        return tsdf 

    def load_franka(self, hand_base_path):
        mesh_path_list = []
        for i in range(8):
            mesh_path_list.append(f'link{i}.obj')
        mesh_path_list.extend(['hand.obj', 'finger.stl','finger.stl'])

        for i in range(len(mesh_path_list)):
            mesh_path = pjoin(hand_base_path, 'meshes', 'visual', mesh_path_list[i])
            if '.obj' in mesh_path:
                preprocess_path = mesh_path.replace('meshes', 'manifoldplus')  
            else:
                preprocess_path = None 
            sdf_path = mesh_path.replace('meshes', 'sdf')
            sdf_path = sdf_path[:-4] + '.npy'
            self.load_sdf(sdf_path, mesh_path, preprocess_path)
        return 

    def preprocess_mesh(self, input_mesh_path, output_mesh_path):
        if os.path.exists(output_mesh_path):
            print(f'Find preprocessed mesh in {output_mesh_path}!')
        else:
            os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
            print(f'Preprocess mesh in {input_mesh_path} and save to {output_mesh_path}!')
            # NOTE If depth is too large, the detail will be better (e.g. 8), but may cause problems like double surface and noise point after MC.
            # If depth is too small (e.g. 3), the detail will miss.
            os.system(f"/home/jiayichen/ManifoldPlus/build/manifold --input {input_mesh_path} --output {output_mesh_path} --depth 6")
        return 

    def merge_sdf_field(self):
        self.part_num = len(self.sdf_dict_list)
        self.sdf_field_res = []
        for i in range(self.part_num):
            res = torch.tensor(self.sdf_dict_list[i]['sdf'].shape, device=self.device)
            self.sdf_field_res.append(res)
        self.sdf_field_res = torch.stack(self.sdf_field_res, dim=0)
        target_res = self.sdf_field_res.max(dim=0)[0]
        self.sdf_field = []
        self.sdf_voxel_size = []
        self.sdf_bbox_min = []
        for i in range(self.part_num):
            pad_shape = target_res - self.sdf_field_res[i]
            pad_shape = (0,pad_shape[2],0,pad_shape[1],0,pad_shape[0])          
            sdf_field = torch.tensor(self.sdf_dict_list[i]['sdf'], device=self.device)
            padded_sdf_field  = F.pad(sdf_field, pad_shape, "constant", 1)
            vox_size = torch.tensor(self.sdf_dict_list[i]['voxel_size'], device=self.device)
            bbox_min = torch.tensor(self.sdf_dict_list[i]['bbox_min'], device=self.device)
            self.sdf_field.append(padded_sdf_field)
            self.sdf_voxel_size.append(vox_size)
            self.sdf_bbox_min.append(bbox_min)
        self.sdf_field = torch.stack(self.sdf_field, dim=0).reshape(self.part_num, -1)
        self.sdf_voxel_size = torch.stack(self.sdf_voxel_size, dim=0)[None,:,None,None]
        self.sdf_bbox_min = torch.stack(self.sdf_bbox_min, dim=0)[None,:,None,:]
        self.help = torch.arange(self.part_num)[None,:,None].repeat(self.num_envs, 1, self.point_num).to(self.device)
        self.bboxResy = self.sdf_field_res.max(dim=0)[0][1]
        self.bboxResz = self.sdf_field_res.max(dim=0)[0][2] 
        self.sdf_field_res = self.sdf_field_res[None,:,None,:]
        print('sdf field shape:', target_res)
        # print('sdf voxel size:', self.sdf_voxel_size)
        return 

    def mesh2sdf(self, mesh_path):
        from kaolin.metrics.trianglemesh import point_to_mesh_distance
        from kaolin.ops.mesh import index_vertices_by_faces, check_sign
        print(f'Use kaolin to compute SDF of {mesh_path}')
        # load meth
        obj_mesh = trimesh.load(mesh_path, force='mesh')
        vertices = torch.FloatTensor(obj_mesh.vertices).reshape(1, -1, 3).to(self.device)
        faces = torch.LongTensor(obj_mesh.faces).to(self.device)
        x1 = (faces[:,0]!=faces[:,1])&(faces[:,0]!=faces[:,2])&(faces[:,2]!=faces[:,1]) 
        faces = faces[x1.nonzero()[:,0]]    # avoid some double-vertices face

        # build volume
        obj_center = (vertices.max(dim=1)[0] + vertices.min(dim=1)[0] ) / 2 # (1, 3)
        max_range = vertices.max(dim=1)[0] - vertices.min(dim=1)[0] + 2 * self.pre_store_sdf_trunc  # (1, 3)
        volume_shape = torch.ceil(max_range / self.pre_store_sdf_voxel_size) # (1, 3)
        size_x, size_y, size_z = int(volume_shape[0, 0]), int(volume_shape[0, 1]), int(volume_shape[0, 2])

        total_length = size_x * size_y * size_z
        query_points = torch.arange(total_length)[:,None].repeat(1,3).to(self.device)
        query_points[:, 2] = query_points[:, 2] % size_z
        query_points[:, 1] = query_points[:, 1] // size_z % size_y
        query_points[:, 0] = query_points[:, 0] // size_z // size_y
        query_points = (query_points - volume_shape//2) * self.pre_store_sdf_voxel_size + obj_center

        # query the sdf value of each voxel. 
        face_vertices = index_vertices_by_faces(vertices, faces)
        distance, _, _ = point_to_mesh_distance(query_points[None,...], face_vertices)  # NOTE the returned distance is squared distance!
        sign = check_sign(vertices, faces, query_points[None,...])
        sdf = (-2*sign+1) * torch.sqrt(distance)

        sdf = torch.clamp(sdf, -self.pre_store_sdf_trunc, self.pre_store_sdf_trunc)
        save_dict = {
            'sdf': sdf.cpu().numpy().reshape(size_x, size_y, size_z), 
            'bbox_min':query_points.cpu().numpy().min(axis=0), 
            'voxel_size': self.pre_store_sdf_voxel_size, 
        }
        return save_dict

    def triplet_interpolation_query_parallel(self, query_points):
        '''
        query_points: [b, m, n, 3]
        This function will also use following variables.
            self.sdf_field: [m, x*y*z] 
            self.sdf_field_res: [m, 3]
            self.sdf_voxel_size: [m]
            self.sdf_bbox_min: [m, 3]
        '''
        query_ind = (query_points - self.sdf_bbox_min) / self.sdf_voxel_size #  [b, m, n, 3]
        valid_region = (query_ind >= 1) & (query_ind-self.sdf_field_res <= -2)
        query_ind = query_ind * valid_region 
        valid_region =(valid_region.sum(dim=-1) == 3)

        long_ind = query_ind.data.long()    # [b,m,n,3]
        delta_ind = query_ind - long_ind
        x, y, z = delta_ind[..., 0], delta_ind[..., 1], delta_ind[..., 2]   # [b,m,n]

        i000 = (long_ind[..., 0] * self.bboxResy + long_ind[..., 1]) * self.bboxResz + long_ind[..., 2]
        i001 = i000 + 1
        i010 = i000 + self.bboxResz
        i011 = i001 + self.bboxResz
        i100 = i000 + self.bboxResz * self.bboxResy
        i101 = i001 + self.bboxResz * self.bboxResy
        i110 = i010 + self.bboxResz * self.bboxResy
        i111 = i011 + self.bboxResz * self.bboxResy       # [b,m,n]

        query_value = ((self.sdf_field[self.help,i000] * (1 - z) + self.sdf_field[self.help,i001] * z) * (1 - y)\
            + (self.sdf_field[self.help,i010] * (1 - z) + self.sdf_field[self.help,i011] * z) * y) * (1 - x)\
            + ((self.sdf_field[self.help,i100] * (1 - z) + self.sdf_field[self.help,i101] * z) * (1 - y)\
            + (self.sdf_field[self.help,i110] * (1 - z) + self.sdf_field[self.help,i111] * z) * y) * x
        query_value = query_value * valid_region + 1 * (~valid_region)  # [b, m, n]

        return query_value

    def query_tsdf_naive(self, pose_R, pose_T):
        '''
        This is a little slower! 8 env 12 part: 0.043s
        pose_R: [b, m, 3, 3] b: env num, m: part num
        pose_T: [b, m, 3]
        '''
        tsdf = [self.init_tsdf]
        for i in range(len(self.sdf_dict_list)):
            sdf_dict = self.sdf_dict_list[i]
            field = torch.tensor(sdf_dict['sdf'], device=self.device)
            bbox_min = torch.tensor(sdf_dict['bbox_min'], device=self.device)
            objspace_vox_coords = torch.bmm(self.vox_coords.unsqueeze(0) - pose_T[:, i].unsqueeze(1), pose_R[:, i])  # [b, n, 3]
            single_tsdf = self.triplet_interpolation_query(objspace_vox_coords.reshape(-1, 3), field, sdf_dict['voxel_size'], bbox_min)
            tsdf.append(single_tsdf.reshape(self.num_envs, self.point_num))
        tsdf = torch.stack(tsdf, dim=0)
        tsdf = tsdf.min(dim=0)[0]
        tsdf = torch.clamp(tsdf / self.sdf_trunc, -1, 1)
        tsdf = tsdf.reshape(self.num_envs, self.resolution, self.resolution, self.resolution)

        if self.debug:
            tsdf = tsdf.cpu().numpy()
            for i in range(self.num_envs):
                self.extract_surface_points_from_volume(tsdf[i], f'debug/scene{i}.txt')
                # self.visualize(self.tsdf[i], self.vox_size, self.vox_origin.cpu().numpy(), f'debug/scene{i}.txt')
        return tsdf 

    def triplet_interpolation_query(self, query_points, field, voxel_scale, bbox_min):
        '''
        query_points: [N, 3]
        field: [x, y, z] the shape is different for each obj! So can't parallel. TODO One solution is use the same size.
        voxel_scale: int
        bbox_min: [3]
        '''
        query_ind = (query_points - bbox_min[None,:]) / voxel_scale
        valid_region = (query_ind[:, 0] >= 1) & (query_ind[:, 0] <= field.shape[0]-2) \
                &  (query_ind[:, 1] >= 1) & (query_ind[:, 1] <= field.shape[1]-2) \
                &  (query_ind[:, 2] >= 1) & (query_ind[:, 2] <= field.shape[2]-2) 
        query_ind = query_ind * valid_region[..., None] 
        
        long_ind = query_ind.data.long()
        delta_ind = query_ind - long_ind
        x, y, z = delta_ind[:, 0], delta_ind[:, 1], delta_ind[:, 2]
        
        bboxResx, bboxResy, bboxResz = field.shape
        i000 = (long_ind[:, 0] * bboxResy + long_ind[:, 1]) * bboxResz + long_ind[:, 2]
        i001 = i000 + 1
        i010 = i000 + bboxResz
        i011 = i001 + bboxResz
        i100 = i000 + bboxResz * bboxResy
        i101 = i001 + bboxResz * bboxResy
        i110 = i010 + bboxResz * bboxResy
        i111 = i011 + bboxResz * bboxResy
        
        query_value = ((field[i000] * (1 - z) + field[i001] * z) * (1 - y)\
            + (field[i010] * (1 - z) + field[i011] * z) * y) * (1 - x)\
            + ((field[i100] * (1 - z) + field[i101] * z) * (1 - y)\
            + (field[i110] * (1 - z) + field[i111] * z) * y) * x
        query_value = query_value * valid_region + 1 * (~valid_region)
        
        return query_value

    def visualize(self, sdf_field, voxel_size, bbox_min, save_path):
        verts, _, _, _ = measure.marching_cubes_lewiner(sdf_field, level=0)
        verts = verts * voxel_size + bbox_min
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path, verts)
        print(f'DEBUG: save pc to {save_path}')
        return 

    def extract_surface_points_from_volume(self, vol, save_path):
        assert len(vol.shape) == 3  
        thres_high = 0.2 
        thres_low = -0.2 
        ind = np.transpose(np.array(np.where((vol < thres_high)&(vol > thres_low)))) 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path, ind)
        print(f'DEBUG: save pc to {save_path}')
        return 

if __name__ == '__main__':

    device = 'cuda:6'
    num_envs = 64

    tsdf = TSDFfromMesh(num_envs, 0.5, 50, device)
    
    import time
    count = 20
    st = 0
    for i in range(count):
        # load pose file and query sdf in the scene
        pose_path = f'./logs/scene_pose/grasp_cube_state_ppo/control_tip_large_lift_reward_seed8177/Iter9000/{80+i}.npy'
        pose_dict = np.load(pose_path, allow_pickle=True).item()
        pose_R = torch.tensor(pose_dict['rot'][:num_envs], device=device)   # [b, m, 3, 3] b: env num, m: mesh num
        pose_T = torch.tensor(pose_dict['pos'][:num_envs], device=device)   # [b, m, 3]
        t = time.time()
        sdf_volume = tsdf.query_tsdf(pose_R, pose_T)    # [b, r, r, r]
        print('for test speed(no use)',sdf_volume.mean())
        st += time.time() - t
    print('average speed:', st/count)
    
   
    

