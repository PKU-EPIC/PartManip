import torch 
from skimage import measure
import numpy as np 
import math 

class TSDFVolume(object):
    """
    Integration of multiple depth images using a TSDF.
    Default size = 0.5 (workspace)
    Default resolution = 50 (50*50*50)
    
    """

    def __init__(self, device, size=0.5, resolution=50, _vol_origin=[-0.25, -0.25, -0.0503]):
        self._size = size
        self._resolution = resolution
        self._voxel_size = self._size / self._resolution
        self._sdf_trunc = 4 * self._voxel_size
        self.device = device

        help = torch.arange(0, self._resolution)
        xv, yv, zv = torch.meshgrid(help, help, help)
        self._vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(self.device)
        
        # Convert voxel coordinates to world coordinates. 
        self._vol_origin = torch.tensor(_vol_origin, device=self.device)
        self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords)
        # self._world_c = torch.cat([self._world_c, torch.ones(len(self._world_c), 1, device=self.device)], dim=1).float() # [n^3, 4]

        self.default_tsdf = 1

    def register_camera(self, cam_pose, cam_intr, im_h, im_w, num_env):
        '''
        cam_pose: [m, 4, 4], m: different view number
        cam_intr: [3, 3]
        im_h: int
        im_w: int
        num_env: int
        '''
        num_view = cam_pose.shape[0]
        self.registered_shape = (num_env, num_view, im_h, im_w)
        cam_pose = torch.tensor(cam_pose, device=self.device).float()
        self.cam_pose = cam_pose.unsqueeze(0).repeat(num_env,1,1,1).reshape(-1, 4, 4) 
        self.cam_intr = cam_intr

        world_c = self._world_c[None,...].repeat(cam_pose.shape[0], 1, 1)
        cam_c = torch.bmm(world_c-cam_pose[:,:3,3].unsqueeze(-2), cam_pose[:,:3,:3])

        # Convert camera coordinates to pixel coordinates
        fx, fy = cam_intr[0, 0], cam_intr[1, 1]
        cx, cy = cam_intr[0, 2], cam_intr[1, 2]
        pix_z = cam_c[..., 2]   # [m, n^3]
        pix_x = torch.round((cam_c[..., 0] * fx / cam_c[..., 2]) + cx).long()
        pix_y = torch.round((cam_c[..., 1] * fy / cam_c[..., 2]) + cy).long()
        self.pix_z = pix_z

        self.valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
        self.valid_pix_env = torch.tensor(range(num_env), device=self.device)[:,None,None].repeat(1, num_view, self.valid_pix.shape[-1])
        self.valid_pix_view = torch.tensor(range(num_view), device=self.device)[None,:,None].repeat(num_env, 1, self.valid_pix.shape[-1])
        self.valid_pix_y = torch.where(self.valid_pix, pix_y, 0)    
        self.valid_pix_x = torch.where(self.valid_pix, pix_x, 0)      

        # for pc 
        self.xmap = torch.tensor([[j for i in range(im_w)] for j in range(im_h)]).flatten().to(self.device)
        self.ymap = torch.tensor([[i for i in range(im_w)] for j in range(im_h)]).flatten().to(self.device)
        return 

    def integrate(self, depth_im):
        '''
        depth_im: [b, m, h, w] (must be correspond to self.register_camera()) b: env num
        '''

        assert depth_im.shape == self.registered_shape

        # Eliminate pixels outside view frustum
        depth_val = depth_im[self.valid_pix_env, self.valid_pix_view, self.valid_pix_y, self.valid_pix_x]
        
        # Integrate tsdf
        depth_diff =  depth_val - self.pix_z 
        tsdf = torch.clamp(depth_diff / self._sdf_trunc, max=1) # [m, n^3]
        valid_pts = self.valid_pix & (depth_val > 0) & (depth_diff >= -self._sdf_trunc) # [b,m,n^3]
        sum_valid_pts = valid_pts.float().sum(dim=1)    # [b,n^3]
        weight_vol = torch.where(valid_pts!=0, 1.0/sum_valid_pts.unsqueeze(1), torch.zeros(1, device=self.device))     # [b,m,n^3]
        self._tsdf_vol = (tsdf * weight_vol).sum(dim=1) + self.default_tsdf * (sum_valid_pts==0)    # [b,n^3]
        self._tsdf_vol = self._tsdf_vol.reshape(self.registered_shape[0], self._resolution, self._resolution, self._resolution)
        return self._tsdf_vol

    def sparse_voxel(self, depth_im):

        assert depth_im.shape == self.registered_shape

        # Eliminate pixels outside view frustum
        depth_val = depth_im[self.valid_pix_env, self.valid_pix_view, self.valid_pix_y, self.valid_pix_x]
        
        # Integrate tsdf
        depth_diff =  depth_val - self.pix_z 
        tsdf = torch.clamp(depth_diff / self._sdf_trunc, max=1) # [m, n^3]
        valid_pts = self.valid_pix & (depth_val > 0) & (depth_diff >= -self._sdf_trunc) # [b,m,n^3]
        sum_valid_pts = valid_pts.float().sum(dim=1)    # [b,n^3]
        weight_vol = torch.where(valid_pts!=0, 1.0/sum_valid_pts.unsqueeze(1), torch.zeros(1, device=self.device))     # [b,m,n^3]
        _tsdf_vol = (tsdf * weight_vol).sum(dim=1) + self.default_tsdf * (sum_valid_pts==0)    # [b,n^3]
        _tsdf_vol = _tsdf_vol.reshape(self.registered_shape[0], self._resolution, self._resolution, self._resolution)

        thres_high = 0.2
        thres_low = -0.2

        all_ind = []
        from pytorch3d.ops import sample_farthest_points

        for i in range(_tsdf_vol.shape[0]):
            ind = torch.where((_tsdf_vol[i] < thres_high)&(_tsdf_vol[i] > thres_low))
            ind = torch.stack(ind, dim=-1)
            final_ind, _ = sample_farthest_points(ind.unsqueeze(0), K=1024)
            all_ind.append(final_ind)

        all_ind = torch.cat(all_ind, dim=0).long()      # [b, 1024, 3]
        help = torch.arange(0, all_ind.shape[0]).unsqueeze(-1).repeat(1, all_ind.shape[1]).long()
        vol_sparse_tsdf = _tsdf_vol[help, all_ind[..., 0], all_ind[..., 1], all_ind[..., 2]]
        vol_sparse = torch.cat((all_ind, vol_sparse_tsdf.unsqueeze(-1)), dim=-1)
        return vol_sparse
    
    def extract_point_cloud(self):
        """Extract a point cloud from the voxel volume.
        """
        pc_list = []
        for i in range(self._tsdf_vol.shape[0]):
            tsdf_vol = self._tsdf_vol[i].cpu().numpy()
            vol_origin = self._vol_origin.cpu().numpy()

            # Marching cubes
            verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
            pc = verts*self._voxel_size + vol_origin
            pc_list.append(pc)
        return pc_list

    def depth2pc(self, depth_im):
        '''
        depth_im: [b, m, h, w] (must be correspond to self.register_camera()) b: env num
        '''
        from pytorch3d.ops import sample_farthest_points

        assert depth_im.shape == self.registered_shape
            
        pt2 = depth_im.reshape(self.registered_shape[0], self.registered_shape[1], -1)
        cam_cx, cam_cy = self.cam_intr[0, 2], self.cam_intr[1, 2]
        cam_fx, cam_fy = self.cam_intr[0, 0], self.cam_intr[1, 1]
        pt0 = (self.ymap - cam_cx) * pt2 / cam_fx
        pt1 = (self.xmap - cam_cy) * pt2 / cam_fy
        cld = torch.stack((pt0, pt1, pt2), dim=-1) # [b, m, h*w, 3]
        cld = cld.reshape(self.registered_shape[0]*self.registered_shape[1], -1, 3)  # [b*m, h*w, 3]

        world_cld = torch.bmm(cld, self.cam_pose[:, :3, :3].transpose(-1,-2)) + self.cam_pose[:,:3,3].unsqueeze(-2)  # [b*m, h*w, 3]
        world_cld = world_cld.reshape(self.registered_shape[0], -1, 3)  # [b, m*h*w, 3]
        # valid_mask = (world_cld < self._size + self._vol_origin) & (world_cld > self._vol_origin + torch.tensor([0,0,0.055], device=self.device)) 
        valid_mask = (world_cld < self._size + self._vol_origin) & (world_cld > self._vol_origin) 
        valid_mask = (valid_mask.sum(dim=-1, keepdim=True) == 3)    # [b, m*h*w]

        # # slow.. ~0.5s for [64, 6, 180, 320]
        world_cld = world_cld * valid_mask
        final_pc, _ = sample_farthest_points(world_cld, K=1024)

        # # not accurate. can't see cube. 
        # pc_list = []
        # for i in range(self.registered_shape[0]):
        #     pc = world_cld[i, valid_mask[i,:,0]]
        #     randnum = torch.randperm(pc.shape[0])
        #     pc = pc[randnum[:1024]]
        #     pc_list.append(pc)
        # final_pc = torch.stack(pc_list, dim=0)
        # np.savetxt('debug.txt', final_pc[0].cpu().numpy())
        # np.savetxt('debug0.txt', final_pc[1].cpu().numpy())
        # exit(1)
        return final_pc 

if __name__ == '__main__':
    img_width = 640
    img_height = 320
    horizontal_fov = 69.75
    fx = img_width / 2.0 / math.tan((horizontal_fov / 180.0 * math.pi) / 2.0)
    intrinsic = np.array([[fx, 0, img_width//2],
                        [0, fx, img_height//2],
                        [0, 0, 1]])

    tsdf = TSDFVolume()
