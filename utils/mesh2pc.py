'''
Paralelly compute SDF from mesh using pytorch/kaolin.
Define a volume first and query the SDF value of each voxel and finally save it.

'''

import numpy as np
import torch 
import trimesh 
from os.path import join as pjoin 

class PCfromMesh:
    def __init__(self, num_envs, device, num_points=1024):
        self.num_envs = num_envs
        self.device = device
        self.num_points = num_points
        self.all_pc = []

        # load franka
        hand_base_path = '/home/jiayichen/NerfRL/assets/franka_description'
        self.load_franka(hand_base_path)

        # load obj
        obj_mesh_path = '/home/jiayichen/NerfRL/assets/objs/cube/cube.obj' 
        self.load_pc_from_mesh(obj_mesh_path)

        self.all_pc = torch.stack(self.all_pc, dim=0)   # [m, p, 3]
        self.all_pc = self.all_pc.unsqueeze(0).repeat(self.num_envs, 1, 1, 1).reshape(-1, self.num_points, 3)   #[b*m, p, 3]
        return 
    

    def load_pc_from_mesh(self, mesh_path):
        '''
        load mesh and sample point cloud on the surface
        '''
        mesh = trimesh.load(mesh_path, force='mesh')        
        pc,_ = trimesh.sample.sample_surface(mesh, self.num_points)
        print(pc.shape)
        pc_tensor = torch.tensor(pc, device=self.device).float() 
        self.all_pc.append(pc_tensor)
        return 
    

    def load_franka(self, hand_base_path):
        mesh_path_list = []
        for i in range(8):
            mesh_path_list.append(f'link{i}.obj')
        mesh_path_list.extend(['hand.obj', 'finger.stl','finger.stl'])

        for i in range(len(mesh_path_list)):
            mesh_path = pjoin(hand_base_path, 'meshes', 'visual', mesh_path_list[i])
            self.load_pc_from_mesh(mesh_path)
        return 


    def query_pc(self, pose_R, pose_T):
        '''
        pose_R: [b, m, 3, 3] b: env num, m: part num
        pose_T: [b, m, 3]
        '''
        posed_pc = torch.bmm(self.all_pc, pose_R.reshape(-1, 3, 3).transpose(-1,-2)) + pose_T.reshape(-1, 1, 3)
        posed_pc = posed_pc.reshape(self.num_envs, -1, 3) # [b, m*p, 3]
        randperm = torch.randperm(posed_pc.shape[1])
        posed_pc = posed_pc[:, randperm[:self.num_points], :] # [b, p, 3]
        return posed_pc

    

if __name__ == '__main__':

    device = 'cuda:6'
    num_envs = 64

    PC = PCfromMesh(num_envs, device)
    
    import time
    count = 20
    st = 0
    for i in range(count):
        # load pose file and query sdf in the scene
        pose_path = f'/home/jiayichen/NerfRL/logs/scene_pose/grasp_cube_state_ppo/control_tip_large_lift_reward_seed8177/Iter9000/{80+i}.npy'
        pose_dict = np.load(pose_path, allow_pickle=True).item()
        pose_R = torch.tensor(pose_dict['rot'][:num_envs], device=device)   # [b, m, 3, 3] b: env num, m: mesh num
        pose_T = torch.tensor(pose_dict['pos'][:num_envs], device=device)   # [b, m, 3]
        t = time.time()
        posed_pc = PC.query_pc(pose_R, pose_T)    # [b, p, 3]
        np.savetxt(f'debug{i}.txt', posed_pc[5].cpu().numpy())
        print('for test speed(no use)',posed_pc.mean())
        st += time.time() - t
    print('average speed:', st/count)
    
   
    

