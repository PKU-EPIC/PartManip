from lib2to3.pgen2.token import OP
from optparse import Option
import types
from typing import Optional, Union
import torch
import numpy as np
from dataclasses import dataclass
from PointCloudViewer import PointcloudVisualizer
import open3d as o3d
from time import sleep

@dataclass
class PointCloudFrames():
    pointcloud_frames : Union[torch.Tensor, np.ndarray]

    def to_np(self):
        
        if type(self.pointcloud_frames) == torch.Tensor:
            self.pointcloud_frames = self.pointcloud_frames.cpu().numpy()
    
    def visu(self, sleep_time = 0.1):
        self.to_np()
        
        self.viewer = PointcloudVisualizer()
        self.sleep_time = sleep_time
        self.num_frames = self.pointcloud_frames.shape[0]
        pcd = o3d.geometry.PointCloud()
        # import pdb
        # pdb.set_trace()
        pcd.points = o3d.utility.Vector3dVector(self.pointcloud_frames[0, :, :3])
        pcd.colors = o3d.utility.Vector3dVector(self.pointcloud_frames[0, :, 3:6])
        self.viewer.add_geometry(pcd)
        for frame in range(self.num_frames):
            points = self.pointcloud_frames[frame]
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
            self.viewer.update(pcd)
            sleep(self.sleep_time)
        sleep(100)



if __name__ == "__main__":
    # data = torch.load("/data2/haoran/StorageFurniture-45696-link_1-handle_3-joint_1-handlejoint_3.tar")
    data = np.load("/data2/haoran/StorageFurniture-46443-link_1-handle_5-joint_1-handlejoint_5.npy", allow_pickle=True).item()
    import pdb
    pdb.set_trace()
    pcs = data["pcs"]
    # pcs = torch.ones((3,200,400,6))*0.5
    
    pc_f = PointCloudFrames(pcs[0])
    pc_f.visu()
    print(pcs.shape)