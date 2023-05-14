import numpy as np
import yaml
import os
import open3d as o3d

class PointcloudVisualizer() :
    
	def __init__(self) -> None:
		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window()
		# self.vis.register_key_callback(key, your_update_function)
	
	def add_geometry(self, cloud) :
		self.vis.add_geometry(cloud)

	def update(self, cloud):
		self.vis.update_geometry(cloud)
		self.vis.update_renderer()
		self.vis.poll_events()

def save(task, path, iteration) :

    save_dict = task.cfg
    cabinet_success_rate = task.success_rate.view(task.cabinet_num, task.env_per_asset).mean(dim=1)
    cabinet_train_success_rate = cabinet_success_rate[:task.cabinet_num_train]
    cabinet_valIntra_success_rate = cabinet_success_rate[task.cabinet_num_train:task.cabinet_num_train+task.cabinet_num_valIntra]
    cabinet_valInter_success_rate = cabinet_success_rate[task.cabinet_num_train+task.cabinet_num_valIntra:]
    for id, (name, tensor) in enumerate(zip(task.select_train_asset_paths, cabinet_train_success_rate)) :
        save_dict["env"]["asset"]["result"][f"trainAssets_{name}_successRate"] = tensor.cpu().item()
        save_dict["env"]["asset"]["result"][f"trainAssets{name}_envIds"] = id * task.env_per_asset
    for id, (name, tensor) in enumerate(zip(task.select_valIntra_asset_paths, cabinet_valIntra_success_rate)) :
        save_dict["env"]["asset"]["result"][f"valIntraAssets_{name}_successRate"] = tensor.cpu().item()
        save_dict["env"]["asset"]["result"][f"valntraAssets{name}_envIds"] = id * task.env_per_asset
    for id, (name, tensor) in enumerate(zip(task.select_valInter_asset_paths, cabinet_valInter_success_rate)) :
        save_dict["env"]["asset"]["result"][f"valInterAssets_{name}_successRate"] = tensor.cpu().item()
        save_dict["env"]["asset"]["result"][f"valnterAssets{name}_envIds"] = id * task.env_per_asset
    with open(os.path.join(path, "cfg_{}.yaml".format(iteration)), "w") as f:
        yaml.dump(save_dict, f)

def _draw_line(task, src, dst, clear = True, env_id = 0):
    line_vec = np.stack([src, dst]).flatten().astype(np.float32)
    color = np.array([1,0,0], dtype=np.float32)
    if clear:
        task.gym.clear_lines(task.viewer)
    # print(env_id)
    task.gym.add_lines(
        task.viewer,
        task.env_ptr_list[env_id],
        task.env_num,
        line_vec,
        color
    )
    
def draw_bbox(img, bbox_list, K = None):
    for i,bbox in enumerate(bbox_list):
        if len(bbox) == 0:
            continue
        # bbox = bbox * trans[0]+trans[1:4]
        if K == None:
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