import numpy as np
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

if __name__ == "__main__" :

	visualizer = PointcloudVisualizer()
	cloud = o3d.io.read_point_cloud("/data2/haoran/RL-Pose/MyGym/assets/4.ply")
	visualizer.add_geometry(cloud)
	while True :
		print("update")
		visualizer.update(cloud)
		xyz = np.asarray(cloud.points)
