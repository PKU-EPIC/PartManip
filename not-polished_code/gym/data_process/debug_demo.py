import torch
import numpy as np
import os
qpose_num = 16
demo = torch.load("/data2/ziming/RL-Pose/PoseOrientedGym/expert_data/pc/door/StorageFurniture-46037-link_0-handle_0-joint_0-handlejoint_0.tar")
asset_name = "StorageFurniture-46037-link_0-handle_0-joint_0-handlejoint_0"
asset_num = 2
demo_pc = demo["pcs"][asset_num]
demo_obs = demo["observations"][asset_num]
demo_actions = demo["actions"][asset_num]
if not os.path.exists("/data2/ziming/exp/{}".format(asset_name)):
    os.makedirs("/data2/ziming/exp/{}".format(asset_name))
if not os.path.exists("/data2/ziming/exp/{}/{}".format(asset_name, asset_num)):
    os.makedirs("/data2/ziming/exp/{}/{}".format(asset_name, asset_num))
np.savetxt("/data2/ziming/exp/{}/{}/pc0.txt".format(asset_name, asset_num), demo_pc[0].cpu().numpy())
np.savetxt("/data2/ziming/exp/{}/{}/pc1.txt".format(asset_name, asset_num), demo_pc[1].cpu().numpy())
np.savetxt("/data2/ziming/exp/{}/{}/pc2.txt".format(asset_name, asset_num), demo_pc[2].cpu().numpy())
np.savetxt("/data2/ziming/exp/{}/{}/pc100.txt".format(asset_name, asset_num), demo_pc[100].cpu().numpy())
np.savetxt("/data2/ziming/exp/{}/{}/pc200.txt".format(asset_name, asset_num), demo_pc[-1].cpu().numpy())
np.savetxt("/data2/ziming/exp/{}/{}/handle.txt".format(asset_name, asset_num), demo_obs[:, qpose_num+25:qpose_num+28].cpu().numpy())
np.savetxt("/data2/ziming/exp/{}/{}/part.txt".format(asset_name, asset_num), demo_obs[:, qpose_num+37:qpose_num+40].cpu().numpy())
np.savetxt("/data2/ziming/exp/{}/{}/frankaxyz.txt".format(asset_name, asset_num), demo_obs[:, qpose_num:qpose_num+3].cpu().numpy())
np.savetxt("/data2/ziming/exp/{}/{}/actionsxyz.txt".format(asset_name, asset_num), demo_actions[:, 0:3].cpu().numpy())
np.savetxt("/data2/ziming/exp/{}/{}/actionstips.txt".format(asset_name, asset_num), demo_actions[:, 7:9].cpu().numpy())


