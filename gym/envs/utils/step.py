import torch
from .get_observation import _refresh_observation
from isaacgym import gymapi
# @TimeCounter
def step(task, actions) :
    # print(actions[:, :3])
    task._perform_actions(actions)
    
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    
    if not task.headless :
        task.render()
    if task.cfg["env"]["enableCameraSensors"] == True:
        task.gym.step_graphics(task.sim)

    task._refresh_observation()

    reward, done ,dist_tip, grasp_success_rate = task._get_reward_done()

    done = task.reset_buf.clone()
    success = task.success.clone()
    ###在这里记录log的信息, 不会因为reset丢失
    
    task.extras["successes"] = success
    task.extras["success_rate"] = task.success_rate
    task.extras["success_entropy"] = task.success_entropy
    if task.cfg["task"]["task"] == 'open':
        task.extras["grasp_success_rate"] = grasp_success_rate
    #task.extras["current_success_rate"] = task.current_success_rate
    task.extras["dist_tip"] = dist_tip
    #task.extras["part_pos"] = task.cabinet_dof_tensor[:, 0]
    #task.extras["part_position"] = task.cabinet_dof_tensor[:, 0]
    #print(task.reset_buf)
    task._partial_reset(task.reset_buf)

    if task.average_reward == None :
        task.average_reward = task.rew_buf.mean()
    else :
        task.average_reward = task.rew_buf.mean() * 0.01 + task.average_reward * 0.99
    task.progress_buf += torch.tensor([1], device = task.device)
    if task.cfg["save_video"]:
        task.save_video(last_frame = (task.progress_buf[0] == 199))

    # if task.progress_buf[0] == 199:
    #     # cam_pos = gymapi.Vec3(60, 18, 15)
    #     # cam_target = gymapi.Vec3(20, 18, 0)
    #     cam_pos = gymapi.Vec3(19.2, 16.2, 1.9)
    #     cam_target = gymapi.Vec3(17.8, 17.5, 1.2)
    #     task.gym.viewer_camera_look_at(
    #         task.viewer, None, cam_pos, cam_target)
    # elif task.progress_buf[0] == 0:
    #     cam_pos = gymapi.Vec3(1.4, -1.3, 1.7)
    #     cam_target = gymapi.Vec3(0, 0, 0.7)
    #     task.gym.viewer_camera_look_at(
    #             task.viewer, None, cam_pos, cam_target)
    return task.obs_buf, task.rew_buf, done, None

def control_step(task, actions):
    
    # print(actions[:, :3])
    task._perform_actions(actions)
    
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    
    if not task.headless :
        task.render()
    if task.cfg["env"]["enableCameraSensors"] == True:
        task.gym.step_graphics(task.sim)
    #task.progress_buf += torch.tensor([1], device = task.device)

    _refresh_observation(task)

    # cam_pos = gymapi.Vec3(60, 21, 15)
    # cam_target = gymapi.Vec3(35, 21, 0)
    # task.gym.viewer_camera_look_at(
    #                  task.viewer, None, cam_pos, cam_target)
    # # print(task.step_heu)
    # task.gym.write_viewer_image_to_file(task.viewer, "/data2/ziming/video/heuristic/{}.png".format(task.step_heu))
    # task.step_heu += 1

    # reward, done ,dist_tip, grasp_success_rate = task._get_reward_done()

    # done = task.reset_buf.clone()
    # success = task.success.clone()

    # task._partial_reset(task.reset_buf)
