
import gym
from mjrl.utils.gym_env import GymEnv
import mj_envs
from hand_imitation.env.environments.dapg_env.dapg_wrapper import DAPGWrapper
from hand_imitation.env.environments.shapenet_pointcloud_relocate_env import SHAPENETPCRelocate


def get_environment(env_name=None, **kwargs):
    if env_name is None: 
        print("Need to specify environment name")
        return
    # env format task-obj-size
    env_info = env_name.split('-')
    task = env_info[0]
    obj_name = env_info[1]
    obj_scale = float(env_info[2])

    if 'density' in kwargs:
        density = kwargs['density']
    else:
        density = 1000
    env = SHAPENETPCRelocate(has_renderer=False, friction=(1, 0.5, 0.01),
                    category=obj_name, obj_poses=f'shapenet_{obj_name}_poses.npy', density=density,
                    object_scale=obj_scale)
    return DAPGWrapper(env)