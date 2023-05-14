from .config import warn_task_name

from envs.base.vec_task import VecTaskPythonArm, VecTaskPythonArmPC
from envs.franka_pose_cabinet_base import FrankaPoseCabinetBase


def load_env(args, cfg, cfg_algo, sim_params, log_dir):

    # device info
    device_id = args.device_id
    rl_device = args.sim_device
    print(device_id, rl_device)
    # seed
    cfg["seed"] = cfg_algo
    cfg_env = cfg["env"]
    cfg_env["seed"] = cfg["seed"]

    log_dir = log_dir + "_seed{}".format(cfg["seed"])
    try:
        task = eval(args.task)(
            cfg = cfg,
            sim_params = sim_params,
            physics_engine = args.physics_engine,
            device_type=args.device,
            device_id = device_id,
            headless = args.headless,
            log_dir = log_dir,
        )
    
    except NameError as e:
        print(e)
        warn_task_name()
    if args.task == "FrankaPoseCabinetBase":
        print("FrankaPoseCabinetBase")
        env = VecTaskPythonArm(task, rl_device)

    if args.task == "FrankaPoseCabinetPC":
        env = VecTaskPythonArmPC(task,rl_device)
    
    return task, env