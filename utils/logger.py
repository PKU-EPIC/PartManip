from re import L
import wandb 
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import os 

class Logger:
    def __init__(self, cfg, exp_name, task_name, algo_name):
        cfg_log = cfg['log']

        self.max_iterations = cfg['algo']['max_iterations']
        self.wandb_flag = False 
        self.tensorboard_flag = False  
        self.screen_flag = False 

        group_name = f"{task_name}_{algo_name}" if cfg_log['group'] is None else cfg_log['group']
        id_name = exp_name if cfg_log['id'] is None else cfg_log['id']

        # for algos to save model
        self.save_ckpt_dir = pjoin(cfg_log['log_root'], 'ckpts', group_name, id_name)
        self.save_pose_dir = pjoin(cfg_log['log_root'], 'scene_pose', group_name, id_name)
        self.save_video_dir = pjoin(cfg_log['log_root'], 'video', group_name, id_name)

        if 'wandb' in cfg_log['mode']:
            if cfg['resume']:
                wandb_resume = "allow"
            else:
                wandb_resume = None 
            cfg_wandb = cfg_log['wandb']
            wandb.init(
                dir=cfg_log['log_root'],
                project=cfg_wandb['project'],
                entity=cfg_wandb['entity'],
                group=group_name,
                id=id_name,
                config=cfg,         # record all cfgs
                sync_tensorboard=True,
                monitor_gym=True,   
                resume=wandb_resume,   
                # mode="offline"  
                )
            self.wandb_flag = True 
            print(f"save wandb log to {pjoin(cfg_log['log_root'], 'wandb')}")

        # if use wandb, no need to use tensorboard any more
        if 'tensorboard' in cfg_log['mode'] and 'wandb' not in cfg_log['mode']:
            tensorboard_dir = pjoin(cfg_log['log_root'], 'tensorboard', group_name, id_name)
            self.writer = SummaryWriter(log_dir=tensorboard_dir) 
            self.tensorboard_flag = True 
            print(f"save tensorboard log to {tensorboard_dir}")

        if 'screen' in cfg_log['mode']:
            self.pad = 35
            self.width = 80
            self.screen_flag = True 

    def info(self, record_dict, iteration):
        if self.wandb_flag:
            wandb.log(data=record_dict, step=iteration)

        if self.tensorboard_flag:
            for k,v in record_dict.items():
                self.writer.add_scalar(k, v, iteration)
        
        if self.screen_flag:
            log_str = '#' * self.width + '\n'
            log_str += f" \033[1m Learning iteration {iteration}/{self.max_iterations} \033[0m ".center(self.width, ' ') + '\n\n'
            for k,v in record_dict.items():
                log_str += f"{k:<{self.pad}}: {v:.6f} \n"
            print(log_str)
        return 

    def update_resume_path(self, resume_path):
        return pjoin(os.path.dirname(self.save_ckpt_dir), resume_path)

