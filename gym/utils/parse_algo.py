from gc import collect
from re import I
import torch.nn as nn
import os

import wandb

def process_dagger_ppo(args, env, cfg_train, log_dir, wandb_writer):
    from algorithms.pregrasp_ppo import DAGGER_PPO, ActorCriticPC #, ActorCriticPCNew
    learn_cfg = cfg_train["learn"]
    is_testing = False
    ckpt_path = ''

    if args.model_dir != "":
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    if args.ckpt is not None:
        is_testing = learn_cfg["test"]
        
    logdir = log_dir + "_algo-seed{}".format(cfg_train["seed"])
    actor_critic_class = ActorCriticPC

    # if cfg_train["policy"]["debug"]:
    #     actor_critic_class = ActorCriticPC ############only for debug################
        
    """Set up the dagger system for training or inferencing."""
    DAGGER_PPO = DAGGER_PPO(vec_env=env,
              actor_critic_class=actor_critic_class,
              actor_critic_expert_class=actor_critic_class,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              expert_cfg=cfg_train["expert_policy"],
              discriminator_cfg=cfg_train["discriminator"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_freq=learn_cfg["eval_freq"],
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
              checkpoint_path= ckpt_path,
              canonical= cfg_train["policy"]["canonicalize"],
              canon_space= cfg_train["policy"]["canon_space"] ,
              use_residual_traj = cfg_train["policy"]["use_residual_traj"],
              use_adv_norm = learn_cfg["use_adv_norm"],           ### trick 1
              adv_norm_epsilon= learn_cfg["adv_norm_epsilon"],
              use_state_norm= learn_cfg["use_state_norm"],         ### trick 2
              use_reward_norm= learn_cfg["use_reward_norm"],        ### trick 3
              use_reward_scaling= learn_cfg["use_reward_scaling"],     ### trick 4
              learning_rate_decay = learn_cfg["learning_rate_decay"],   ### trick 6
              use_grad_clip = learn_cfg["use_grad_clip"], 
              use_orthogonal_init= cfg_train["policy"]["use_orthogonal_init"],     ### trick 8
              adam_epsilon = learn_cfg["adam_epsilon"],           ### trick 9
              )

    if is_testing and args.model_dir != "":
        DAGGER_PPO.load(ckpt_path)
        DAGGER_PPO.load_expert(cfg_train["expert_policy"]["expert_path"])
        print("Loading model from {}".format(ckpt_path))
        DAGGER_PPO.test(ckpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        DAGGER_PPO.load(ckpt_path)
    
    if not is_testing:
        # if cfg_train["policy"]["use_expert"]:
        if DAGGER_PPO.model_cfg["use_bc_pretrain"]:
            print("Loading model and demo from {}, {}".format(cfg_train["expert_policy"]["expert_path"], cfg_train["expert_policy"]["demo_path"]))
            DAGGER_PPO.load_expert(cfg_train["expert_policy"]["expert_path"], cfg_train["expert_policy"]["demo_path"])
        else:
            print("Loading model from {}".format(cfg_train["expert_policy"]["expert_path"]))
            DAGGER_PPO.load_expert(cfg_train["expert_policy"]["expert_path"])
    return DAGGER_PPO

def process_pregrasp_ppo_pn(args, env, cfg_train, logdir, wandb_writer):
    from algorithms.pregrasp_ppo import PREGRASP_PPO, ActorCriticPC
    learn_cfg = cfg_train["learn"]
    is_testing = False
    ckpt_path = ''

    if args.model_dir != "" or (args.ckpt != "" and args.ckpt != "None" and args.ckpt != None):
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    
    logdir = logdir + "_algo-seed{}".format(cfg_train["seed"])
    """Set up the PPO system for training or inferencing."""
    ppo = PREGRASP_PPO(vec_env=env,
              actor_critic_class=ActorCriticPC,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_freq=learn_cfg["eval_freq"],
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
              checkpoint_path= ckpt_path,

              canonical= cfg_train["policy"]["canonicalize"],
              canon_space= cfg_train["policy"]["canon_space"] ,
              use_residual_traj = cfg_train["policy"]["use_residual_traj"],
              use_adv_norm = learn_cfg["use_adv_norm"],           ### trick 1
              adv_norm_epsilon= learn_cfg["adv_norm_epsilon"],
              use_state_norm= learn_cfg["use_state_norm"],         ### trick 2
              use_reward_norm= learn_cfg["use_reward_norm"],        ### trick 3
              use_reward_scaling= learn_cfg["use_reward_scaling"],     ### trick 4
              learning_rate_decay = learn_cfg["learning_rate_decay"],   ### trick 6
              use_grad_clip = learn_cfg["use_grad_clip"], 
              use_orthogonal_init= cfg_train["policy"]["use_orthogonal_init"],     ### trick 8
              adam_epsilon = learn_cfg["adam_epsilon"],           ### trick 9
            
              )

    # ppo.test("/home/hp-3070/DexterousHandEnvs/dexteroushandenvs/logs/shadow_hand_lift_underarm2/ppo/ppo_seed2/model_40000.pt")
    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.test(ckpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.load(ckpt_path)

    return ppo

def process_ppo_pn(args, env, cfg_train, logdir, wandb_writer):
    # from algorithms.ppo import PPO, ActorCriticPC
    # learn_cfg = cfg_algo["learn"]
    # is_testing = False
    # ckpt_path = ""
    # if args.model_dir != "":
    #     is_testing = learn_cfg["test"]
    #     ckpt_path = args.model_dir
    
    # logdir = log_dir + "_algo-seed{}".format(cfg_algo["seed"])
    from algorithms.ppo import PPO, ActorCriticPC
    learn_cfg = cfg_train["learn"]
    is_testing = False
    ckpt_path = ''
    if args.model_dir != "":
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    #is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    
    logdir = logdir + "_algo-seed{}".format(cfg_train["seed"])
    # init algo
    ppo = PPO(vec_env=env,
              actor_critic_class=ActorCriticPC,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
            #   checkpoint= learn_cfg["checkpoint"],
              checkpoint_path= ckpt_path,

              use_adv_norm = learn_cfg["use_adv_norm"],           ### trick 1
              adv_norm_epsilon= learn_cfg["adv_norm_epsilon"],
              use_state_norm= learn_cfg["use_state_norm"],         ### trick 2
              use_reward_norm= learn_cfg["use_reward_norm"],        ### trick 3
              use_reward_scaling= learn_cfg["use_reward_scaling"],     ### trick 4
              learning_rate_decay = learn_cfg["learning_rate_decay"],   ### trick 6
              use_grad_clip = learn_cfg["use_grad_clip"], 
              use_orthogonal_init= cfg_train["policy"]["use_orthogonal_init"],     ### trick 8
              adam_epsilon = learn_cfg["adam_epsilon"],           ### trick 9
            
              )
    # ppo = PPO(
    #     vec_env=env,
    #     actor_critic_class=ActorCriticPC,
    #     num_transitions_per_env=learn_cfg["nsteps"],
    #     num_learning_epochs=learn_cfg["noptepochs"],
    #     num_mini_batches=learn_cfg["nminibatches"],
    #     clip_param=learn_cfg["cliprange"],
    #     gamma=learn_cfg["gamma"],
    #     lam=learn_cfg["lam"],
    #     init_noise_std=learn_cfg.get("init_noise_std", 0.3),
    #     value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
    #     entropy_coef=learn_cfg["ent_coef"],
    #     learning_rate=learn_cfg["optim_stepsize"],
    #     max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
    #     use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
    #     schedule=learn_cfg.get("schedule", "fixed"),
    #     desired_kl=learn_cfg.get("desired_kl", None),
    #     max_lr=learn_cfg.get("lr_upper", 1e-3),
    #     min_lr=learn_cfg.get("lr_lower", 1e-7),
    #     model_cfg=cfg_algo["network"],
    #     device=env.rl_device,
    #     sampler=learn_cfg.get("sampler", 'sequential'),
    #     log_dir=logdir,
    #     is_testing=is_testing,
    #     eval_round=learn_cfg["eval_round"],
    #     print_log=learn_cfg["print_log"],
    #     apply_reset=False,
    max_iterations =learn_cfg["max_iterations"],
    wandb_writer = wandb_writer, 
    checkpoint_path= ckpt_path,

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.test(ckpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.load(ckpt_path)
    return ppo

def process_ppo(args, env, cfg_train, logdir, wandb_writer):
    from algorithms.ppo import PPO, ActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = False
    ckpt_path = ''

    if args.model_dir != "":
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    
    logdir = logdir + "_algo-seed{}".format(cfg_train["seed"])
    
    """Set up the PPO system for training or inferencing."""
    ppo = PPO(vec_env=env,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
              checkpoint_path= ckpt_path,

              use_adv_norm = learn_cfg["use_adv_norm"],           ### trick 1
              adv_norm_epsilon= learn_cfg["adv_norm_epsilon"],
              use_state_norm= learn_cfg["use_state_norm"],         ### trick 2
              use_reward_norm= learn_cfg["use_reward_norm"],        ### trick 3
              use_reward_scaling= learn_cfg["use_reward_scaling"],     ### trick 4
              learning_rate_decay = learn_cfg["learning_rate_decay"],   ### trick 6
              use_grad_clip = learn_cfg["use_grad_clip"], 
              use_orthogonal_init= cfg_train["policy"]["use_orthogonal_init"],     ### trick 8
              adam_epsilon = learn_cfg["adam_epsilon"],           ### trick 9
            
              )

    # ppo.test("/home/hp-3070/DexterousHandEnvs/dexteroushandenvs/logs/shadow_hand_lift_underarm2/ppo/ppo_seed2/model_40000.pt")
    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.test(ckpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.load(ckpt_path)

    return ppo

def process_heuristics(args, env, cfg_train, logdir, wandb_writer):
    from algorithms.heuristics import HEURISTICS
    print(cfg_train)
    heuristics = HEURISTICS(
        vec_env = env,
        algo_cfg = cfg_train,
        
    )
    
    return heuristics

def process_sac(args, env, cfg_train, logdir, wandb_writer):
    from algorithms.sac import SAC, MLPActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt_path = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        # is_testing = True
        chkpt_path = args.model_dir
    
    logdir = logdir + "_algo-seed{}".format(cfg_train["seed"])
    
    """Set up the SAC system for training or inferencing."""
    sac = SAC(vec_env=env,
              actor_critic=MLPActorCritic,
              ac_kwargs = dict(hidden_sizes=learn_cfg["hidden_size"],model_cfg = cfg_train["policy"],),
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              replay_size = learn_cfg["replay_size"] ,
              # clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              polyak = learn_cfg["polyak"],
              learning_rate = learn_cfg["learning_rate"],
              max_grad_norm = learn_cfg.get("max_grad_norm", 2.0),
              entropy_coef = learn_cfg["ent_coef"],
              use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False),
              reward_scale=learn_cfg["reward_scale"],
              batch_size=learn_cfg["batch_size"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_freq=learn_cfg["eval_freq"],
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              wandb_writer = wandb_writer,
              checkpoint_path = chkpt_path,
              canonical= env.task.cfg["task"]["canonicalize"],
              canon_space= env.task.cfg["task"]["canon_space"] ,
              model_cfg = cfg_train["policy"], 
              )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        sac.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        sac.load(chkpt_path)

    return sac

def process_pregrasp_ppo(args, env, cfg_train, logdir, wandb_writer):
    from algorithms.pregrasp_ppo import PREGRASP_PPO, ActorCriticPC
    learn_cfg = cfg_train["learn"]
    is_testing = False
    ckpt_path = ''

    if args.model_dir != "":
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    
    logdir = logdir + "_algo-seed{}".format(cfg_train["seed"])
    
    """Set up the PPO system for training or inferencing."""
    ppo = PREGRASP_PPO(vec_env=env,
              actor_critic_class=ActorCriticPC,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_freq=learn_cfg["eval_freq"],
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
              checkpoint_path= ckpt_path,

              canonical= env.task.cfg["task"]["canonicalize"],
              canon_space= env.task.cfg["task"]["canon_space"] ,
              use_residual_traj = cfg_train["policy"]["use_residual_traj"],
              use_adv_norm = learn_cfg["use_adv_norm"],           ### trick 1
              adv_norm_epsilon= learn_cfg["adv_norm_epsilon"],
              use_state_norm= learn_cfg["use_state_norm"],         ### trick 2
              use_reward_norm= learn_cfg["use_reward_norm"],        ### trick 3
              use_reward_scaling= learn_cfg["use_reward_scaling"],     ### trick 4
              learning_rate_decay = learn_cfg["learning_rate_decay"],   ### trick 6
              use_grad_clip = learn_cfg["use_grad_clip"], 
              use_orthogonal_init= cfg_train["policy"]["use_orthogonal_init"],     ### trick 8
              adam_epsilon = learn_cfg["adam_epsilon"],           ### trick 9
            
              )

    # ppo.test("/home/hp-3070/DexterousHandEnvs/dexteroushandenvs/logs/shadow_hand_lift_underarm2/ppo/ppo_seed2/model_40000.pt")
    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.load(ckpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.load(ckpt_path)

    return ppo

def process_dagger(args, env, cfg_train, log_dir, wandb_writer):
    from algorithms.pregrasp_ppo import DAGGER, ActorCriticPC #, ActorCriticPCNew
    learn_cfg = cfg_train["learn"]
    is_testing = False
    ckpt_path = ''

    if args.model_dir != "":
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    if args.ckpt is not None:
        is_testing = learn_cfg["test"]
        
    logdir = log_dir + "_algo-seed{}".format(cfg_train["seed"])
    actor_critic_class = ActorCriticPC

    # if cfg_train["policy"]["debug"]:
    #     actor_critic_class = ActorCriticPC ############only for debug################
        
    """Set up the dagger system for training or inferencing."""
    DAGGER = DAGGER(vec_env=env,
              actor_critic_class=actor_critic_class,
              actor_critic_expert_class = actor_critic_class,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_freq=learn_cfg["eval_freq"],
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
              checkpoint_path= ckpt_path,
              canonical= cfg_train["policy"]["canonicalize"],
              canon_space= cfg_train["policy"]["canon_space"] ,
              use_residual_traj = cfg_train["policy"]["use_residual_traj"],         ### trick 9
              )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        DAGGER.test(ckpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        DAGGER.load(ckpt_path)
    
    if not is_testing:
        if cfg_train["policy"]["use_expert"]:
            DAGGER.load_expert(cfg_train["expert_policy"]["expert_path"])
        else:
            DAGGER.load_expert(cfg_train["expert_policy"]["demo_path"])

    return DAGGER

def process_ILAD(args, env, cfg_train, log_dir, wandb_writer):
    from algorithms.ILAD_baseline import ILAD, ActorCriticPC #, ActorCriticPCNew
    learn_cfg = cfg_train["learn"]
    is_testing = False
    ckpt_path = ''

    if args.model_dir != "":
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    if args.ckpt is not None:
        is_testing = learn_cfg["test"]
        
    logdir = log_dir + "_algo-seed{}".format(cfg_train["seed"])
    actor_critic_class = ActorCriticPC

    # if cfg_train["policy"]["debug"]:
    #     actor_critic_class = ActorCriticPC ############only for debug################
        
    """Set up the dagger system for training or inferencing."""
    ILAD =ILAD(vec_env=env,
              actor_critic_class=actor_critic_class,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_freq=learn_cfg["eval_freq"],
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
              checkpoint_path= ckpt_path,
              canonical= cfg_train["policy"]["canonicalize"],
              canon_space= cfg_train["policy"]["canon_space"] ,
              use_residual_traj = cfg_train["policy"]["use_residual_traj"],         ### trick 9
              )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ILAD.test(ckpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ILAD.load(ckpt_path)
    
    if not is_testing:
        if cfg_train["policy"]["use_expert"]:
            ILAD.load_expert(cfg_train["expert_policy"]["expert_path"])
        else:
            ILAD.load_expert(cfg_train["expert_policy"]["demo_path"])

    return ILAD

def process_imitation_learning(args, env, cfg_train, logdir, wandb_writer):
    from algorithms.pregrasp_ppo import imitation_learning, ActorCriticPC #, ActorCriticPCNew
    learn_cfg = cfg_train["learn"]
    is_testing = False
    ckpt_path = ''

    if args.model_dir != "":
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    if args.ckpt is not None:
        is_testing = learn_cfg["test"]

    
    ##load expert####
    # expert_path = learn_cfg["expert_dir"]
    # print("Loading model from {}".format(expert_path))
    # imitation_learning.load_expert(ckpt_path)

    logdir = logdir + "_algo-seed{}".format(cfg_train["seed"])
    # if cfg_train["policy"]["Spconv"]["freeze"]:
    #     actor_critic_class = ActorCriticPCNew
    # else:
    #     actor_critic_class = ActorCriticPC
    actor_critic_class = ActorCriticPC

    # if cfg_train["policy"]["debug"]:
    #     actor_critic_class = ActorCriticPC ############only for debug################
        
    """Set up the imitation_learning system for training or inferencing."""
    imitation_learning = imitation_learning(vec_env=env,
              actor_critic_class=actor_critic_class,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              discriminator_cfg=cfg_train["discriminator"], 
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_freq=learn_cfg["eval_freq"],
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
              checkpoint_path= ckpt_path,

              canonical= cfg_train["policy"]["canonicalize"],
              canon_space= cfg_train["policy"]["canon_space"] ,
              use_residual_traj = cfg_train["policy"]["use_residual_traj"],
              use_adv_norm = learn_cfg["use_adv_norm"],           ### trick 1
              adv_norm_epsilon= learn_cfg["adv_norm_epsilon"],
              use_state_norm= learn_cfg["use_state_norm"],         ### trick 2
              use_reward_norm= learn_cfg["use_reward_norm"],        ### trick 3
              use_reward_scaling= learn_cfg["use_reward_scaling"],     ### trick 4
              learning_rate_decay = learn_cfg["learning_rate_decay"],   ### trick 6
              use_grad_clip = learn_cfg["use_grad_clip"], 
              use_orthogonal_init= cfg_train["policy"]["use_orthogonal_init"],     ### trick 8
              adam_epsilon = learn_cfg["adam_epsilon"],           ### trick 9
            
              )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        imitation_learning.test(ckpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        imitation_learning.load(ckpt_path)
    
    if not is_testing:
        if cfg_train["policy"]["use_expert"]:
            imitation_learning.load_expert(cfg_train["expert_policy"]["expert_path"])
        else:
            imitation_learning.load_expert(cfg_train["expert_policy"]["demo_path"])

    return imitation_learning

def process_sac_il(args, env, cfg_train, logdir, wandb_writer):
    from algorithms.imitation_learning import sac_il, MLPActorCritic, MLPActorCritic_PC
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    chkpt_path = ' '

    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        # is_testing = True
        chkpt_path = args.model_dir

    if cfg_train["policy"]["debug"]:
        actor_critic_class = MLPActorCritic ############only for debug################
    else:
        actor_critic_class = MLPActorCritic_PC    
    
    logdir = logdir + "_algo-seed{}".format(cfg_train["seed"])
    
    """Set up the SAC system for training or inferencing."""
    sac_il = sac_il(vec_env=env,
              actor_critic=actor_critic_class,
              ac_kwargs = dict(hidden_sizes=[learn_cfg["hidden_nodes"]]* learn_cfg["hidden_layer"]),
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              replay_size = learn_cfg["replay_size"] ,
              # clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              polyak = learn_cfg["polyak"],
              learning_rate = learn_cfg["learning_rate"],
              max_grad_norm = learn_cfg.get("max_grad_norm", 2.0),
              entropy_coef = learn_cfg["ent_coef"],
              use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False),
              reward_scale=learn_cfg["reward_scale"],
              batch_size=learn_cfg["batch_size"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_freq=learn_cfg["eval_freq"],
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              wandb_writer = wandb_writer,
              checkpoint_path = chkpt_path,

              canonical= cfg_train["policy"]["canonicalize"],
              canon_space= cfg_train["policy"]["canon_space"],

              discriminator_cfg=cfg_train["discriminator"],  
              model_cfg=cfg_train["policy"],
              )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        sac_il.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        sac_il.load(chkpt_path)

    if not is_testing and not cfg_train["policy"]["debug"]:
        if cfg_train["policy"]["use_expert"]:
            sac_il.load_expert(cfg_train["expert_policy"]["expert_path"])
        else:
            sac_il.load_expert(cfg_train["expert_policy"]["demo_path"])


    return sac_il


def process_collect_data(args, env, cfg_train, logdir, wandb_writer):
    from algorithms.imitation_learning import collect_data, ActorCritic
    learn_cfg = cfg_train["learn"]
    is_testing = False
    ckpt_path = ''

    if args.model_dir != "":
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    
    logdir = logdir + "_algo-seed{}".format(cfg_train["seed"])

    """Set up the PPO system for training or inferencing."""
    ppo = collect_data(vec_env=env,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
              checkpoint_path= ckpt_path,

              canonical= cfg_train["policy"]["canonicalize"],
              canon_space= cfg_train["policy"]["canon_space"] ,
              use_residual_traj = cfg_train["policy"]["use_residual_traj"],
              use_adv_norm = learn_cfg["use_adv_norm"],           ### trick 1
              adv_norm_epsilon= learn_cfg["adv_norm_epsilon"],
              use_state_norm= learn_cfg["use_state_norm"],         ### trick 2
              use_reward_norm= learn_cfg["use_reward_norm"],        ### trick 3
              use_reward_scaling= learn_cfg["use_reward_scaling"],     ### trick 4
              learning_rate_decay = learn_cfg["learning_rate_decay"],   ### trick 6
              use_grad_clip = learn_cfg["use_grad_clip"], 
              use_orthogonal_init= cfg_train["policy"]["use_orthogonal_init"],     ### trick 8
              adam_epsilon = learn_cfg["adam_epsilon"],           ### trick 9
            
              )

    # ppo.test("/home/hp-3070/DexterousHandEnvs/dexteroushandenvs/logs/shadow_hand_lift_underarm2/ppo/ppo_seed2/model_40000.pt")
    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.test(ckpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.load_expert(ckpt_path)

    return ppo


def process_behavior_cloning(args, env, cfg_train, logdir, wandb_writer):
    from algorithms.imitation_learning import behavior_cloning, ActorCriticPC, ActorCritic, ActorCriticPCNew
    learn_cfg = cfg_train["learn"]
    is_testing = False
    ckpt_path = ''

    if args.model_dir != "":
        is_testing = learn_cfg["test"]
        ckpt_path = args.model_dir
    
    ##load expert####
    # expert_path = learn_cfg["expert_dir"]
    # print("Loading model from {}".format(expert_path))
    # imitation_learning.load_expert(ckpt_path)

    logdir = logdir + "_algo-seed{}".format(cfg_train["seed"])
    if cfg_train["policy"]["Spconv"]["freeze"]:
        actor_critic_class = ActorCriticPCNew
    else:
        actor_critic_class = ActorCriticPC
        
    """Set up the imitation_learning system for training or inferencing."""
    behavior_cloning = behavior_cloning(vec_env=env,
              actor_critic_class=actor_critic_class,
              actor_critic_expert_class=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              expert_cfg=cfg_train["expert_policy"],
              discriminator_cfg=cfg_train["discriminator"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
              checkpoint_path= ckpt_path,

              canonical= cfg_train["policy"]["canonicalize"],
              canon_space= cfg_train["policy"]["canon_space"] ,
              use_residual_traj = cfg_train["policy"]["use_residual_traj"],
              use_adv_norm = learn_cfg["use_adv_norm"],           ### trick 1
              adv_norm_epsilon= learn_cfg["adv_norm_epsilon"],
              use_state_norm= learn_cfg["use_state_norm"],         ### trick 2
              use_reward_norm= learn_cfg["use_reward_norm"],        ### trick 3
              use_reward_scaling= learn_cfg["use_reward_scaling"],     ### trick 4
              learning_rate_decay = learn_cfg["learning_rate_decay"],   ### trick 6
              use_grad_clip = learn_cfg["use_grad_clip"], 
              use_orthogonal_init= cfg_train["policy"]["use_orthogonal_init"],     ### trick 8
              adam_epsilon = learn_cfg["adam_epsilon"],           ### trick 9
              )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        behavior_cloning.test(ckpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(ckpt_path))
        behavior_cloning.load(ckpt_path)
    
    if cfg_train["expert_policy"]["expert_path"] is None:
        raise NotImplementedError
    else:
        behavior_cloning.load_expert(cfg_train["expert_policy"]["expert_path"])

    return behavior_cloning