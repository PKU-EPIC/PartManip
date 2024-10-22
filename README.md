# Two-Finger Gripper Manipulation Tasks

Use RL to finish manipulation tasks based on [Isaac Gym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).


### Train and Test

Basic command is 

```
# Train
python train.py --algocfg {$ALGO} --taskcfg {$TASK} --exp_name {$EXP} --device_id {$GPU} 
# e.g. python train.py --algocfg ppo --taskcfg open_drawer --exp_name first_try --device_id 0 --log.mode wandb

# Test
python train.py --exp_name {$EXP} --algocfg {$ALGO} --taskcfg {$TASK} --device_id {$GPU} --resume {$EXP}_seed{$SEED}/model_{$CKPT_NUM}.pth --test_only 
# e.g. python train.py --algocfg ppo --taskcfg open_drawer --exp_name first_try --device_id 0 --log.mode wandb --test_only --resume assets/ckpts/model_200000.pth --task.asset.splits valIntra

```
You can also overwrite any parameters in config file by adding ``` --KEY1.KEY2.KEY3 xxx```. See ```def add_cfg``` in  ```utils/config.py``` for more. 

## Implemented Algorithms and Tasks
#### Algo: PPO 
[OpenAI introduction](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

[OpenAI implementation](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo)

[ppo-implementation-details blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

[ppo-implementation-details paper](https://arxiv.org/abs/2006.05990)

Important changes:

    1. Support reset some envs. 
    
        See comments in 'def step' in hand_base.py
        
    2. If reset, set adv=0. 
    
        Reason: If reset, the returned observation is not the result of this action. So we don't know this action is good or bad. If we don't use reward aug tricks, setting adv=0 means this action will not have gradient. 
        
        See line 223 in storage.py.
        
    3. Reset if reward don't increase in previous K (hyperparameter) steps. 
    
        Reason: Tasks with dense reward don't need a very long exploring time. Once the reward stop increasing, it is likely to be trapped. Early reset can help reduce meaningless exploration. On the other hand, if the reward keeps increasing, we shouldn't reset even if current step id is bigger than max_episode_length.
        
        See 'def pre_physics_step' in hand_base.py
    
#### Algo: Dagger

reward-driven early stop can speed up and learn better: If the reward of student in time t is less than the reward of teacher in time t-delta (delta is a hyperparameter), then reset.

#### Task: Grasp_cube
Use franka to grasp a 5x5x5(cm) cube to achieve 0.2m height.

Important changes:

    1. Using SDF mode for collision detection to avoid severe penetration.
    
    2. De-ambiguity. If the cube rotate 90 degree, the state-based policy will get confused.

#### Task: open_drawer

Important changes:

    1. Using SDF mode for collision detection to avoid severe penetration. Tune PhysX parameters. [Isaac Gym Tutorial](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/factory.md#collisions-and-contacts)
    
    2. random reset object. (TODO?: random object scale)
    
    3. robot base can move.
    

