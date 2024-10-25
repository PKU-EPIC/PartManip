<h2 align="center">
  <b>PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations</b>

  <b><i>CVPR 2023</i></b>


<div align="center">
    <a href="https://cvpr.thecvf.com/virtual/2023/poster/22553" target="_blank">
    <img src="https://img.shields.io/badge/CVPR 2023-Conference Paper-red" alt="PyTorch"/></a>
    <a href="https://arxiv.org/abs/2303.16958" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://pku-epic.github.io/PartManip/" target="_blank">
    <img src="https://img.shields.io/badge/Page-PartManip-blue" alt="Project Page"/></a>
</div>

</h2>

This is the official repository of [**PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations**](https://arxiv.org/abs/2303.16958).

For more information, please visit our [**project page**](https://pku-epic.github.io/PartManip/).

## Overview
![overview](imgs/teaser.jpg)
We introduce a large-scale cross-category part manipulation benchmark PartManip with diverse object datasets, realistic settings, and rich annotations. We propose a generalizable vision-based policy learning strategy and boost the performance of part-based object manipulation by a large margin, which can generalize to unseen object categories and novel objects in the real world.

![pipeline](imgs/pipeline.jpg)
We first train state-based expert policy using our proposed canonicalization to the part coordinate frame and the part-aware reward. We then use the learned expert to collect demonstrations for pre-training the vision-based policy by behavior cloning. After pre-training, we train the vision-based policy to imitate the state-based expert policy using DAgger. We also introduce several point cloud augmentation techniques to boost the generalization ability. For the vision backbone, we introduce 3D Sparse-UNet which has a large expression capability. Furthermore, we introduced an extra domain adversarial learning module for better cross-category generalization.

## Dataset & Assets
To obtain our assets, please fill out [this form](https://forms.gle/DqdPvLE6pNWZf2XR8) and check the Terms&Conditions in it. Please cite our paper if you use our dataset.

## How to use our code

### Installation

1. install issacgym following the [official guide](https://developer.nvidia.com/isaac-gym)

2. install dependencies
```
pip install wandb tensorboard ipdb gym tqdm rich opencv_python pytorch3d pyparsing pytorch_lightning addict yapf h5py sorcery pynvml torchdata==0.5.1 einops
```

3. (Optional for full implementation, not necessary) install GAPartNet environment following the [official guide](https://github.com/geng-haoran/GAPartNet_env)

### Training and Testing

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
    
### Training Curve
There are some examples of our tested training curves:

A training example of drawer:

![Training Curve for Drawer](imgs/drawer.jpg)

A training example of door:

![Training Curve for Door](imgs/door.jpg)

## License

 This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

 [![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

 [cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
 [cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

## Citation
If you find our work useful in your research, please consider citing:

```
@article{geng2023partmanip,
    title={PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations},
    author={Geng, Haoran and Li, Ziming and Geng, Yiran and Chen, Jiayi and Dong, Hao and Wang, He},
    journal={arXiv preprint arXiv:2303.16958},
    year={2023}
}
```

## Contact
If you have any questions, please open a github issue or contact us:

Haoran Geng: ghr@stu.pku.edu.cn

He Wang: hewang@pku.edu.cn




