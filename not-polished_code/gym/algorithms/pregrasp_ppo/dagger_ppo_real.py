import os
import time
from webbrowser import get
import random
import copy
from data_structure.observation import Observations
import ipdb
from gym.spaces import Space, Box
import math
import numpy as np
import statistics
from collections import deque
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from algorithms.ppo_utils.loss import focal_loss, pixel_accuracy
# from algorithms.imitation_learning import RolloutStorage, RolloutStoragePC, RolloutStoragePC_feature, RolloutStorage_expert
from .module import Normalization, RewardScaling, Discriminator
from envs.utils.misc import _draw_line
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
     quat_mul, tensor_clamp
import torch    
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert, quaternion_multiply, quaternion_apply, euler_angles_to_matrix
from isaacgym.gymtorch import wrap_tensor
from envs.utils.get_running_bbox import get_bbox_isaac_tensor, _draw_bbox_tensor, get_bbox_pt, _draw_bbox_pt
from envs.utils.get_reward import quat_axis
from .ppo import PREGRASP_PPO
from .storage import RolloutStorage, RolloutStorage_expert, RolloutStoragePC_feature, RolloutStoragePC

class DAGGER_PPO(PREGRASP_PPO):
    def __init__(self,
                 vec_env,
                 actor_critic_class,
                 actor_critic_expert_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 max_lr=1e-3,
                 min_lr=1e-7,
                 model_cfg=None,
                 expert_cfg=None,
                 discriminator_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 eval_round=1,
                 eval_freq = 50,
                 print_log=True,
                 apply_reset=False,

                 max_iterations = 50000,
                 wandb_writer = None,
                 checkpoint_path = ' ',

                 canonical = False,
                 canon_space = 'part',
                 use_residual_traj = False,


                 use_adv_norm = True,          ### trick 1
                 adv_norm_epsilon= 1e-8,
                 use_state_norm= False,         ### trick 2
                 use_reward_norm= False,        ### trick 3
                 use_reward_scaling= False,     ### trick 4
                 learning_rate_decay = False,   ### trick 6
                 use_grad_clip = True,          ###trick 7
                 use_orthogonal_init= True,     ### trick 8
                 adam_epsilon = 1e-8,           ### trick 9
                 ):
        
        super().__init__(
            vec_env = vec_env,
            actor_critic_class = actor_critic_class,
            num_transitions_per_env = num_transitions_per_env,
            num_learning_epochs = num_learning_epochs,
            num_mini_batches = num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            init_noise_std=init_noise_std,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            max_lr=max_lr,
            min_lr=min_lr,
            model_cfg=model_cfg,
            device=device,
            sampler=sampler,
            log_dir=log_dir,
            is_testing=is_testing,
            eval_round=eval_round,
            eval_freq = eval_freq,
            print_log=print_log,
            apply_reset=apply_reset,
            max_iterations = max_iterations,
            wandb_writer =wandb_writer,
            checkpoint_path = checkpoint_path,
            canonical = canonical,
            canon_space = canon_space,
            use_residual_traj = use_residual_traj,
            use_adv_norm = use_adv_norm,          ### trick 1
            adv_norm_epsilon= adv_norm_epsilon,
            use_state_norm= use_state_norm,         ### trick 2
            use_reward_norm= use_reward_norm,        ### trick 3
            use_reward_scaling= use_reward_scaling,     ### trick 4
            learning_rate_decay = learning_rate_decay,   ### trick 6
            use_grad_clip = use_grad_clip,          ###trick 7
            use_orthogonal_init= use_orthogonal_init,     ### trick 8
            adam_epsilon = adam_epsilon,           ### trick 9
        )
        print("finish ppo initialization~")
        torch.autograd.set_detect_anomaly(True)

        self.backbone_freeze = model_cfg["freeze"]
        self.pc_feature_shape = model_cfg["feature_dim"]
        self.debug = model_cfg["GAIL_debug"]

        self.actor_critic_expert = actor_critic_expert_class(self.state_space.shape, self.action_space.shape,
                                               init_noise_std, expert_cfg, device = self.device)
       
        self.discriminator = Discriminator(discriminator_cfg, self.state_space.shape,  self.action_space.shape, feature_dim=self.pc_feature_shape ,device = self.device)
        
        print(self.actor_critic)
        print(self.actor_critic_expert)
        print(self.discriminator)

        self.actor_critic_expert.to(self.device)
        self.discriminator.to(self.device)
        self.use_expert = model_cfg["use_expert"] # use expert policy network or use demostration
        self.model_cfg = model_cfg
        self.expert_cfg = expert_cfg
        self.discriminator_cfg = discriminator_cfg
        self.rew_clip_magnitude =  discriminator_cfg["rew_clip_magnitude"]
        self.mean_rew_policy = 0
        self.mean_rew_expert = 0
        self.use_self_imitation = model_cfg["use_self_imitation"]
        self.expert_buffer_size = expert_cfg["expert_buffer_size"]
        self.use_seg = self.model_cfg["use_seg"]
        self.use_domain_discriminator = self.model_cfg["use_domain_discriminator"]
        self.train_asset_num = self.train_env_num // self.vec_env.task.env_per_asset

        self.qpose_num = (self.vec_env.task.franka_num_dofs)*2 + 2 # robot qpose qvel + object qpos qvel

        if not self.is_testing:
            if not self.use_pc:
                self.storage = RolloutStorage(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                                        self.state_space.shape, self.action_space.shape, self.device, sampler)
            else:
                # if self.backbone_freeze:
                #     self.storage = RolloutStoragePC_feature(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                #                             self.state_space.shape, self.action_space.shape, self.pc_feature_shape, self.device, sampler)
                # else:
                #     self.storage = RolloutStoragePC(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                #                             self.state_space.shape, self.action_space.shape, self.pc_space.shape, self.device, sampler)
                self.storage = RolloutStoragePC(self.train_env_num, num_transitions_per_env, self.observation_space.shape,
                                            self.state_space.shape, self.action_space.shape, self.pc_space.shape, self.device, sampler)
            
            if self.use_expert:
                self.expert_storage = RolloutStorage_expert(self.train_env_num, self.expert_buffer_size, self.observation_space.shape, self.pc_space.shape,
                                        self.state_space.shape, self.action_space.shape, self.device, sampler, use_seg = self.model_cfg["use_seg"])
            if self.model_cfg["use_bc_pretrain"]:
                self.optimizer_bc = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
                self.loss_function_bc = torch.nn.MSELoss()
            #     self.demo_num = model_cfg["demo_num"]
            #     self.demo_obs = torch.zeros((self.demo_num, self.train_env_num, self.num_transitions_per_env, *self.observation_space.shape), device = self.device)
            #     if self.backbone_freeze:
            #         self.demo_pcs = torch.zeros((self.demo_num, self.train_env_num, self.num_transitions_per_env, self.pc_feature_shape), device = self.device)
            #     else:
            #         self.demo_pcs = torch.zeros((self.demo_num, self.train_env_num, self.num_transitions_per_env, *self.pc_space.shape), device = self.device)
            
            #     self.demo_actions = torch.zeros((self.demo_num, self.train_env_num, self.num_transitions_per_env, *self.action_space.shape), device = self.device)
            #     self.demo_step =  torch.zeros((self.train_env_num), device=self.device, dtype=torch.float)
            #     self.demo_part_center = torch.zeros((self.demo_num, self.train_env_num, 3), device = self.device)
            #     ############TODO############
                #setup deque for each object trajectory list()


            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps= adam_epsilon)   # , weight_decay=float(self.weight_decay), trick 9
            self.optimizer_dagger = optim.Adam(self.actor_critic.parameters(), lr=model_cfg["learning_rate_dagger"])   # , weight_decay=float(self.weight_decay), trick 9
            self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_cfg["lr"])
            self.bce = nn.BCEWithLogitsLoss().to(self.device)
            self.l2 = nn.MSELoss().to(self.device)
            self.disc_gt = torch.cat(
            [
                torch.zeros((self.discriminator_cfg["update_freq"]  * self.train_env_num, 1), device=self.device),
                torch.ones((self.discriminator_cfg["update_freq"]  * self.train_env_num, 1), device=self.device),
            ],
            dim=0,
        )

    def load(self, path):
        assert os.path.exists(self.checkpoint_path)
        checkpoint_dict= torch.load(self.checkpoint_path, map_location=self.device)
        # self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
        try:
            self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
        except:
            self.actor_critic.backbone.load_state_dict(checkpoint_dict["model_state_dict"])

    def load_expert(self, path):
        assert os.path.exists(path)
        if self.use_expert:
            checkpoint_dict= torch.load(path, map_location=self.device)
            self.actor_critic_expert.load_state_dict(checkpoint_dict["model_state_dict"])
        if self.model_cfg["use_bc_pretrain"]:
            self.demo_num = self.model_cfg["demo_num"]
            self.demo_states = torch.zeros((self.train_asset_num, self.demo_num, self.max_episode_length, *self.state_space.shape), device = self.device)
            self.demo_pcs = torch.zeros((self.train_asset_num, self.demo_num, self.max_episode_length, *self.pc_space.shape), device = self.device)
            self.demo_actions = torch.zeros((self.train_asset_num, self.demo_num, self.max_episode_length, *self.action_space.shape), device = self.device)
            print("loading demo")
            # path = os.path.join(path, '{}'.format(self.vec_env.task.cfg["task"]["target_part_category"]))
            # i = 0
            # for _ in range(self.train_env_num // self.vec_env.task.env_per_asset):
            #     path_i_new = path + '/{}.npy'.format(str(self.vec_env.task.selected_asset_path_list[i]).split("/")[-1])
            #     if os.path.exists(path_i_new):
            #         print("load demo_{} from ".format(i), path_i_new)
            #         dict_load = np.load(path_i_new, allow_pickle=True).item()
            #         dict = {}
            #         for k in dict_load.keys():
            #             if k != "traj_num":
            #                 dict[k] = torch.from_numpy(dict_load[k]).to(self.device)
            #             else:
            #                 dict[k] = dict_load[k]
            #         assert(self.demo_num <= dict["traj_num"])
            #         self.demo_states[i, :] = dict["observations"][:self.demo_num]
            #         self.demo_actions[i, :] = dict["actions"][:self.demo_num]
            #         self.demo_pcs[i, :] = dict["pcs"][:self.demo_num, :, :, :7]
            #         i += 1
            #     else:
            #         print(f"{path_i_new} is not found !")
            # print("total_demo_num", i * self.demo_num)

            # self.demo_states = self.demo_states[:i]
            # self.demo_pcs = self.demo_pcs[:i]
            # self.demo_actions = self.demo_actions[:i]
            # import pdb
            # pdb.set_trace()
            self.demo_part_center = self.demo_states[:, :, 0, -3:].unsqueeze(2).repeat([1, 1, self.max_episode_length, 1])

    def pretrain_bc(self):
        ap = self.actor_critic.actor_mlp.parameters()
        cp = self.actor_critic.critic_mlp.parameters()
        self.actor_critic.backbone.unfreeze()
        demo_states = self.demo_states.reshape(-1, *self.state_space.shape)
        demo_pcs = self.demo_pcs.reshape(-1, *self.pc_space.shape)
        demo_actions = self.demo_actions.reshape(-1, *self.action_space.shape)
        demo_part_center = self.demo_part_center.reshape(-1, 3)
        num_samples = demo_states.shape[0]

        for ep in tqdm(range(self.model_cfg["bc_epochs"])):
            l = int(num_samples / self.model_cfg["mb_size"])
            for mb in range(l):
                rand_idx = torch.from_numpy(np.random.choice(num_samples, size=self.model_cfg["mb_size"]))
                obs = Observations(obs=demo_states[rand_idx], state=demo_states[rand_idx], points=demo_pcs[rand_idx])
                obs_new = torch.cat((obs.obs[:, :22], obs.obs[:, 24:40], demo_part_center[rand_idx]), dim=1)
                obs.obs = obs_new
                act = demo_actions[rand_idx]
                act_pred, _, _ = self.actor_critic.act_dagger(obs)
                self.optimizer_bc.zero_grad()
                loss = self.loss_function_bc(act, act_pred)
                loss.backward()
                self.optimizer_bc.step()
                ###Log###
                print("loss:", loss)
                self.writer.add_scalar("BC/" + "loss", loss, ep * l + mb)
                if self.wandb_writer is not None:
                    self.wandb_writer.log({
                        "BC/" + "loss", loss
                    })
                ###Log end###

    def discriminator_update(self, it, update=True):
        ####expert_batch####
        states_batch = self.storage.states.reshape(-1, self.storage.states.size(-1))
        actions_batch = self.storage.actions.reshape(-1, self.storage.actions.size(-1))
        pcs_batch = None
        expert_pcs_batch = None
        if not self.debug:
            # if self.backbone_freeze:
            #     pcs_batch = self.storage.pointcloud_features.reshape(-1, self.storage.pointcloud_features.size(-1))
            # else:
            #     #pcs_batch = self.storage.pointcloud_features.reshape(-1, self.storage.pointcloud.size(-1))
            #     pcs_batch = self.storage.pointcloud.reshape(-1, self.storage.pointcloud.size(-2), self.storage.pointcloud.size(-1))
            pcs_batch = self.storage.pointcloud.reshape(-1, self.storage.pointcloud.size(-2), self.storage.pointcloud.size(-1))

        rand_num = random.randint(0, self.demo_num - 1)
        if self.discriminator_cfg["use_random_transitions"]:
            rand_num_ = torch.randint(0, self.max_episode_length, (self.num_transitions_per_env,), device=self.device)
            expert_states_batch = self.demo_states[rand_num][:, rand_num_].reshape(-1, self.storage.states.size(-1))
            expert_actions_batch = self.demo_actions[rand_num][:, rand_num_].reshape(-1, self.storage.actions.size(-1))
            if not self.debug:
                # if self.backbone_freeze:
                #     expert_pcs_batch = self.demo_pcs[rand_num][:, rand_num_].reshape(-1, self.demo_pcs[rand_num].size(-1))
                # else:
                #     expert_pcs_batch = self.demo_pcs[rand_num][:, rand_num_].reshape(-1, self.demo_pcs[rand_num].size(-2), self.demo_pcs[rand_num].size(-1))
                expert_pcs_batch = self.demo_pcs[rand_num][:, rand_num_].reshape(-1, self.demo_pcs[rand_num].size(-2), self.demo_pcs[rand_num].size(-1))
        else:
            expert_states_batch = self.demo_states[rand_num][:, self.episode_step_for_now+1-self.num_transitions_per_env:self.episode_step_for_now+1].reshape(-1, self.storage.states.size(-1))
            expert_actions_batch = self.demo_actions[rand_num][:, self.episode_step_for_now+1-self.num_transitions_per_env:self.episode_step_for_now+1].reshape(-1, self.storage.actions.size(-1))

            if not self.debug:
                # if self.backbone_freeze:
                #     expert_pcs_batch = self.demo_pcs[rand_num][:, self.episode_step_for_now+1-self.num_transitions_per_env:self.episode_step_for_now+1].reshape(-1, self.demo_pcs[rand_num].size(-1))
                # else:
                #     expert_pcs_batch = self.demo_pcs[rand_num][:, self.episode_step_for_now+1-self.num_transitions_per_env:self.episode_step_for_now+1].reshape(-1, self.demo_pcs[rand_num].size(-2), self.demo_pcs[rand_num].size(-1))
                expert_pcs_batch = self.demo_pcs[rand_num][:, self.episode_step_for_now+1-self.num_transitions_per_env:self.episode_step_for_now+1].reshape(-1, self.demo_pcs[rand_num].size(-2), self.demo_pcs[rand_num].size(-1))

        ############not freeze##############
        if not self.backbone_freeze and not self.debug:
            expert_pcs_batch_new = torch.zeros((self.train_asset_num*self.num_transitions_per_env, self.pc_feature_shape), device = self.device)
            pcs_batch_new = torch.zeros((self.train_env_num*self.num_transitions_per_env, self.pc_feature_shape), device = self.device)
            with torch.no_grad():
                expert_pcs_batch_new = self.actor_critic.get_features(expert_pcs_batch)
                pcs_batch_new = self.actor_critic.get_features(pcs_batch)
            # for i in range(self.train_asset_num):
            #     with torch.no_grad():
            #         expert_pcs_batch_new[i*self.num_transitions_per_env:(i+1)*self.num_transitions_per_env] = self.actor_critic.get_features(expert_pcs_batch[i*self.num_transitions_per_env:(i+1)*self.num_transitions_per_env] )####变成feature
            
            # for i in range(self.train_env_num):
            #     with torch.no_grad():
            #         pcs_batch_new[i*self.num_transitions_per_env:(i+1)*self.num_transitions_per_env]  = self.actor_critic.get_features(pcs_batch[i*self.num_transitions_per_env:(i+1)*self.num_transitions_per_env] )
        else:
            expert_pcs_batch_new = expert_pcs_batch
            pcs_batch_new = pcs_batch

        disc_logits_policy = self.discriminator(states_batch, actions_batch, pcs_batch_new)
        disc_logits_expert = self.discriminator(expert_states_batch, expert_actions_batch, expert_pcs_batch_new)
        disc_logits_expert = disc_logits_expert.repeat(self.vec_env.task.env_per_asset, 1)
        disc_logits = torch.cat((disc_logits_policy, disc_logits_expert), dim = 0)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())

        if self.discriminator_cfg["use_grad_pen"]:
            
            expert_input = expert_actions_batch
            policy_input = actions_batch

            if self.discriminator_cfg["use_feature"]:
                expert_input = torch.cat((expert_pcs_batch_new, expert_input), dim = 1)
                policy_input = torch.cat((pcs_batch_new, policy_input), dim = 1)
            if self.discriminator_cfg["use_gt_obs"]:
                expert_input = torch.cat((expert_states_batch, expert_input), dim = 1)
                policy_input = torch.cat((states_batch, policy_input), dim = 1)


            expert_input = expert_input.repeat(self.vec_env.task.env_per_asset, 1)

            eps = torch.rand(expert_input.size(0), 1, device=self.device)
            # print(eps.shape)
            # print(expert_input.shape)
            interp_states = eps * expert_input + (1 - eps) * policy_input
            interp_states = interp_states.detach()
            interp_states.requires_grad_(True)

            gradients = torch.autograd.grad(
                outputs=self.discriminator.disc(interp_states).sum(),
                inputs=[interp_states],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.discriminator_cfg["grad_pen_weight"]
        else:
            disc_grad_pen_loss = 0

        disc_ce_loss = self.bce(disc_logits, self.disc_gt) + disc_grad_pen_loss

        accuracy = (disc_preds == self.disc_gt).type(torch.FloatTensor).mean()
       
        self.mean_rew_policy = F.softplus(disc_logits_policy, beta=1).detach().mean()
        self.mean_rew_expert = F.softplus(disc_logits_expert, beta=1).detach().mean()
        
        #print(length)
        if self.wandb_writer is not None:
            self.wandb_writer.log({
                "discriminator/loss": disc_ce_loss, 
                "discriminator/accuracy": accuracy, 
                "discriminator/policy_rews": self.mean_rew_policy, 
                "discriminator/expert_rews": self.mean_rew_expert, 
            })
        
        self.writer.add_scalar('discriminator/loss', disc_ce_loss, it)
        self.writer.add_scalar('discriminator/accuracy', accuracy, it)
        self.writer.add_scalar('discriminator/policy_rews', self.mean_rew_policy, it)
        self.writer.add_scalar('discriminator/expert_rews', self.mean_rew_expert, it)        


        #########update discriminator#############
        if it % self.discriminator_cfg["update_freq"] == 0 and update:
            self.disc_optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            disc_ce_loss.backward() ####防止gradient graph中断
            self.disc_optimizer.step()

        return disc_ce_loss.detach()

    def save(self, path, it):    
        torch.save({
                    'iteration': it + 1,
                    'model_state_dict': self.actor_critic.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'optimizer_dagger_state_dict': self.optimizer_dagger.state_dict(),
                    'total_steps' : self.tot_timesteps,
                    'discriminator_dict': self.discriminator.state_dict(),
                    "discriminator_optimizer": self.disc_optimizer.state_dict(),
                    }, path)

    def change_obs(self, part_bbox_tensor, handle_bbox_tensor, current_obs):
        # By default, the expert use edge bbox and need to change observations
        
        # if self.vec_env.task.cfg["obs"]["state"]["use_bbox_type"] != "None" and self.vec_env.task.cfg["obs"]["state"]["use_bbox_form"] == "vertex":
        #     # if not self.use_pc:
        #     current_obs.state[:, self.qpose_num+16:self.qpose_num+40]\
        #         .copy_(handle_bbox_tensor.view(-1, 24))
        #     current_obs.state[:, self.qpose_num+40:self.qpose_num+16+64]\
        #         .copy_(part_bbox_tensor.view(-1, 24))
            # else:
            #     current_obs["state"][:, self.qpose_num+16:self.qpose_num+40]\
            #         .copy_(handle_bbox_tensor.reshape(-1, 24))
            #     current_obs["state"][:, self.qpose_num+40:self.qpose_num+64]\
            #         .copy_(part_bbox_tensor.reshape(-1, 24))

        # elif self.vec_env.task.cfg["obs"]["state"]["use_bbox_type"] != "None" and self.vec_env.task.cfg["obs"]["state"]["use_bbox_form"] == "edge":

        handle_out_canon = handle_bbox_tensor[:, 0] -  handle_bbox_tensor[:, 4]
        handle_long_canon =  handle_bbox_tensor[:, 1] -  handle_bbox_tensor[:, 0]
        handle_short_canon =  handle_bbox_tensor[:, 3] -  handle_bbox_tensor[:, 0]
        handle_center_canon = (handle_bbox_tensor[:, 0] + handle_bbox_tensor[:, 6]) / 2
        part_out_canon = part_bbox_tensor[:, 0] - part_bbox_tensor[:, 4]
        part_long_canon =  part_bbox_tensor[:, 1] - part_bbox_tensor[:, 0]
        part_short_canon =  part_bbox_tensor[:, 3] - part_bbox_tensor[:, 0]
        part_center_canon = (part_bbox_tensor[:, 0] + part_bbox_tensor[:, 6]) / 2
        #34-46: part bounding box
        # if not self.use_pc:
        
        current_obs.obs[:, self.qpose_num+16:self.qpose_num+19]\
            .copy_(handle_out_canon)
        current_obs.obs[:, self.qpose_num+19:self.qpose_num+22]\
            .copy_(handle_long_canon)
        current_obs.obs[:, self.qpose_num+22:self.qpose_num+25]\
            .copy_(handle_short_canon)
        current_obs.obs[:, self.qpose_num+25:self.qpose_num+28]\
            .copy_(handle_center_canon)
        #46-57: handle bounding box
        current_obs.obs[:, self.qpose_num+28:self.qpose_num+31]\
            .copy_(part_out_canon)
        current_obs.obs[:, self.qpose_num+31:self.qpose_num+34]\
            .copy_(part_long_canon)
        current_obs.obs[:, self.qpose_num+34:self.qpose_num+37]\
            .copy_(part_short_canon)
        current_obs.obs[:, self.qpose_num+37:self.qpose_num+40]\
            .copy_(part_center_canon)

            # else:
            #     current_obs["state"][:, self.qpose_num+16:self.qpose_num+19]\
            #         .copy_(handle_out_canon)
            #     current_obs["state"][:, self.qpose_num+19:self.qpose_num+22]\
            #         .copy_(handle_long_canon)
            #     current_obs["state"][:, self.qpose_num+22:self.qpose_num+25]\
            #         .copy_(handle_short_canon)
            #     current_obs["state"][:, self.qpose_num+25:self.qpose_num+28]\
            #         .copy_(handle_center_canon)
            #     #46-57: handle bounding box
            #     current_obs["state"][:, self.qpose_num+28:self.qpose_num+31]\
            #         .copy_(part_out_canon)
            #     current_obs["state"][:, self.qpose_num+31:self.qpose_num+34]\
            #         .copy_(part_long_canon)
            #     current_obs["state"][:, self.qpose_num+34:self.qpose_num+37]\
            #         .copy_(part_short_canon)
            #     current_obs["state"][:, self.qpose_num+37:self.qpose_num+40]\
            #         .copy_(part_center_canon)
        # else:
        #     raise NotImplementedError

        return current_obs

    def canonicalize(self, current_obs_old):
        # if not self.use_pc:
        #     current_obs = current_obs_old.clone()
        # else:

        if self.use_pc:
            current_obs = Observations(state = current_obs_old.state.clone(), obs=current_obs_old.obs.clone(), points = current_obs_old.points.clone())
            # return current_obs
        else:
            current_obs = Observations(state = current_obs_old.state.clone(), obs=current_obs_old.obs.clone())
            

        if self.use_first_bbox is not None:
            part_bbox_tensor = self.part_bbox_tensor
            handle_bbox_tensor = self.handle_bbox_tensor
        else:
            if self.vec_env.task.target_part == "door":
                part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
            elif self.vec_env.task.target_part == "drawer":
                part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 1)
        
        
        #part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.vec_env.task, self.vec_env.task.cabinet_dof_tensor[:,0], 0)
        # _draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
        # _draw_line(self.vec_env.task, handle_bbox_tensor[0, 0].cpu(), (handle_bbox_tensor[0, 0]).cpu() + torch.tensor([1, 0, 0]), clear=False)
        # _draw_line(self.vec_env.task, handle_bbox_tensor[0, 0].cpu(), (handle_bbox_tensor[0, 0]).cpu() + torch.tensor([0, 1, 0]), clear=False)
        # _draw_line(self.vec_env.task, handle_bbox_tensor[0, 0].cpu(), (handle_bbox_tensor[0, 0]).cpu() + torch.tensor([0, 0, 1]), clear=False)
    
        ########hand_pose
        # if not self.use_pc:
        #     hand_pose = current_obs[:,self.qpose_num:self.qpose_num+7]   #N * 7
        # else:
        hand_pose = current_obs.obs[:,self.qpose_num+3:self.qpose_num+10]   #N * 7
        hand_pose[:, :3] += self.vec_env.task.franka_root_tensor[:, :3]  
        hand_pose[:, :3] -= self.center
        
        hand_pose[:, :3] = quaternion_apply(self.quaternion_rot, hand_pose[:, :3])      
        
        hand_pose[:, 3:7] = quaternion_multiply(self.quaternion_rot, hand_pose[:, 3:7])
        
        # root_tensor = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), root_tensor - self.center)
        #######bbox
        handle_bbox_tensor_canon = handle_bbox_tensor - self.center.view(-1, 1, 3)
        part_bbox_tensor_canon = part_bbox_tensor - self.center.view(-1, 1, 3)
        handle_bbox_tensor_canon = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), handle_bbox_tensor_canon) #@self.rot_matrix
        part_bbox_tensor_canon = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), part_bbox_tensor_canon) #@self.rot_matrix
        # hand_grip_dir = quat_axis(hand_pose[:, 3:7], 2)
        # hand_sep_dir = quat_axis(hand_pose[:, 3:7], 1)
        # hand_down_dir = quat_axis(hand_pose[:, 3:7], 0)
        #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor)
        #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor)
        #_draw_bbox_tensor(self.vec_env.task, part_bbox_tensor_canon)
        #_draw_bbox_tensor(self.vec_env.task, handle_bbox_tensor_canon)
        #_draw_line(self.vec_env.task, hand_pose[0, :3].cpu(), torch.tensor([0, 0, 0]), clear=False)
        #_draw_line(self.vec_env.task, hand_pose[0, :3].cpu(), (hand_grip_dir[0] + hand_pose[0, :3]).cpu(), clear=False)
        #time.sleep(1)
        # if self.use_pc:
        #     current_obs.points[:, :, :3] = quaternion_apply(self.quaternion_rot.view(-1, 1, 4), current_obs.points[:,:, :3]-self.center.view(-1,1,3))
        
        # else:
        current_obs.obs[:,self.qpose_num+3:self.qpose_num+10] = hand_pose
        current_obs.obs[:,self.qpose_num:self.qpose_num+3] = quaternion_apply(self.quaternion_rot.view(-1, 4), self.vec_env.task.franka_root_tensor[:, :3] - self.center)

        return self.change_obs(part_bbox_tensor_canon, handle_bbox_tensor_canon, current_obs)

    def run(self, num_learning_iterations, log_interval=1):

        if self.canonical or self.expert_cfg["canonicalize"]:
            self.canonicalize_init()
        self.pre_grasp()

        current_states = self.vec_env.get_state()   # size (16, 0) ?

        if self.is_testing:

            self.vec_env.task.eval()
            _ = self.eval(self.current_learning_iteration)
            _ = self.eval_expert(self.current_learning_iteration)

        else:

            self.vec_env.task.train()

            if self.model_cfg["use_bc_pretrain"]:
                self.pretrain_bc()

            rewbuffer = deque(maxlen=200)
            gt_rewbuffer = deque(maxlen=200)
            # disc_loss_all = deque(maxlen=10)
            lenbuffer = deque(maxlen=200)

            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_gt_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            
            success_rate = []
            reward_sum = []
            gt_reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                current_obs = self.vec_env.reset()
                start = time.time()
                ep_infos = []
                task_info = {}
                if it % self.eval_freq == 0:
                    _ = self.eval_expert(it)
                    train_reward, valIntra_reward, valInter_reward,  train_success,  valIntra_success, valInter_success = self.eval(it)
                    
                    if self.wandb_writer is not None:
                        self.wandb_writer.log({
                            "Val/train_reward": train_reward,
                            "Val/valIntra_reward": valIntra_reward,
                            "Val/valInter_reward": valInter_reward,
                            "Val/train_success": train_success,
                            "Val/valIntra_success": valIntra_success,
                            "Val/valInter_success": valInter_success,
                        })
                    self.save(os.path.join(self.log_dir, 'model_{}.tar'.format(it)), it)
                    
                total_success = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num), device=self.device)
                for i in range(self.max_episode_length): 

                    if self.use_state_norm : ### trick 2
                        current_obs = self.state_norm(current_obs)
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()

                    if self.use_residual_traj:
                        current_obs = self.residual_actions(current_obs, i)

                    if not self.use_pc:
                        current_train_obs = Observations(state=current_obs.state[:self.train_env_num], obs=current_obs.obs[:self.train_env_num])
                        if self.valIntra_env_num:
                            current_valIntra_obs = Observations(state=current_obs.state[self.train_env_num:self.valIntra_env_num + self.train_env_num], obs=current_obs.obs[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                        if self.valIntra_env_num:
                            current_valInter_obs = Observations(state=current_obs.state[self.valIntra_env_num + self.train_env_num:], obs=current_obs.obs[self.valIntra_env_num + self.train_env_num:])

                    else:
                        current_train_obs = Observations(state=current_obs.state[:self.train_env_num], points= current_obs.points[:self.train_env_num], obs=current_obs.obs[:self.train_env_num])
                        if self.valIntra_env_num:
                            current_valIntra_obs = Observations(state=current_obs.state[self.train_env_num:self.valIntra_env_num + self.train_env_num],obs=current_obs.obs[self.train_env_num:self.valIntra_env_num + self.train_env_num], points= current_obs.points[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                        if self.valIntra_env_num:
                            current_valInter_obs = Observations(state=current_obs.state[self.valIntra_env_num + self.train_env_num:],obs=current_obs.obs[self.valIntra_env_num + self.train_env_num:], points=current_obs.points[self.valIntra_env_num + self.train_env_num:])

                        #####use noqpose and no boundingbox observations for actor
                        # Compute the action
                    train_actions, train_actions_log_prob, train_values, train_mu, train_sigma, others = self.actor_critic.act(current_train_obs, require_grad = False, concat_part_center = self.model_cfg["concat_part_center"])
                    actions = train_actions
                    # train_actor_actions, _ , _ = self.actor_critic.act_dagger(current_train_obs)
                    
                    if self.valIntra_env_num:
                        valIntra_actions, others = self.actor_critic.act_inference(current_valIntra_obs, concat_part_center = self.model_cfg["concat_part_center"])
                        actions = torch.cat((actions, valIntra_actions))
                        # if self.use_pc:
                        #     valIntra_features = others["feature"]
                        #     if valIntra_features is not None:
                        #         features = torch.cat((features, valIntra_features)) 
                    if self.valInter_env_num:
                        valInter_actions, others = self.actor_critic.act_inference(current_valInter_obs, concat_part_center = self.model_cfg["concat_part_center"])
                        actions = torch.cat((actions, valInter_actions))
                        # if self.use_pc:
                        #     valInter_features = others["feature"]
                        #     if valInter_features is not None:
                        #         features = torch.cat((features, valInter_features)) 
                        #####compute expert action@######
                    if self.use_expert:
                        current_obs_expert = current_obs
                        current_obs_expert.obs = current_obs.state
                        if self.canonical:
                            current_obs_expert = current_obs_expert
                        elif self.expert_cfg["canonicalize"]:
                            current_obs_expert = self.canonicalize(current_obs_expert)
                        else:
                            current_obs_expert = current_obs
                        # if self.canonical and not self.use_pc:
                        #     current_obs_expert = current_obs
                        # elif self.expert_cfg["canonicalize"]:
                        #     current_obs_expert = self.canonicalize(current_obs)
                        # else:
                        #     current_obs_expert = current_obs
                        expert_actions, _ = self.actor_critic_expert.act_inference(current_obs_expert, concat_part_center = self.model_cfg["concat_part_center"])
                        if self.expert_cfg["canonicalize"]:
                            expert_actions = self.uncanonicalize(expert_actions)
                        #if self.expert_cfg["canonicalize"]:
                        #    current_obs_expert = self.canonicalize(current_obs)
                        #else:
                        #    current_obs_expert = current_obs
                        #current_obs_expert.obs = current_obs_expert.state
                        #expert_actions, _ = self.actor_critic_expert.act_inference(current_obs_expert)
                        #
                        #if self.expert_cfg["canonicalize"]:
                        #    expert_actions = self.uncanonicalize(expert_actions)

                    next_obs, gt_rews, dones, infos = self.vec_env.step(actions)
                    total_success = torch.logical_or(infos["successes"].to(self.device), total_success)

                    train_rews = gt_rews[:self.train_env_num]



                    if self.use_expert:
                        if not self.use_seg:
                            self.expert_storage.add_transitions(
                                current_train_obs.points[...,:self.numPCObservations],
                                current_train_obs.obs,
                                train_actions,
                                expert_actions[:self.train_env_num],
                            )
                        else:
                            self.expert_storage.add_transitions(
                                current_train_obs.points[...,:self.numPCObservations],
                                current_train_obs.obs,
                                train_actions,
                                expert_actions[:self.train_env_num],
                                seg_label = current_train_obs.points[..., -1],
                            )
                    #####ppo trick 3, 4#########
                    if self.use_reward_norm:    
                        rews = self.reward_norm(rews)
                    elif self.use_reward_scaling:
                        rews = self.reward_scaling(rews)

                    next_states = self.vec_env.get_state()
                    train_dones = dones[:self.train_env_num]
                    # Record the transition
                    if not self.use_pc:
                        self.storage.add_transitions(
                            observations = current_train_obs.obs,
                            states = current_train_obs.state,
                            actions = train_actions,
                            rewards = train_rews, 
                            dones = train_dones,
                            values = train_values, 
                            actions_log_prob = train_actions_log_prob,
                            mu = train_mu,
                            sigma = train_sigma,
                        )
                    else:
                        self.storage.add_transitions(
                            observations = current_train_obs.obs, 
                            points = current_train_obs.points[...,:self.numPCObservations],
                            states = current_train_obs.state,
                            actions = train_actions,
                            rewards = train_rews, 
                            dones = train_dones,
                            values = train_values, 
                            actions_log_prob = train_actions_log_prob,
                            mu = train_mu,
                            sigma = train_sigma
                        )

                    current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)
                    
                    #current_states.copy_(next_states)
                    # Book keeping

                    ep_infos.append(infos)
                    # torch.cuda.synchronize()

                    if self.print_log:
                        
                        cur_reward_sum[:] += gt_rews
                        cur_gt_reward_sum[:] += gt_rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                        gt_reward_sum.extend(cur_gt_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].detach().cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_gt_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    
                    ####update_discriminator
                    # if (i+1) % self.discriminator_cfg["update_freq"] == 0:
                    #     disc_loss = self.discriminator_update()
                    #     if self.use_expert:
                    #         self.expert_storage.clear() 
                    #     disc_loss_all.extend([disc_loss.item()])

                    # Learning step
                    self.actor_critic.train() 
                    if (i + 1) % self.num_transitions_per_env == 0 and (i + 1) != self.max_episode_length:

                        _, _, last_values, _, _, _ = self.actor_critic.act(current_obs, concat_part_center = self.model_cfg["concat_part_center"])
                        
                        stop = time.time()
                        collection_time = stop - start

                        # mean_trajectory_length, mean_reward = self.storage.get_statistics()
                        mean_reward = self.storage.rewards.mean()
                        # Learning step
                        start = stop
                        self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)
                        #(self.storage.observations[:,:,14])
                        mean_value_loss, mean_surrogate_loss, mean_dagger_loss, mean_rl_loss, mean_seg_loss, dagger_loss_proportion, seg_loss_proportion, rl_loss_proportion, mean_seg_accuracy = self.update(it)
                        self.storage.clear()
                        # self.expert_storage.clear()
                        torch.cuda.synchronize()
                        stop = time.time()
                        learn_time = stop - start
                        start = stop
                        if self.print_log:
                            self.log(locals())
                    
                    if self.print_log:
                        rewbuffer.extend(reward_sum)
                        gt_rewbuffer.extend(gt_reward_sum)
                        lenbuffer.extend(episode_length)

                # current_obs_new = {"state": current_obs["state"][:, self.qpose_num:self.qpose_num+16], "pc": current_obs["pc"]}
                _, _, last_values, _, _ , _= self.actor_critic.act(current_obs, current_states, concat_part_center = self.model_cfg["concat_part_center"])
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop

                self.storage.compute_returns(last_values[:self.train_env_num], self.gamma, self.lam)

                mean_value_loss, mean_surrogate_loss, mean_dagger_loss, mean_rl_loss, mean_seg_loss, dagger_loss_proportion, seg_loss_proportion, rl_loss_proportion, mean_seg_accuracy = self.update(it)

                if self.vec_env.task.task_meta["need_update"] :
                    task_info = self.vec_env.task.update(it)

                self.storage.clear()
                # self.expert_storage.clear()
                stop = time.time()

                learn_time = stop - start

                #################self imitate, update the demostration buffer##################
                ###############borrowed from Hao Shen, Weikang Wan###################
                # if not self.use_expert and self.use_self_imitation:
                #     for i in range(self.train_env_num):
                #         if total_success[i]:
                #             self.demo_obs[self.demo_step[i], i] = self.storage.observations[:, i]
                #             self.demo_actions[self.demo_step[i], i] = self.storage.actions[:, i]
                #             if self.backbone_freeze:
                #                 self.demo_pcs[self.demo_step[i], i] = self.storage.pointcloud_features[:, i]
                #             else:
                #                 self.demo_pcs[self.demo_step[i], i] = self.storage.pointcloud[:, i]
                    
                #         self.demo_step[i] += 1
                #     self.demo_step = self.demo_step % self.demo_num

                #####################END################

            
                if self.print_log:
                    self.log(locals())
                    
                if it % log_interval == 0:
                    if self.vec_env.task.task_meta["need_update"] :
                        self.vec_env.task.save(self.log_dir, it)
                ep_infos.clear()

            if self.vec_env.task.task_meta["need_update"] :
                self.vec_env.task.save(self.log_dir, num_learning_iterations)

    def log(self, locs, width=80, pad=35):
        #mean_vel = torch.mean(torch.mean(self.storage.observations[:, :, 7:13], dim = 0), dim = 1)
        #print(mean_vel)
        #print((torch.abs(mean_vel) < 0.005).sum().item())
        self.tot_timesteps += self.max_episode_length * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if_val = self.train_env_num<self.vec_env.num_envs
        if locs['ep_infos']:

            for key in locs['ep_infos'][0]:
                
                if key == "successes" or key == "success_entropy":
                    continue
                    infotensor_train = torch.zeros((self.train_env_num), device= self.device)
                    infotensor_val = torch.zeros((self.val_env_num), device=self.device)
                    for ep_info in locs['ep_infos']:
                        infotensor_train = torch.logical_or(infotensor_train, ep_info[key].to(self.device)[:self.train_env_num])
                        infotensor_val = torch.logical_or(infotensor_val, ep_info[key].to(self.device)[self.train_env_num:])
                        #print(ep_info[key])
                elif key == "part_pos" or key == "grasp_success_rate" :
                    infotensor_train = torch.zeros((self.train_env_num), device= self.device)
                    infotensor_valIntra = torch.zeros((self.valIntra_env_num), device=self.device)
                    infotensor_valInter = torch.zeros((self.valInter_env_num), device=self.device)
                    for ep_info in locs['ep_infos']:
                        infotensor_train = torch.max(infotensor_train, ep_info[key].to(self.device)[:self.train_env_num])
                        infotensor_valIntra = torch.max(infotensor_valIntra, ep_info[key].to(self.device)[self.train_env_num:self.valIntra_env_num + self.train_env_num])
                        infotensor_valInter = torch.max(infotensor_valInter, ep_info[key].to(self.device)[self.valIntra_env_num + self.train_env_num:])
                    #print(infotensor_train , infotensor_val)
                else: 
                    infotensor_train = torch.tensor([], device=self.device)
                    infotensor_valIntra = torch.tensor([], device=self.device)
                    infotensor_valInter = torch.tensor([], device=self.device)
                    for ep_info in locs['ep_infos']:

                        infotensor_train = torch.cat((infotensor_train, ep_info[key].to(self.device)[:self.train_env_num].float()))
                        infotensor_valIntra = torch.cat((infotensor_valIntra, ep_info[key].to(self.device)[self.train_env_num:self.valIntra_env_num + self.train_env_num].float()))
                        infotensor_valInter = torch.cat((infotensor_valInter, ep_info[key].to(self.device)[self.valIntra_env_num + self.train_env_num:].float()))
                if key=="success_rate":
                    sorted_success_rate_train, _ = torch.sort(infotensor_train)
                    sorted_success_rate_valIntra , _ = torch.sort(infotensor_valIntra)
                    sorted_success_rate_valInter , _ = torch.sort(infotensor_valInter)
                    worst_rate = 0.5
                    num_worst_train = int(infotensor_train.shape[0]*worst_rate)
                    num_worst_valIntra = int(infotensor_valIntra.shape[0]*worst_rate)
                    num_worst_valInter = int(infotensor_valInter.shape[0]*worst_rate)
                    worst_success_rate_train = sorted_success_rate_train[:num_worst_train]
                    worst_success_rate_valIntra = sorted_success_rate_valIntra[:num_worst_valIntra]
                    worst_success_rate_valInter = sorted_success_rate_valInter[:num_worst_valInter]
                    worst_mean_train = worst_success_rate_train.mean()
                    worst_mean_valIntra = worst_success_rate_valIntra.mean()
                    worst_mean_valInter = worst_success_rate_valInter.mean()

                    self.writer.add_scalar(f"""Episode/worst_{worst_rate*100}%_success_rate_train""", worst_mean_train, locs['it'])
                    if self.wandb_writer is not None : self.wandb_writer.log({f"""Episode/worst_{worst_rate*100}%_success_rate_train""": worst_mean_train})
                    ep_string += f"""{f'Mean episode worst {worst_rate*100}% success rate train:':>{pad}} {worst_mean_train:.4f} \n"""
                    if if_val:
                        self.writer.add_scalar(f"""Episode/worst_{worst_rate*100}%_success_rate_valIntra""", worst_mean_valIntra, locs['it'])
                        self.writer.add_scalar(f"""Episode/worst_{worst_rate*100}%_success_rate_valInter""", worst_mean_valInter, locs['it'])
                        if self.wandb_writer is not None :
                            self.wandb_writer.log({f"""Episode/worst_{worst_rate*100}%_success_rate_valIntra""": worst_mean_valIntra,
                                                    f"""Episode/worst_{worst_rate*100}%_success_rate_valInter""": worst_mean_valInter})
                        ep_string += f"""{f'Mean episode worst {worst_rate*100}% success rate valIntra:':>{pad}} {worst_mean_valIntra:.4f} \n"""  
                        ep_string += f"""{f'Mean episode worst {worst_rate*100}% success rate valInter:':>{pad}} {worst_mean_valInter:.4f} \n"""  

                value_train = torch.mean(infotensor_train.float())
                value_valIntra = torch.mean(infotensor_valIntra.float())
                value_valInter = torch.mean(infotensor_valInter.float())

                if key == "dist_tip" or key == "part_pos" :
                    self.writer.add_scalar('Train/' + key + '_train', value_train, locs['it'])
                    if self.wandb_writer is not None : self.wandb_writer.log({'Train/' + key + '_train': value_train})
                    ep_string += f"""{f'{key} train:':>{pad}} {value_train:.4f}\n"""
                    if if_val:
                        self.writer.add_scalar('Train/' + key + '_valIntra', value_valIntra, locs['it'])
                        self.writer.add_scalar('Train/' + key + '_valInter', value_valInter, locs['it'])
                        if self.wandb_writer is not None : 
                            self.wandb_writer.log({'Train/' + key + '_valIntra': value_valIntra, 
                                                    'Train/' + key + '_valInter': value_valInter})
                        ep_string += f"""{f'{key} valIntra:':>{pad}} {value_valIntra:.4f}\n"""
                        ep_string += f"""{f'{key} valInter:':>{pad}} {value_valInter:.4f}\n"""
                else :
                    self.writer.add_scalar('Episode/' + key + '_train', value_train, locs['it'])
                    if self.wandb_writer is not None : self.wandb_writer.log({'Episode/' + key + '_train': value_train})
                    ep_string += f"""{f'Mean episode {key} train:':>{pad}} {value_train:.4f}\n"""
                    if if_val:
                        ep_string += f"""{f'Mean episode {key} valIntra:':>{pad}} {value_valIntra:.4f}\n"""
                        self.writer.add_scalar('Episode/' + key + '_valIntra', value_valIntra, locs['it'])
                        ep_string += f"""{f'Mean episode {key} valInter:':>{pad}} {value_valInter:.4f}\n"""
                        self.writer.add_scalar('Episode/' + key + '_val', value_valInter, locs['it'])
                        if self.wandb_writer is not None :
                            self.wandb_writer.log({'Episode/' + key + '_valIntra': value_valIntra, 
                                                    'Episode/' + key + '_val': value_valInter})
                            

        if locs['task_info']:
            for key in locs['task_info']:
                value = locs['task_info'][key]
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f} \n"""

        mean_std = self.actor_critic.log_std.exp().mean()

        mean_part_position_train, _ = torch.max(self.storage.observations[:self.train_env_num, :, self.order_part_pos], dim=0)
        #mean_part_position_valIntra, _ = torch.max(self.storage.observations[self.train_env_num:self.valIntra_env_num + self.train_env_num, :, self.order_part_pos], dim=0) if self.valIntra_env_num > 0 else (0, 0)
        #mean_part_position_valInter, _ = torch.max(self.storage.observations[self.valIntra_env_num + self.train_env_num:, :, self.order_part_pos], dim=0) if self.valInter_env_num > 0 else (0, 0)
        mean_part_position_train = mean_part_position_train.mean()
        #mean_part_position_valIntra = mean_part_position_valIntra.mean() if self.valIntra_env_num > 0 else 0
        #mean_part_position_valInter = mean_part_position_valInter.mean() if self.valInter_env_num > 0 else 0
        ###wandb###

        if self.wandb_writer is not None :
            try:
                self.wandb_writer.log({
                        'Policy/mean_noise_std': mean_std.item(),
                        'Policy/lr': self.step_size,
                        'Train/mean_reward/step': locs['mean_reward'],
                        'Mean gt reward/step:': statistics.mean(locs['gt_rewbuffer']) / self.max_episode_length,
                        #'Train_/mean_episode_length/episode': locs['mean_trajectory_length'],
                        'Train/part_pos_train' : mean_part_position_train,
                        #'Train/part_pos_valIntra' : mean_part_position_valIntra,
                        #'Train/part_pos_valInter' : mean_part_position_valInter,
                        'Train/dagger_loss_proportion': locs['dagger_loss_proportion'],
                        'Train/seg_loss_proportion': locs['seg_loss_proportion'],
                        'Train/rl_loss_proportion': locs['rl_loss_proportion'],
                        "total_steps": locs['it'], 
                        "step_now": locs['it'] % self.max_episode_length, 
                        'Loss/value_function': locs['mean_value_loss'],
                        'Loss/surrogate': locs['mean_surrogate_loss'],
                        'Loss/rl': locs['mean_rl_loss'],
                        'Loss/dagger': locs['mean_dagger_loss'],
                        'Loss/semseg': locs['mean_seg_loss'],
                        'Loss/seg_accu': locs['mean_seg_accuracy'],
                        })
            except:
                self.wandb_writer.log({
                        'Policy/mean_noise_std': mean_std.item(),
                        'Policy/lr': self.step_size,
                        'Train/mean_reward/step': locs['mean_reward'],
                        #'Train_/mean_episode_length/episode': locs['mean_trajectory_length'],
                        'Train/part_pos_train' : mean_part_position_train,
                        #'Train/part_pos_valIntra' : mean_part_position_valIntra,
                        #'Train/part_pos_valInter' : mean_part_position_valInter,
                        'Train/dagger_loss_proportion': locs['dagger_loss_proportion'],
                        'Train/rl_loss_proportion': locs['rl_loss_proportion'],
                        "total_steps": locs['it'], 
                        "step_now": locs['it'] % self.max_episode_length, 
                        'Loss/value_function': locs['mean_value_loss'],
                        'Loss/surrogate': locs['mean_surrogate_loss'],
                        'Loss/rl': locs['mean_rl_loss'],
                        'Loss/dagger': locs['mean_dagger_loss'],
                        'Loss/semseg': locs['mean_seg_loss'],
                        'Loss/seg_accu': locs['mean_seg_accuracy'],
                        })
                
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/dagger', locs['mean_dagger_loss'], locs['it'])
        self.writer.add_scalar('Loss/semseg', locs['mean_seg_loss'], locs['it'])
        self.writer.add_scalar('Loss/seg_accu', locs['mean_seg_accuracy'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Policy/lr', self.step_size, locs['it'])

        if len(locs['rewbuffer']) > 0:
            if self.wandb_writer is not None : self.wandb_writer.log({
                    'Train/reward': torch.tensor(list(collections.deque(locs['rewbuffer']))),
                     'Mean gt reward': statistics.mean(locs['gt_rewbuffer']) / self.max_episode_length, 
                    #'Train/episode_length/time': torch.tensor(list(collections.deque(locs['lenbuffer']))),
                    })
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            #self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            #self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train/mean_reward/step', locs['mean_reward'], locs['it'])
        #self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.max_episode_length * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        it_now = locs['i'] % self.max_episode_length
        str = f" \033[1m Learning iteration {it_now}-{locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Exp Name':>{pad}} {self.exp_name} \n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'RL loss:':>{pad}} {locs['mean_rl_loss']:.4f}\n"""
                          f"""{'Dagger loss:':>{pad}} {locs['mean_dagger_loss']:.4f}\n"""
                          f"""{'Semseg loss:':>{pad}} {locs['mean_seg_loss']:.4f}\n"""
                          f"""{'Semseg Accu:':>{pad}} {locs['mean_seg_accuracy']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                        #   f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.3f}\n"""
                          f"""{'Mean gt reward:':>{pad}} {(statistics.mean(locs['gt_rewbuffer']) / self.max_episode_length):.2f}\n"""
                          #f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean expert reward/step:':>{pad}} {self.mean_rew_expert:.3f}\n"""
                          f"""{'Mean policy reward/step:':>{pad}} {self.mean_rew_policy:.3f}\n"""
                          #f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                          f"""{'Learning Rate:':>{pad}} {self.step_size}\n"""
                          f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                          #f"""{'Mean_part_position_valIntra:':>{pad}} {mean_part_position_valIntra:.5f}\n"""\
                          #f"""{'Mean_part_position_valInter:':>{pad}} {mean_part_position_valInter:.5f}\n"""\
                          )
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Exp Name':>{pad}} {self.exp_name} \n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'RL loss:':>{pad}} {locs['mean_rl_loss']:.4f}\n"""
                          f"""{'Dagger loss:':>{pad}} {locs['mean_dagger_loss']:.4f}\n"""
                          f"""{'Semseg loss:':>{pad}} {locs['mean_seg_loss']:.4f}\n"""
                          f"""{'Semseg Accu:':>{pad}} {locs['mean_seg_accuracy']:.4f}\n"""
                        #   f"""{'discriminator loss:':>{pad}} {statistics.mean(locs['disc_loss_all']):.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}s\n"""
                          f"""{'Mean expert reward/step:':>{pad}} {self.mean_rew_expert:.3f}\n"""
                          f"""{'Mean policy reward/step:':>{pad}} {self.mean_rew_policy:.3f}\n"""
                        #   f"""{'Mean gt reward:':>{pad}} {locs['gt_rewbuffer']:.2f}\n"""
                          #f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                          f"""{'Learning Rate:':>{pad}} {self.step_size}\n"""
                          f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                          #f"""{'Mean_part_position_valIntra:':>{pad}} {mean_part_position_valIntra:.5f}\n"""\
                          #f"""{'Mean_part_position_valInter:':>{pad}} {mean_part_position_valInter:.5f}\n"""\
                          )

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self, it):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_dagger_loss = 0.
        mean_rl_loss = 0.
        mean_seg_loss = 0.
        mean_seg_accuracy = 0.
        for epoch in range(self.num_learning_epochs):
            batch = self.storage.mini_batch_generator(self.num_mini_batches)
            batch_expert = self.expert_storage.mini_batch_generator(self.num_mini_batches)
            for indices, indices_expert in zip(batch, batch_expert):
                # if not self.backbone_freeze:
                #     observations_batch_expert = self.expert_storage.observations.view(-1, *self.expert_storage.observations.size()[2:])[indices_expert]
                #     if self.use_pc:
                #         pc_batch_expert = self.expert_storage.pointcloud.view(-1, *self.expert_storage.pointcloud.size()[2:])[indices_expert]
                #     # observations_batch = observations_batch[:, self.qpose_num:self.qpose_num+16]
                #     # obs_batch = {"pc": pc_batch, "state": observations_batch}
                # else:
                #     observations_batch_expert = self.expert_storage.observations.view(-1, *self.expert_storage.observations.size()[2:])[indices_expert]
                #     if self.use_pc:
                #         pc_batch_expert = self.expert_storage.pointcloud_features.view(-1, *self.expert_storage.pointcloud_features.size()[2:])[indices_expert]
                #     # observations_batch = observations_batch[:, self.qpose_num:self.qpose_num+16]
                #     # obs_batch = {"pc": pc_batch, "state": observations_batch}
                observations_batch_expert = self.expert_storage.observations.view(-1, *self.expert_storage.observations.size()[2:])[indices_expert]
                if self.use_pc:
                    pc_batch_expert = self.expert_storage.pointcloud.view(-1, *self.expert_storage.pointcloud.size()[2:])[indices_expert]
                # states_batch_expert = self.expert_storage.states.view(-1, *self.expert_storage.states.size()[2:])[indices_expert]

                if self.use_pc:
                    obs_now_expert = Observations(state = observations_batch_expert, obs=observations_batch_expert, points = pc_batch_expert)
                else:
                    obs_now_expert = Observations(state = observations_batch_expert, obs=observations_batch_expert)

                expert_actions_batch = self.expert_storage.expert_actions.reshape(-1, self.expert_storage.actions.size(-1))[indices_expert]
                actions_batch, _ , others = self.actor_critic.act_dagger(obs_now_expert, concat_part_center = self.model_cfg["concat_part_center"]) #self.expert_storage.actions.view(-1, self.storage.actions.size(-1))[indices]

                if self.model_cfg["use_dagger_activation"]:
                    if self.model_cfg["action_normalization"] == "tanh":
                        expert_actions_tensor = torch.tanh(expert_actions_batch) # torch.clamp(expert_actions_batch, -self.vec_env.clip_actions, self.vec_env.clip_actions) # expert_actions_batch # 
                        actions_tensor = torch.tanh(actions_batch) # torch.clamp(actions_batch, -self.vec_env.clip_actions, self.vec_env.clip_actions)
                    elif self.model_cfg["action_normalization"] == "clip":
                        expert_actions_tensor = torch.clamp(expert_actions_batch, -self.vec_env.clip_actions, self.vec_env.clip_actions) # expert_actions_batch # 
                        actions_tensor = torch.clamp(actions_batch, -self.vec_env.clip_actions, self.vec_env.clip_actions)
                    else:
                        expert_actions_tensor = expert_actions_batch # torch.clamp(expert_actions_batch, -self.vec_env.clip_actions, self.vec_env.clip_actions) # expert_actions_batch # 
                        actions_tensor = actions_batch # torch.clamp(actions_batch, -self.vec_env.clip_actions, self.vec_env.clip_actions)
                else:
                    expert_actions_tensor = expert_actions_batch # torch.clamp(expert_actions_batch, -self.vec_env.clip_actions, self.vec_env.clip_actions) # expert_actions_batch # 
                    actions_tensor = actions_batch
                    

                # actions_batch_new[:,3:7] = actions_batch[:,3:7] /  torch.norm(actions_batch[:, 3:7],dim=-1, keepdim=True)
                # expert_actions_batch_new[:,3:7] = expert_actions_batch[:,3:7] /  torch.norm(expert_actions_batch[:, 3:7],dim=-1, keepdim=True)
                # expert_actions_batch[:,3:7] = expert_actions_batch[:,3:7] /  torch.sqrt((expert_actions_batch[:, 3:7]**2).sum(dim=-1)+1e-8).reshape(-1, 1)
                # actions_batch_new[:, 7:9] = torch.sigmoid(actions_batch_new[:, 7:9]/0.1)
                # expert_actions_batch_new[:, 7:9] = torch.sigmoid(expert_actions_batch_new[:, 7:9]/0.1)

                # surrogate_loss = F.mse_loss(actions_batch, expert_actions_batch)  # self.l2(actions_batch, expert_actions_batch)/ actions_batch.shape[0] * self.model_cfg["dagger_loss_supervise"]
                if self.use_seg:
                    seg_loss_proportion = self.model_cfg["dagger_loss_seg"]
                else:
                    seg_loss_proportion = 0.
                
                dagger_loss_proportion = self.model_cfg["dagger_loss_supervise"] * max(0,(1-it*self.model_cfg["dagger_loss_decay_proportion"]))  # self.l2(actions_batch, expert_actions_batch)/ actions_batch.shape[0] * self.model_cfg["dagger_loss_supervise"]
                dagger_loss = F.mse_loss(actions_tensor, expert_actions_tensor) * dagger_loss_proportion

                if self.use_seg:
                    sem_logits = others["sem_logits"]
                    sem_labels = self.expert_storage.seg_labels.reshape(-1, self.expert_storage.seg_labels.shape[-1])[indices_expert].to(torch.long).reshape(-1)
                    if self.model_cfg["seg_focal_loss"]:
                        seg_loss = focal_loss(sem_logits, sem_labels,gamma=2.0,reduction="mean",)
                    else:
                        seg_loss = F.cross_entropy(sem_logits, sem_labels,weight=None,reduction="mean",)
                    seg_loss *= seg_loss_proportion
                else:
                    seg_loss = 0.
                # supervise_loss = dagger_loss
                if self.use_seg:
                    # supervise_loss += seg_loss
                    seg_accuracy = pixel_accuracy(others["sem_preds"], sem_labels)
                else:
                    seg_accuracy = 0.
                self.optimizer_dagger.zero_grad()
                if self.model_cfg["joint_update"] and it % self.model_cfg["update_pn_interval"] != 0:
                    if self.use_seg:
                        seg_loss.backward()
                    self.optimizer_dagger.step()
                elif self.model_cfg["joint_update"] : #只update backbone
                    if self.model_cfg["update_which"] == "backbone":
                        ap = copy.deepcopy(self.actor_critic.actor_mlp.state_dict())
                        cp = copy.deepcopy(self.actor_critic.critic_mlp.state_dict())
                        self.actor_critic.backbone.unfreeze()
                    (dagger_loss + seg_loss).backward()
                    self.optimizer_dagger.step()
                    if self.model_cfg["update_which"] == "backbone":
                        self.actor_critic.backbone.freeze()
                        self.backbone.eval()
                        self.actor_critic.actor_mlp.load_state_dict(ap)
                        self.actor_critic.critic_mlp.load_state_dict(cp)

                else:
                    (dagger_loss + seg_loss).backward()
                    self.optimizer_dagger.step()
                
                
                
                # if not self.backbone_freeze:
                #     observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                #     if self.use_pc:
                #         pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
                #     # observations_batch = observations_batch[:, self.qpose_num:self.qpose_num+16]
                #     # obs_batch = {"pc": pc_batch, "state": observations_batch}
                # else:
                #     observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                #     if self.use_pc:
                #         pc_batch = self.storage.pointcloud_features.view(-1, *self.storage.pointcloud_features.size()[2:])[indices]
                #     # observations_batch = observations_batch[:, self.qpose_num:self.qpose_num+16]
                #     # obs_batch = {"pc": pc_batch, "state": observations_batch}
                observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.use_pc:
                    pc_batch = self.storage.pointcloud.view(-1, *self.storage.pointcloud.size()[2:])[indices]
                states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                if self.use_pc:
                    obs_now = Observations(state = states_batch, obs=observations_batch, points = pc_batch)
                else:
                    obs_now = Observations(state = states_batch, obs=observations_batch)

                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]
                if self.use_pc:   
                    observations = Observations(state = states_batch, obs= observations_batch, points= pc_batch)
                else:
                    observations = Observations(state = states_batch, obs= observations_batch)
                    
                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = \
                    self.actor_critic.evaluate(observations, actions_batch, self.model_cfg["actor_freeze"] ,self.model_cfg["rl_backbone_freeze"], concat_part_center = self.model_cfg["concat_part_center"])

                if self.use_adv_norm:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + self.adv_norm_epsilon)

                # KL learning rate schedule
                if self.desired_kl > 0 and self.schedule == 'adaptive':

                    kl = torch.sum(
                        sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(self.lr_lower, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(self.lr_upper, self.step_size * 1.5)
                        
                    # if it > 2000 :
                    #     self.step_size = max(min(self.step_size, 3e-4 - (it-2000)/1000*3e-4), 0.0)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size
                    
                    # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                
                
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean() 

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                rl_loss_proportion = self.model_cfg["dagger_loss_RL"]
                rl_loss = (value_loss + surrogate_loss) * rl_loss_proportion
                
                # self.network_lr_scheduler.step()

                # update
                self.optimizer.zero_grad()
                rl_loss.backward()
                if self.use_grad_clip:   ###trick 7
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_dagger_loss += dagger_loss.item()
                if self.use_seg:
                    mean_seg_loss += seg_loss.item()

                mean_rl_loss += rl_loss.item()
                mean_seg_accuracy += seg_accuracy
        
        num_update = self.num_mini_batches * self.num_learning_epochs
        mean_value_loss /= num_update
        mean_surrogate_loss /= num_update
        mean_dagger_loss /= num_update
        mean_seg_loss /= num_update
        mean_rl_loss /= num_update
        mean_seg_accuracy /= num_update

        return mean_value_loss, mean_surrogate_loss, mean_dagger_loss, mean_rl_loss, mean_seg_loss, dagger_loss_proportion, seg_loss_proportion, rl_loss_proportion, mean_seg_accuracy
    
    
    def eval_expert(self, it):
        self.vec_env.task.eval()
        self.actor_critic_expert.eval()
        current_obs = self.vec_env.reset()
        self.episode_step_for_now = 0
        total_reward = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num), device=self.device)
        total_success = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num, self.eval_round), device=self.device)
        
        with tqdm(total=self.eval_round) as pbar:
            pbar.set_description('Validating Expert:')
            with torch.no_grad() :
                for r in range(self.eval_round) :
                    current_obs = self.vec_env.reset()
                    self.episode_step_for_now = 0
                    for i in range(self.max_episode_length) :

                        # if self.apply_reset:
                        if self.use_state_norm and not self.use_pc : ### trick 2
                            current_obs = self.state_norm(current_obs)
                
                        # if self.use_first_bbox is not None:
                        #     if not self.use_pc:
                        #         current_obs.state[:, self.qpose_num+16:] = self.init_obs[:, self.qpose_num+16:]
                        #     else:
                        #         current_obs.state[:, self.qpose_num+16:] = self.init_obs["state"][:, self.qpose_num+16:]
                        #         # current_obs.points[:, self.qpose_num+16:] = self.init_obs.points[:, self.qpose_num+16:]
                            # current_obs.state[:, self.qpose_num+16:] = self.init_obs[:, self.qpose_num+16:]

                        if self.use_residual_traj:
                            current_obs = self.residual_actions(current_obs, i)

                        # if self.canonical:
                        #     current_obs = self.canonicalize(current_obs)

                        if self.use_expert:
                            #if self.canonical and not self.use_pc:
                            #    current_obs_expert = current_obs
                            #elif self.expert_cfg["canonicalize"]:
                            #    current_obs_expert = self.canonicalize(current_obs)
                            #else:
                            current_obs_expert = current_obs
                            current_obs_expert.obs = current_obs.state
                            if self.canonical:
                                current_obs_expert = current_obs_expert
                            elif self.expert_cfg["canonicalize"]:
                                current_obs_expert = self.canonicalize(current_obs_expert)
                            else:
                                current_obs_expert = current_obs
                            # if self.expert_cfg["canonicalize"]:
                            #     current_obs_expert = self.canonicalize(current_obs)
                            # else:
                            #     current_obs_expert = current_obs
                            # # current_obs_expert.obs = current_obs_expert.state

                            expert_actions, _ = self.actor_critic_expert.act_inference(current_obs_expert, concat_part_center = self.model_cfg["concat_part_center"])

                            if self.expert_cfg["canonicalize"]:
                                expert_actions = self.uncanonicalize(expert_actions)
                        

                        # if self.use_residual_traj:
                        #     actions = self.residual_actions(actions, i)
                        # next_obs = current_obs
                        # current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)
                        if self.test_save:
                            self.test_obs.append(current_obs)
                        
                        
                        
                        if self.expert_cfg["action_clip"]:
                            expert_actions = torch.clamp(expert_actions, -self.vec_env.clip_actions, self.vec_env.clip_actions) # expert_actions_batch # 

                        next_obs, rews, dones, infos = self.vec_env.step(expert_actions)
                        # next_obs_clouds, next_obs_states, rews, dones, infos = self.vec_env.step(actions)
                        current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)

                        total_reward += rews.to(self.device)
                        total_success[:, r] = torch.logical_or(infos["successes"].to(self.device), total_success[:, r])
                        # if infos["successes"].item() != 0 :

                    # print(total_reward/self.max_episode_length, total_success)
                    # pdb.set_trace()
                    pbar.update(1)

        
        train_reward = total_reward[:self.train_env_num].mean() / self.max_episode_length / self.eval_round
        valIntra_reward = total_reward[self.train_env_num:self.valIntra_env_num + self.train_env_num].mean() / self.max_episode_length / self.eval_round
        valInter_reward = total_reward[self.valIntra_env_num + self.train_env_num:].mean() / self.max_episode_length / self.eval_round
        train_success = total_success[:self.train_env_num].float().mean()
        valIntra_success = total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float().mean()
        valInter_success = total_success[self.valIntra_env_num + self.train_env_num:].float().mean()

        train_reward = train_reward.cpu().item()
        valIntra_reward = valIntra_reward.cpu().item()
        valInter_reward = valInter_reward.cpu().item()
        train_success = train_success.cpu().item()
        valIntra_success = valIntra_success.cpu().item()
        valInter_success = valInter_success.cpu().item()

        #if self.is_testing:
        print("Expert $ Training set average reward:     ", train_reward)
        print("Expert $ ValIntra set average reward:      ", valIntra_reward)
        print("Expert $ ValInter set average reward:      ", valInter_reward)
        print("Expert $ Training set average success:    ", train_success)
        print("Expert $ ValIntra set average success:     ", valIntra_success)
        print("Expert $ ValInter set average success:     ", valInter_success)
        print("Expert $ per eval_round success")
        print("Expert $", torch.mean(total_success[:self.train_env_num].float(), dim = 0))
        print("Expert $", torch.mean(total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float(), dim = 0))
        print("Expert $", torch.mean(total_success[self.valIntra_env_num + self.train_env_num:].float(), dim = 0))
        print("Expert $", "per asset success")
        print("Expert $", torch.mean(total_success[:self.train_env_num].float().reshape(self.train_env_num//self.env_per_asset, self.env_per_asset * self.eval_round),dim = 1))
        print("Expert $", torch.mean(total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float().reshape(self.valIntra_env_num//self.env_per_asset, self.env_per_asset * self.eval_round), dim = 1))
        print("Expert $", torch.mean(total_success[self.valIntra_env_num + self.train_env_num:].float().reshape(self.valInter_env_num//self.env_per_asset, self.env_per_asset * self.eval_round), dim = 1))
        asset_train_mean = torch.mean(total_success[:self.train_env_num].float().reshape(self.train_env_num//self.env_per_asset, self.env_per_asset * self.eval_round),dim = 1)
        if self.is_testing:
            not_success = []
            for i in range(self.train_env_num // self.env_per_asset):
                if (asset_train_mean[i] <= 0.8):
                    not_success.append(i)
            print("not success train assert num")
            print(not_success)
       #print("Training set success list:")
        #for x in total_success[:self.train_env_num] / self.eval_round :
        #    print(x.cpu().item(), end=' ')

        #print("\n\nTesting set success list:")
        #for x in total_success[self.train_env_num:] / self.eval_round :
            #print(x.cpu().item(), end=' ')
        
        #print('\n')

        self.writer.add_scalar('"Expert $", Test/' + 'TestSuccessRate/TrainSet', train_success, it)
        self.writer.add_scalar('"Expert $", Test/' + 'TestSuccessRate/ValIntraSet', valIntra_success, it)
        self.writer.add_scalar('"Expert $", Test/' + 'TestSuccessRate/ValInterSet', valInter_success, it)
        self.writer.add_scalar('"Expert $", Test/' + 'TestReward/TrainSet', train_reward, it)
        self.writer.add_scalar('"Expert $", Test/' + 'TestReward/ValIntraSet', valIntra_reward, it)
        self.writer.add_scalar('"Expert $", Test/' + 'TestReward/ValInterSet', valInter_reward, it)
        
        self.vec_env.task.train()

        return (train_reward, valIntra_reward, valInter_reward,  train_success, valIntra_success, valInter_success)
     
    def eval(self, it) :
        
        self.vec_env.task.eval()
        self.actor_critic.eval()
        current_obs = self.vec_env.reset()
        self.episode_step_for_now = 0
        total_reward = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num), device=self.device)
        total_success = torch.zeros((self.train_env_num+self.valIntra_env_num + self.valInter_env_num, self.eval_round), device=self.device)
        
        with tqdm(total=self.eval_round) as pbar:
            pbar.set_description('Validating:')
            with torch.no_grad() :
                for r in range(self.eval_round) :
                    for i in range(self.max_episode_length):
                        if self.apply_reset:
                            current_obs = self.vec_env.reset()
                            self.episode_step_for_now = 0
                        if self.use_state_norm and not self.use_pc : ### trick 2
                            current_obs = self.state_norm(current_obs)
                        if self.use_residual_traj:
                            current_obs = self.residual_actions(current_obs, i)

                        actions, _ = self.actor_critic.act_inference(current_obs, concat_part_center = self.model_cfg["concat_part_center"])

                        if self.canonical:
                            actions = self.uncanonicalize(actions)

                        if self.test_save:
                            self.test_obs.append(current_obs)

                        next_obs, rews, dones, infos = self.vec_env.step(actions)
                        current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)
                    
                        total_reward += rews.to(self.device)
                        total_success[:, r] = torch.logical_or(infos["successes"].to(self.device), total_success[:, r])
                    pbar.update(1)
        if self.test_save:
            
            points = self.test_obs[0].points[..., :self.numPCObservations] #.reshape((1, self.test_save[0].points.shape))
            total =len(self.test_obs)
            for i in range(1, total):
                points= torch.cat((points, self.test_obs[i].points[..., :self.numPCObservations]))
            points = points.reshape(len(self.test_obs)//200, 200, self.test_obs[0].points.shape[0], self.test_obs[0].points.shape[1], 6)
            import pdb
            pdb.set_trace()
            torch.save(points, self.test_save_path)
        
        train_reward = total_reward[:self.train_env_num].mean() / self.max_episode_length / self.eval_round
        valIntra_reward = total_reward[self.train_env_num:self.valIntra_env_num + self.train_env_num].mean() / self.max_episode_length / self.eval_round
        valInter_reward = total_reward[self.valIntra_env_num + self.train_env_num:].mean() / self.max_episode_length / self.eval_round
        train_success = total_success[:self.train_env_num].float().mean()
        valIntra_success = total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float().mean()
        valInter_success = total_success[self.valIntra_env_num + self.train_env_num:].float().mean()

        train_reward = train_reward.cpu().item()
        valIntra_reward = valIntra_reward.cpu().item()
        valInter_reward = valInter_reward.cpu().item()
        train_success = train_success.cpu().item()
        valIntra_success = valIntra_success.cpu().item()
        valInter_success = valInter_success.cpu().item()

        #if self.is_testing:
        print("Training set average reward:     ", train_reward)
        print("ValIntra set average reward:      ", valIntra_reward)
        print("ValInter set average reward:      ", valInter_reward)
        print("Training set average success:    ", train_success)
        print("ValIntra set average success:     ", valIntra_success)
        print("ValInter set average success:     ", valInter_success)
        print("per eval_round success")
        print(torch.mean(total_success[:self.train_env_num].float(), dim = 0))
        print(torch.mean(total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float(), dim = 0))
        print(torch.mean(total_success[self.valIntra_env_num + self.train_env_num:].float(), dim = 0))
        print("per asset success")
        print(torch.mean(total_success[:self.train_env_num].float().reshape(self.train_env_num//self.env_per_asset, self.env_per_asset * self.eval_round),dim = 1))
        print(torch.mean(total_success[self.train_env_num:self.valIntra_env_num + self.train_env_num].float().reshape(self.valIntra_env_num//self.env_per_asset, self.env_per_asset * self.eval_round), dim = 1))
        print(torch.mean(total_success[self.valIntra_env_num + self.train_env_num:].float().reshape(self.valInter_env_num//self.env_per_asset, self.env_per_asset * self.eval_round), dim = 1))
        asset_train_mean = torch.mean(total_success[:self.train_env_num].float().reshape(self.train_env_num//self.env_per_asset, self.env_per_asset * self.eval_round),dim = 1)
        if self.is_testing:
            not_success = []
            for i in range(self.train_env_num // self.env_per_asset):
                if (asset_train_mean[i] <= 0.8):
                    not_success.append(i)
            print("not success train assert num")
            print(not_success)
       #print("Training set success list:")
        #for x in total_success[:self.train_env_num] / self.eval_round :
        #    print(x.cpu().item(), end=' ')

        #print("\n\nTesting set success list:")
        #for x in total_success[self.train_env_num:] / self.eval_round :
            #print(x.cpu().item(), end=' ')
        
        #print('\n')

        self.writer.add_scalar('Test/' + 'TestSuccessRate/TrainSet', train_success, it)
        self.writer.add_scalar('Test/' + 'TestSuccessRate/ValIntraSet', valIntra_success, it)
        self.writer.add_scalar('Test/' + 'TestSuccessRate/ValInterSet', valInter_success, it)
        self.writer.add_scalar('Test/' + 'TestReward/TrainSet', train_reward, it)
        self.writer.add_scalar('Test/' + 'TestReward/ValIntraSet', valIntra_reward, it)
        self.writer.add_scalar('Test/' + 'TestReward/ValInterSet', valInter_reward, it)
        
        self.vec_env.task.train()
        self.actor_critic.train()
        return (train_reward, valIntra_reward, valInter_reward,  train_success, valIntra_success, valInter_success)




def space_add(a, b):

    if len(a.shape) != 1 or len(b.shape) != 1 :
        
        raise TypeError("Shape of two spaces need to be 1d")
    
    elif not isinstance(a, Box) or not isinstance(b, Box) :

        raise TypeError("Type of two spaces need to be Box")
    
    else :

        low = np.concatenate((a.low, b.low))
        high = np.concatenate((a.high, b.high))
        return Box(low=low, high=high)
    
    
    def test(self, path, expert_path):
        #root_path, model_name = os.path.split(path)
        #self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        #self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        #self.actor_critic.eval()
        #if self.vec_env.task.task_meta["need_update"] :
        #    self.vec_env.task.load(path, self.current_learning_iteration)
        #    self.vec_env.task.eval()
        # assert os.path.exists(self.checkpoint_path)
        # checkpoint_dict= torch.load(self.checkpoint_path, map_location=self.device)
        # self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"])
        self.current_learning_iteration = checkpoint_dict["iteration"]
        self.actor_critic.eval()
        self.actor_critic.eval_expert()
        if self.test_save:
            points = (self.test_save[0].points)#.reshape((1, self.test_save[0].points.shape))
            for i in range(1, len(self.save)):
                points.cat((points, self.test_save[0].points))
            points = points.reshape(len(self.save), 200, -1, 6)
