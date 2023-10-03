import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from tpi.core.config import cfg


class PCMLP:
    def __init__(self, env_spec,
                 hidden_sizes=(64,64),
                 min_log_std=-3,
                 init_log_std=0,
                 seed=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std
        self.has_frozen_pointnet = False
        self.is_cuda = True
        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        self.model = MuNet(self.n, self.m, hidden_sizes=hidden_sizes)
        self.learn_log_std = cfg.POLICY_LEARN_LOG_STD
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
            param.data = 1e-2 * param.data
        if cfg.POLICY_LEARN_LOG_STD:
            self.log_std = (torch.ones(self.m) * init_log_std).cuda().requires_grad_()
            #self.log_std = self.log_std.cuda().requires_grad_()
            self.trainable_params = list(self.model.parameters()) + [self.log_std]
        else:
            self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=False)
            self.trainable_params = list(self.model.parameters())

        # Old Policy network
        # ------------------------
        self.old_model = MuNet(self.n, self.m, hidden_sizes=hidden_sizes)
        if cfg.POLICY_LEARN_LOG_STD:
            #self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
            self.old_log_std = Parameter((torch.ones(self.m) * init_log_std).cuda())
            print(self.old_log_std.is_leaf)
            self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        else:
            self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
            self.old_params = list(self.old_model.parameters())
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

        print(f'old is leaf: {self.old_log_std.is_leaf}')
        # Explore Policy Network to sample from environments
        # ------------------------
        """
        self.exp_model = MuNet(self.n, self.m, hidden_sizes=hidden_sizes)
        if cfg.POLICY_LEARN_LOG_STD:
            self.exp_log_std = Variable(torch.ones(self.m) * init_log_std)
            self.exp_params = list(self.exp_model.parameters()) + [self.exp_log_std]
        else:
            self.exp_log_std = Variable(torch.ones(self.m) * init_log_std)
            self.exp_params = list(self.exp_model.parameters())
        for idx, param in enumerate(self.exp_params):
            param.data = self.trainable_params[idx].data.clone()
        """

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.cpu().data.numpy().ravel())
        self.param_shapes = [p.cpu().data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.cpu().data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

        # init with cuda
        self.model.cuda()
        self.old_model.cuda()
        self.model.in_shift = self.model.in_shift.cuda() 
        self.model.in_scale = self.model.in_scale.cuda()
        self.model.out_shift = self.model.out_shift.cuda()
        self.model.out_scale = self.model.out_scale.cuda()


    # Cuda and Cpu conversion
    def to_cuda(self):
        self.is_cuda = True
        self.model.cuda()
        self.old_model.cuda()
        #self.log_std = self.log_std.cpu().data.cuda().requires_grad_()
        self.old_log_std = self.old_log_std.cuda()
        #self.trainable_params

        self.model.in_shift = self.model.in_shift.cuda() 
        self.model.in_scale = self.model.in_scale.cuda()
        self.model.out_shift = self.model.out_shift.cuda()
        self.model.out_scale = self.model.out_scale.cuda()
        #self.trainable_params[-1].requires_grad = True

    def to_cpu(self):
        self.is_cuda = False
        self.model.cpu()
        self.old_model.cpu()
        log_std = self.log_std.cpu().data.requires_grad_()
        old_log_std = self.old_log_std.cpu().data.requires_grad_()
        del self.log_std
        del self.old_log_std
        self.log_std = log_std
        self.old_log_std = old_log_std

        self.trainable_params[-1] = self.log_std

        self.model.in_shift = self.model.in_shift.cpu() 
        self.model.in_scale = self.model.in_scale.cpu()
        self.model.out_shift = self.model.out_shift.cpu()
        self.model.out_scale = self.model.out_scale.cpu()
        #self.trainable_params[-1].requires_grad = True
        #torch.cuda.empty_cache() 
    
    def freeze_pointnet(self):
        self.has_frozen_pointnet = True
        if self.learn_log_std:
            self.trainable_params = list(self.model.parameters())[self.model.num_pointnet_layers:] + [self.log_std]
        else:
            self.trainable_params = list(self.model.parameters())
        self.param_shapes = [p.cpu().data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.cpu().data.numpy().size for p in self.trainable_params]

    
    
    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).cpu().data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    """
    def set_exp_param_values(self, new_params):
        current_idx = 0
        for idx, param in enumerate(self.exp_params):
            vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
            vals = vals.reshape(self.param_shapes[idx])
            param.data = torch.from_numpy(vals).float()
            current_idx += self.param_sizes[idx]
        # clip std at minimum value
        self.exp_params[-1].data = \
            torch.clamp(self.exp_params[-1], self.min_log_std).data
    """

    def set_param_values(self, new_params, set_new=True, set_old=True):
        #print(f'cuda before set param: {next(self.model.parameters()).is_cuda}')
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                if param.is_cuda:
                    param.data = torch.from_numpy(vals).float().cuda()
                else:
                    param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            #print(f'cuda after set new param: {next(self.model.parameters()).is_cuda}')
            # clip std at minimum value
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.cpu().data.numpy().ravel())
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                if param.is_cuda:
                    param.data = torch.from_numpy(vals).float().cuda()
                else:
                    param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[-1].data = \
                torch.clamp(self.old_params[-1], self.min_log_std).data
        #print(f'cuda after set param: {next(self.model.parameters()).is_cuda}')
    # Main functions
    # ============================================
    def get_action(self, observation):
        #self.exp_model.eval()
        self.model.eval()
        
        with torch.no_grad():
            o = np.float32(observation.reshape(1, -1))
            self.obs_var.data = torch.from_numpy(o)
            #mean = self.exp_model(self.obs_var).data.numpy().ravel()
            mean = self.model(self.obs_var).data.numpy().ravel()
            noise = np.exp(self.log_std_val) * np.random.randn(self.m)
            action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def mean_LL(self, observations, actions, model=None, log_std=None):
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        mean = model(obs_var)
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)


class MuNet(nn.Module):
    def __init__(self, obs_dim, act_dim, pc_dim=100*3, emb_dim=32, hidden_sizes=(64,64),
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(MuNet, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pc_dim = pc_dim
        self.emb_dim = emb_dim
        self.hidden_sizes = hidden_sizes
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        
        # note that it will change the order of parameters # 
        self.num_pointnet_layers = 12 # (3 + 3) * 2
        self.conv1 = nn.Conv1d(3, 16, 1)
        self.conv2 = nn.Conv1d(16, 32, 1)
        self.conv3 = nn.Conv1d(32, emb_dim, 1)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(emb_dim) 
        
        self.fc0   = nn.Linear(obs_dim - pc_dim, hidden_sizes[0])
        self.fc1   = nn.Linear(hidden_sizes[0] + emb_dim, hidden_sizes[1])
        self.fc2   = nn.Linear(hidden_sizes[1], act_dim)

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)
        self.in_shift  = Variable(self.in_shift, requires_grad=False)
        self.in_scale  = Variable(self.in_scale, requires_grad=False)
        self.out_shift = Variable(self.out_shift, requires_grad=False)
        self.out_scale = Variable(self.out_scale, requires_grad=False)

    def forward(self, x):
        
        # normalization will be used if the model is pretrained with BC #
        out = (x - self.in_shift)/(self.in_scale + 1e-8)
        #out = x
        num_points = out.shape[0]
        pc_obs = out[:, -self.pc_dim:]
        pc_obs = pc_obs.view(num_points, 3, -1)
        original_obs = out[:, :-self.pc_dim]
        # pointcloud processing
        pc_x = self.conv1(pc_obs)
        pc_x = self.bn1(pc_x)
        pc_x = F.relu(pc_x)
        
        pc_x = self.conv2(pc_x)
        pc_x = self.bn2(pc_x)
        pc_x = F.relu(pc_x)
        
        pc_x = self.conv3(pc_x)
        pc_x = self.bn3(pc_x)
        pc_x = torch.max(pc_x, 2, keepdim=True)[0]
        pc_x = pc_x.view(-1, self.emb_dim)
        out = torch.tanh(self.fc0(original_obs))
        out = torch.cat((out, pc_x), 1)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        out = out * self.out_scale + self.out_shift 
        return out



### for testing ###
"""
class Spec():
    def __init__(self):
        self.observation_dim = 500
        self.action_dim = 10

spec = Spec()

policy = PCMLP(env_spec=spec)
params = policy.model.parameters()
print(params, type(params))
for param in params:
    print(param.shape)


for name, param in policy.model.named_parameters():
    print(name, param.shape)
    

a = nn.parameter.Parameter(torch.randn(5))
#print(a.data.requires_grad)
b = a.data
b.requires_grad = True
#print(f'b: {b}')
#print(a)
#print(list(policy.model.named_parameters()))
"""
