import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math 

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, net_cfg, proprio_shape):
        super().__init__()
        hidden_dim = net_cfg['hid_dim']
        activation = get_activation(net_cfg['activation'])
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(activation)
        for l in range(len(hidden_dim)):
            if l == len(hidden_dim) - 1:
                layers.append(nn.Linear(hidden_dim[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dim[l], hidden_dim[l + 1]))
                layers.append(activation)
        self.model = nn.Sequential(*layers)

        # Initialize the weights like in stable baselines
        init_weights = [math.sqrt(2)] * len(hidden_dim)
        self.output_dim = output_dim
        if output_dim == 1:
            init_weights.append(1)
        else:
            init_weights.append(0.01)
        [torch.nn.init.orthogonal_(module.weight, gain=init_weights[idx]) for idx, module in
         enumerate(mod for mod in self.model if isinstance(mod, nn.Linear))]

    def forward(self, x):
        return self.model(x)


def conv(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


def conv_stride(in_channels, out_channels, kernel_size, stride):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2
    )


class Conv3DNet(nn.Module):
    def __init__(self, input_dim, output_dim, net_cfg, proprio_shape):
        super().__init__()
        self.res = round(input_dim**(1/3))
        self.encoder = Encoder(1, [16, 32, 32], [5, 3, 3], [3, 3, 2], net_cfg['activation'])
        self.activation = get_activation(net_cfg['activation'])
        # self.avgpool = nn.AvgPool3d(kernel_size=7, stride=2)
        self.final_mlp = nn.Sequential(
            nn.Linear(32*27+proprio_shape, 256),
            self.activation,
            nn.Linear(256, output_dim),
        )
        self.proprio_shape = proprio_shape

    def forward(self, x_in):
        batch = x_in.shape[0]
        # tsdf [bs, 1, 50, 50, 50]
        if self.proprio_shape != 0:
            proprio_state = x_in[:, -self.proprio_shape:]
            tsdf = x_in[:, :-self.proprio_shape].reshape(batch, 1, self.res, self.res, self.res) 
        else:
            tsdf = x_in.reshape(batch, 1, self.res, self.res, self.res) 
        x = self.encoder(tsdf)  # [bs, 32, 3, 3, 3]
        # x = self.avgpool(x).squeeze()   # [bs, 64]
        if self.proprio_shape != 0:
            x = torch.cat((x.reshape(batch, -1), proprio_state), dim=-1)
        x = self.final_mlp(x.reshape(batch, -1))    # [bs, 32*27]
        return x


class PoolConv3DNet(nn.Module):
    def __init__(self, input_dim, output_dim, net_cfg, proprio_shape):
        super().__init__()
        self.res = round(input_dim**(1/3))
        self.encoder = Encoder(1, [16, 32, 64], [5, 3, 3], [2, 2, 2], net_cfg['activation'])
        self.activation = get_activation(net_cfg['activation'])
        self.maxpool = nn.MaxPool3d(kernel_size=4)
        self.final_mlp = nn.Sequential(
            nn.Linear(64, 32),
            self.activation,
            nn.Linear(32, output_dim),
        )

    def forward(self, tsdf):
        # tsdf [bs, 1, 50, 50, 50]
        batch = tsdf.shape[0]
        tsdf = tsdf.reshape(batch, 1, self.res, self.res, self.res) 
        x = self.encoder(tsdf)  # [bs, 64, 7, 7, 7]
        x = self.maxpool(x).squeeze()
        x = self.final_mlp(x.reshape(batch, -1))    # [bs, 64]
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, filters, kernels, stride, acti_name):
        super().__init__()
        self.conv1 = conv_stride(in_channels, filters[0], kernels[0],stride[0])
        self.conv2 = conv_stride(filters[0], filters[1], kernels[1],stride[1])
        self.conv3 = conv_stride(filters[1], filters[2], kernels[2],stride[2])
        # self.conv4 = conv_stride(filters[2], filters[3], kernels[3],stride[3])
        self.activation = get_activation(acti_name)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.activation(x)
        # x = self.conv4(x)
        # x = self.activation(x)
        return x

class PointNet(nn.Module):
    def __init__(self, input_dim, output_dim, net_cfg, proprio_shape):
        super().__init__()
        self.activation = get_activation(net_cfg['activation'])
        self.max_mean_concat = net_cfg['max_mean']
        self.point_num = 1024
        self.mlp = nn.Sequential(
            nn.Linear(input_dim//self.point_num, 128),
            self.activation,
            nn.Linear(128, 256),
            self.activation,
            nn.Linear(256, 512),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(512*(1+self.max_mean_concat)+proprio_shape, 128),
            self.activation,
            nn.Linear(128, 32),
            self.activation,
            nn.Linear(32, output_dim),
        )
        self.proprio_shape = proprio_shape
        self.substract_mean = net_cfg['sub_mean']
        self.count = 0

    def forward(self, x):
        if self.proprio_shape != 0:
            proprio_state = x[:, -self.proprio_shape:]
            pc = x[:, :-self.proprio_shape].reshape(x.shape[0], self.point_num, -1) 
        else:
            pc = x.reshape(x.shape[0], self.point_num, -1) 

        if self.substract_mean:
            pc[...,:3] = pc[...,:3] - pc[...,:3].mean(dim=1,keepdim=True)

        x = self.mlp(pc)
        if self.max_mean_concat:
            x1 = x.max(dim=1)[0]
            x2 = x.mean(dim=1)
            x = torch.cat((x1, x2), dim=-1)
        else:
            x = x.max(dim=1)[0]
        
        # if self.count % 50 == 0:
        #     import numpy as np 
        #     pc_ind = x.max(dim=1)[1]
        #     help = np.zeros((1024, 3))
        #     pcsave = pc[0].cpu().numpy()
        #     help[pc_ind[0].cpu().numpy(),0] = 255
        #     pcsave = np.concatenate((pcsave, help), axis=-1)
        #     np.savetxt(f'debug-{self.count}.txt', pcsave)
        # self.count += 1

        if self.proprio_shape != 0:
            x = torch.cat((x, proprio_state), dim=-1)

        out = self.final_mlp(x)
        
        return out 
    


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, net_cfg, proprio_shape):
        super().__init__()
        self.h = 288 // 4
        self.w = 512 // 4
        self.resnet34 = torchvision.models.resnet34(pretrained=True)
        del self.resnet34.fc
        self.resnet34.fc = nn.Identity()
        self.proprio_shape = proprio_shape
        self.activation = get_activation(net_cfg['activation'])
        self.final_mlp = nn.Sequential(
            nn.Linear(512+proprio_shape, 128),
            self.activation,
            nn.Linear(128, 32),
            self.activation,
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        if self.proprio_shape != 0:
            proprio_state = x[:, -self.proprio_shape:]
            img = x[:, :-self.proprio_shape].reshape(x.shape[0], 3, self.h, self.w) 
        else:
            img = x.reshape(x.shape[0], 3, self.h, self.w) 

        x = self.resnet34(img)

        if self.proprio_shape != 0:
            x = torch.cat((x, proprio_state), dim=-1)

        out = self.final_mlp(x)
        
        return out 


class depthResNet(nn.Module):
    def __init__(self, input_dim, output_dim, net_cfg, proprio_shape):
        super().__init__()
        self.h = 288 // 4
        self.w = 512 // 4
        self.resnet34 = torchvision.models.resnet34(pretrained=False)
        del self.resnet34.fc
        self.resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.resnet34.fc = nn.Identity()
        self.proprio_shape = proprio_shape
        self.activation = get_activation(net_cfg['activation'])
        self.final_mlp = nn.Sequential(
            nn.Linear(512+proprio_shape, 128),
            self.activation,
            nn.Linear(128, 32),
            self.activation,
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        if self.proprio_shape != 0:
            proprio_state = x[:, -self.proprio_shape:]
            img = x[:, :-self.proprio_shape].reshape(x.shape[0], 1, self.h, self.w) 
        else:
            img = x.reshape(x.shape[0], 1, self.h, self.w) 

        x = self.resnet34(img)

        if self.proprio_shape != 0:
            x = torch.cat((x, proprio_state), dim=-1)

        out = self.final_mlp(x)
        
        return out 