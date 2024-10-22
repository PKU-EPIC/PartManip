import torch 

class RunningMeanStd:
    def __init__(self, shape, device):  
        self.n = 0
        self.mean = torch.zeros((1,shape),device = device)
        self.S = torch.ones((1,shape), device = device) * 1e-4          
        self.std = torch.sqrt(self.S)

    def update(self, x):
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        self.n += 1
        old_mean = self.mean.clone()
        new_mean = x.mean(dim=0, keepdim=True)
        self.mean = old_mean + (new_mean - old_mean) / self.n
        self.S = self.S + (x-new_mean).pow(2).mean(dim=0, keepdim=True) + (old_mean-new_mean).pow(2) *(self.n-1) / self.n
        self.std = torch.sqrt(self.S / self.n)
        return 
    
    def load(self, load_dict):
        self.mean = load_dict['mean']
        self.std = load_dict['std']
        self.S = load_dict['S']
        self.n = load_dict['n']
        return 

    def save(self):
        save_dict = {
            'mean': self.mean,
            'std': self.std, 
            'S': self.S,
            'n': self.n
        }
        return save_dict

class Normalization:
    def __init__(self, shape, device):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=True):
        # TO check: Is RMS harmful for convergence?
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / self.running_ms.std
        return x


class AdvScaling:
    def __init__(self, shape, device):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x):
        step, env = x.shape[0], x.shape[1]
        self.running_ms.update(x.reshape(-1, 1))
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x.reshape(step, env, 1)



if __name__ == '__main__':
    # check RMS
    k = 1000
    dim = 7
    rms = RunningMeanStd(dim)
    for i in range(10):
        x = torch.randn((k,dim))
        if i == 0:
            allx = x
        else:
            allx = torch.cat((allx,x), dim=0)
        print(x.shape)

        # gt
        mean = allx.mean(dim=0)
        std = torch.sqrt((allx - mean).pow(2).mean(dim=0))
        print('gt', mean, std)

        # rms
        rms.update(x)
        print('rms', rms.mean, rms.std)
    
