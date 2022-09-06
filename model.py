from turtle import forward
import torch
import torchsummary
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Res_block(torch.nn.Module):

    def __init__(self,n_feats):
        super(Res_block, self).__init__()

        self.conv1 = torch.nn.Conv2d(n_feats,n_feats,3,padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(n_feats,n_feats,3,padding=1)

    def forward(self, x):
        sample = x
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = res*0.1
        output = sample + res
        return output
    
class Upsample(torch.nn.Module):
    def __init__(self,scale,n_feats):
        super(Upsample, self).__init__()
        m = []
        for i in range(scale//2):
            m.append(torch.nn.Conv2d(n_feats,3*scale**2,3,padding=1))
            m.append(torch.nn.PixelShuffle(upscale_factor=scale))
        self.body = torch.nn.Sequential(*m)
        
    def forward(self, x):
        output = self.body(x)
        return output

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class EDSR(torch.nn.Module):
    def __init__(self,scale,n_feats):
        super(EDSR,self).__init__()
        self.head = torch.nn.Conv2d(3,n_feats,3,padding=1)
        body = [ Res_block(n_feats) for i in range(32)]
        body.append(torch.nn.Conv2d(n_feats,n_feats,3,padding=1))
        self.body = torch.nn.Sequential(*body)
        self.tail = Upsample(scale,n_feats)
        self.sub_mean = MeanShift(rgb_range=1.0)
        self.add_mean = MeanShift(rgb_range=1.0,sign=1)
        
    def forward(self,x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        x = self.add_mean(x)
        return x
    
if __name__ == '__main__':
    a = EDSR(3,n_feats=64).to(device)
    torchsummary.summary(a,(3,48,48))