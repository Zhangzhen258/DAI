# from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F



def conv7x7(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride= 1, padding= 7//2, bias=False)

def conv9x9(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=9, stride=1, padding=4, bias=False)

def conv5x5(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1, padding=5//2, bias=False)

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)



class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
        self.fc2 = nn.Linear(style_dim, num_features*2)
        self.fc3 = nn.Linear(style_dim, num_features * 2)
        self.noise = conv1x1(num_features, num_features)
        self.sen = nn.Parameter(torch.Tensor([1.0])).cuda()

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        x_nrom = self.norm(x)
        return (1 + gamma) * x_nrom + beta



class AdaIN2d__Noise(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
        self.noise = conv1x1(num_features, num_features)
    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        self.norm_x = self.norm(x) + torch.randn_like(x)
        return (1 + gamma) * self.norm_x + beta


class cnnGenerator(nn.Module):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[192, 320]):
        ''' w_ln 局部噪声权重
        '''
        super().__init__()
        stride = (kernelsize-1)//2
        self.zdim = zdim = 10
        self.imdim = imdim
        self.imsize = imsize
        self.conv1 = nn.Conv2d(imdim, n, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(n, 2*n, kernelsize, 1, stride)
        self.adain2 = AdaIN2d__Noise(self.zdim, 2 * n)
        self.conv3 = nn.Conv2d(2*n, 4 * n, kernelsize, 1, stride)
        self.conv4 = nn.Conv2d(4*n, imdim, kernelsize, 1, stride)
        self.noise = nn.Parameter(torch.Tensor([1.0])).cuda()
    def forward(self, x, rand=False):
        ''' x '''

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if rand:
            z = torch.randn(len(x), self.zdim).cuda()
            x = self.adain2(x, z)
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x



