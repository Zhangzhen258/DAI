import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class RandomConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(RandomConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize weights with Gaussian distribution N(0, 1)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=1, padding=1)


class cnnGenerator(nn.Module):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[192, 320]):

        super().__init__()
        stride = (kernelsize - 1) // 2
        self.zdim = zdim = 10
        self.imdim = imdim
        self.imsize = imsize

        # self.conv1 = RandomConvLayer(imdim, n, kernelsize)
        # self.conv2 = RandomConvLayer(n, 2 * n, kernelsize)
        # self.conv3 = RandomConvLayer(2 * n, 4 * n, kernelsize)
        # self.conv4 = RandomConvLayer(4 * n, imdim, kernelsize)
        self.conv1 = RandomConvLayer(imdim, imdim, kernelsize)
        self.conv2 = RandomConvLayer(imdim, imdim, kernelsize)

    def forward(self, x):
        x_o = x.clone()
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = self.conv2(x)
        rand = torch.rand(len(x), 1, 1, 1).cuda()
        x = rand * x + (1 - rand) * x_o
        x_c = torch.norm(x, p='fro')
        x_cf = torch.norm(x_o, p='fro')
        x = x / x_c * x_cf
        return x




if __name__ == '__main__':
    x = torch.ones(4, 3, 32, 32)
    z = torch.ones(4, 10)

    # g = stnGenerator(10, [32, 32])
    # y = g(x, z)


