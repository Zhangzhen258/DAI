#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# class ConvNet(nn.Module):
#     ''' 网络结构和cvpr2020的 M-ADA 方法一致 '''
#     def __init__(self, imdim=3):
#         super(ConvNet, self).__init__()
#
#         self.conv1 = nn.Conv2d(imdim, 64, kernel_size=5, stride=1, padding=0)
#         self.mp = nn.MaxPool2d(2)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.fc1 = nn.Linear(128*5*5, 1024)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.relu4 = nn.ReLU(inplace=True)
#
#         self.cls_head_src = nn.Linear(1024, 10)
#         # self.cls_head_tgt = nn.Linear(1024, 10)
#         self.pro_head = nn.Linear(1024, 128)
#         # self.relu5 = nn.ReLU(inplace=True) ##added by ys
#         # self.cls_head_src = nn.Linear(128, 10)
#
#     def forward(self, x, mode='test'):
#
#         in_size = x.size(0)
#         out1 = self.mp(self.relu1(self.conv1(x)))
#         out2 = self.mp(self.relu2(self.conv2(out1)))
#         out2 = out2.view(in_size, -1)
#         out3 = self.relu3(self.fc1(out2))
#         out4 = self.relu4(self.fc2(out3))
#
#         if mode == 'test':
#             # m = self.pro_head(out4)
#             # # p = self.cls_head_src(self.relu5(m))
#             # p = self.cls_head_src(m)
#             p = self.cls_head_src(out4)
#             return p
#         elif mode == 'train':
#
#             p = self.cls_head_src(out4)
#             # m = self.pro_head(out4)
#             # # p = self.cls_head_src(self.relu5(m))
#             # p = self.cls_head_src(m)
#             z = self.pro_head(out4)
#             z = F.normalize(z)
#
#             omega = torch.randn(in_size, 1024).cuda()
#             phi = -2*math.pi*torch.rand(in_size, 1024) + 2*math.pi
#             phi = phi.cuda()
#             fout = math.sqrt(2) * torch.cos(omega*out4+phi)
#
#
#             # z = F.normalize(m)
#             return p,z,fout
#         elif mode == 'p_f':
#             p = self.cls_head_src(out4)
#             # m = self.pro_head(out4)
#             # # p = self.cls_head_src(self.relu5(m))
#             # p = self.cls_head_src(m)
#             return p, out4
#         #elif mode == 'target':
#         #    p = self.cls_head_tgt(out4)
#         #    z = self.pro_head(out4)
#         #    z = F.normalize(z)
#         #    return p,z
#
# class ConvNetVis(nn.Module):
#     ''' 方便可视化，特征提取器输出2-d特征
#     '''
#     def __init__(self, imdim=3):
#         super(ConvNetVis, self).__init__()
#
#         self.conv1 = nn.Conv2d(imdim, 64, kernel_size=5, stride=1, padding=0)
#         self.mp = nn.MaxPool2d(2)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.fc1 = nn.Linear(128*5*5, 1024)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(1024, 2)
#         self.relu4 = nn.ReLU(inplace=True)
#
#         self.cls_head_src = nn.Linear(2, 10)
#         self.cls_head_tgt = nn.Linear(2, 10)
#         self.pro_head = nn.Linear(2, 128)
#
#     def forward(self, x, mode='test'):
#
#         in_size = x.size(0)
#         out1 = self.mp(self.relu1(self.conv1(x)))
#         out2 = self.mp(self.relu2(self.conv2(out1)))
#         out2 = out2.view(in_size, -1)
#         out3 = self.relu3(self.fc1(out2))
#         out4 = self.relu4(self.fc2(out3))
#
#         if mode == 'test':
#             p = self.cls_head_src(out4)
#             return p
#         elif mode == 'train':
#             p = self.cls_head_src(out4)
#             z = self.pro_head(out4)
#             z = F.normalize(z)
#             return p,z
#         elif mode == 'p_f':
#             p = self.cls_head_src(out4)
#             return p, out4
#         #elif mode == 'target':
#         #    p = self.cls_head_tgt(out4)
#         #    z = self.pro_head(out4)
#         #    z = F.normalize(z)
#         #    return p,z
#
#
import random
import torch
import torch.nn as nn
import torch.nn.functional as F



class MixStyle(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix

class ConvNet(nn.Module):
    ''' 网络结构和cvpr2020的 M-ADA 方法一致 '''

    def __init__(self, imdim=3):
        super(ConvNet, self).__init__()
        self.eps = 0.1
        self.conv1 = nn.Conv2d(imdim, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu4 = nn.ReLU()
        self.mix = MixStyle(p=0.5, alpha=0.1)
        self.cls_head_src = nn.Linear(1024, 10)
        # self.cls_head_tgt = nn.Linear(1024, 10)
        self.pro_head = nn.Linear(1024, 128)
        # self.relu5 = nn.ReLU(inplace=True)
        # self.pro_head2 = nn.Linear(1024, 128)
        # self.relu5 = nn.ReLU(inplace=True) ##added by ys
        # self.fc_inde = nn.Linear(1024, 500)
        # self.relu5 = nn.ReLU(inplace=True)
        # self.cls_head_src2 = nn.Linear(1024, 128)
        self.feats = None
        self.feats2 = None
        self.pro_head3 = nn.Linear(10, 5)

    def clip(self, img, img_min=None, img_max=None):
        if img_min is None:
            img_min = torch.tensor([-2.1179, -2.0357, -1.8044]).view(1, 3, 1, 1).cuda()

        if img_max is None:
            img_max = torch.tensor([2.2489, 2.4286, 2.6400]).view(3, 1, 1).cuda()

        img = torch.clip(img, min=img_min, max=img_max)

        return img

    def grad_norm(self, grad):
        grad = F.normalize(grad, p=2, dim=1)
        grad = grad.pow(2)  # optional
        grad = grad * self.eps
        return grad


    def forward(self, x, shm=False, mode='test'):
        end_points = {}
        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        self.feats = out2.detach()
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))

        out4 = self.relu4(self.fc2(out3))
        end_points['Embedding'] = out4
        end_points['cls'] = self.cls_head_src
        # self.feats2 = out4.detach()
        if mode == 'test':


            p = self.cls_head_src(out4)

            return p
        elif mode == 'train':

            # p = self.cls_head_src(rec)
            p = self.cls_head_src(out4)
            # m = self.pro_head(out4)
            # # p = self.cls_head_src(self.relu5(m))
            # p = self.cls_head_src(m)
            z = self.pro_head(out4)
            # z = self.pro_head(rec)
            # z = self.pro_head2(self.relu5(z))
            z = F.normalize(z)
            # self.relu6 = nn.ReLU(inplace=True)
            # z2 = self.pro_head2(rec)
            # z2 = self.pro_head2(out4)
            # # z2 = p2
            # z2 = F.normalize(z2)

            z3 = F.softmax(p,dim=-1)
            # # # z3 = self.relu6(p)
            z3 = self.pro_head3(z3)
            # #
            z3 = F.normalize(z3)
            # z3 = p
            # print(z3.shape)
            # z3 = torch.transpose(z3,1,0)
            # print(z3.shape)
            # return p, z, z2, end_points
            return p, z, z3, end_points
        elif mode == 'p_f':
            p = self.cls_head_src(out4)

            return p, out4
        # elif mode == 'target':
        #    p = self.cls_head_tgt(out4)
        #    z = self.pro_head(out4)
        #    z = F.normalize(z)
        #    return p,z


class ConvNetVis(nn.Module):
    ''' 方便可视化，特征提取器输出2-d特征
    '''

    def __init__(self, imdim=3):
        super(ConvNetVis, self).__init__()

        self.conv1 = nn.Conv2d(imdim, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 2)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(2, 10)
        self.cls_head_tgt = nn.Linear(2, 10)
        self.pro_head = nn.Linear(2, 128)

    def forward(self, x, mode='test'):

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            p = self.cls_head_src(out4)
            return p
        elif mode == 'train':
            p = self.cls_head_src(out4)
            z = self.pro_head(out4)
            z = F.normalize(z)
            return p, z
        elif mode == 'p_f':
            p = self.cls_head_src(out4)
            return p, out4
        # elif mode == 'target':
        #    p = self.cls_head_tgt(out4)
        #    z = self.pro_head(out4)
        #    z = F.normalize(z)
        #    return p,z


