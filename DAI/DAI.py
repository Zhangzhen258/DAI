import math
from loss import Lim
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
# from ops.config import parser
from torch import autograd
# import matplotlib.pyplot as plt
import os
import click
import time
import numpy as np
import pandas as pd

from con_losses import SupConLoss

from network import mnist_net, generator
import data_loader
from main_base import evaluate
from torch.autograd import Variable

HOME = os.environ['HOME']


@click.command()
@click.option('--gpu', type=str, default='1', help='选择gpu')
@click.option('--data', type=str, default='mnist', help='数据集名称')
@click.option('--ntr', type=int, default=None, help='选择训练集前ntr个样本')
@click.option('--gen', type=str, default='cnn', help='cnn/hr')
@click.option('--gen_mode', type=str, default=None, help='生成器模式')
@click.option('--n_tgt', type=int, default=10, help='学习多少了tgt模型')
@click.option('--tgt_epochs', type=int, default=10, help='每个目标域训练多少了epochs')
@click.option('--tgt_epochs_fixg', type=int, default=None, help='当epoch大于该值，将G fix掉')
@click.option('--nbatch', type=int, default=None, help='每个epoch中包含多少了batch')
@click.option('--batchsize', type=int, default=256)
@click.option('--lr', type=float, default=1e-3)
@click.option('--lr_scheduler', type=str, default='none', help='是否选择学习率衰减策略')
@click.option('--svroot', type=str, default='./saved')
@click.option('--ckpt', type=str, default='./saved/best.pkl')
@click.option('--w_cls', type=float, default=1.0, help='cls项权重')
@click.option('--w_info', type=float, default=1.0, help='infomin项权重')
@click.option('--w_cyc', type=float, default=10.0, help='cycleloss项权重')
@click.option('--w_tgt', type=float, default=1.0, help='entropy_desc项权重')
@click.option('--w_div', type=float, default=1.0, help='多形性loss权重')
@click.option('--div_thresh', type=float, default=0.1, help='div_loss 阈值')
# @click.option('--w_tgt', type=float, default=1.0, help='target domain样本更新 tasknet 的强度控制')
@click.option('--interpolation', type=str, default='img', help='在源域和生成域之间插值得到新的域，两种方式：img/pixel')
def experiment(gpu, data, ntr, gen, gen_mode, \
               n_tgt, tgt_epochs, tgt_epochs_fixg, nbatch, batchsize, lr, lr_scheduler, svroot, ckpt, \
               w_cls, w_info, w_cyc, w_div, div_thresh, w_tgt, interpolation):
    settings = locals().copy()
    print(settings)
    # 全局设置
    zdim = 10
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    g1root = os.path.join(svroot, 'g1')
    if not os.path.exists(g1root):
        os.makedirs(g1root)
    writer = SummaryWriter(svroot)
    image_size = (32, 32)
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize([0.1307] * 3, [0.3081] * 3)
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((32, 32), (0.8, 1.0)),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])

    test_transform = preprocess

    # 加载数据集
    imdim = 3  # 默认3通道
    if data in ['mnist', 'mnist_t', 'mnistvis']:
        if data in ['mnist', 'mnistvis']:
            trset = data_loader.load_mnist('train', ntr=ntr, translate=train_transform)
            teset = data_loader.load_mnist('test', ntr=ntr, translate=test_transform)
        elif data == 'mnist_t':
            trset = data_loader.load_mnist_t('train', ntr=ntr)
            teset = data_loader.load_mnist('test')
        imsize = [32, 32]

    trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                          sampler=RandomSampler(trset, True, nbatch * batchsize))

    teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)

    # 加载模型
    def get_generator(name):
        if name == 'cnn':
            g1_net = generator.cnnGenerator(imdim=imdim, imsize=imsize).cuda()
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr)
        return g1_net, g1_opt


    def get_amplitude_phase(x):
        t_tranform_x = torch.fft.rfftn(x, dim=(-2, -1), norm ="ortho")
        amplitude_x = torch.abs(t_tranform_x)
        phase_x = torch.angle(t_tranform_x)
        return amplitude_x, phase_x


    g1_list = []

    if data in ['mnist', 'mnist_t']:
        src_net = mnist_net.ConvNet().cuda()
        src_opt = optim.Adam(src_net.parameters(), lr=lr)
        saved_weight = torch.load(ckpt)
        src_net.load_state_dict(saved_weight['cls_net'])

    elif data == 'mnistvis':
        src_net = mnist_net.ConvNetVis().cuda()
        saved_weight = torch.load(ckpt)
        src_net.load_state_dict(saved_weight['cls_net'])
        src_opt = optim.Adam(src_net.parameters(), lr=lr)

    cls_criterion = nn.CrossEntropyLoss()

    con_criterion = SupConLoss()


    kldiv = nn.KLDivLoss(reduction='batchmean')
    criterion_im = Lim(1e-5)
    # 开始训练
    global_best_acc = 0
    columns_best = ['mnist_m', 'usps', 'svhn', 'syndigit', 'ave', 'best']
    df_best = pd.DataFrame()
    best = 0


    for i_tgt in range(n_tgt):
        print(f'target domain {i_tgt}')
        g1_net, g1_opt = get_generator(gen)
        best_acc = 0

        for epoch in range(tgt_epochs):
            t1 = time.time()
            dummy_w = nn.Parameter(torch.Tensor([1.0])).cuda()
            flag_fixG = False
            if (tgt_epochs_fixg is not None) and (epoch >= tgt_epochs_fixg):
                flag_fixG = True
            loss_list = []
            time_list = []
            # src_net.train()
            src_net.eval()
            for i, (x, y) in enumerate(trloader):
                lable_num = np.unique(y)
                number = y.shape[0]
                x, y = x.cuda(), y.cuda()

                if len(g1_list) > 0:
                    idx = np.random.randint(0, len(g1_list))
                    if gen in ['hr', 'cnn']:
                        with torch.no_grad():
                            x2_src = g1_list[idx](x, rand=True)
                        if interpolation == 'img':
                            rand = torch.rand(len(x), 1, 1, 1).cuda()
                            x3_mix = rand * x + (1 - rand) * x2_src

                # 合成新数据
                if gen in ['cnn', 'hr']:
                    x_tgt = g1_net(x, rand=True)
                    x2_tgt = g1_net(x, rand=True)

                p1_src, z1_src, z1_src_2, tuple_x = src_net(x, mode='train')
                p_list = []
                if len(g1_list) > 0:  # 如果生成器
                    p2_src, z2_src, z2_src_2, tuple_x2 = src_net(x2_src, mode='train')
                    p_list.append(p2_src)
                    p3_mix, z3_mix, z3_mix_2, tuple_x3 = src_net(x3_mix, mode='train')
                    zsrc = torch.cat([z1_src.unsqueeze(1), z2_src.unsqueeze(1), z3_mix.unsqueeze(1)], dim=1)
                    zsrc_2 = torch.cat([z1_src_2.unsqueeze(1), z2_src_2.unsqueeze(1), z3_mix_2.unsqueeze(1)], dim=1)
                    src_cls_loss = cls_criterion(p1_src, y) + cls_criterion(p2_src, y) + cls_criterion(p3_mix, y)
                else:
                    label_loss = 0
                    zsrc = z1_src.unsqueeze(1)
                    zsrc_2 = z1_src_2.unsqueeze(1)
                    src_cls_loss = cls_criterion(p1_src, y)

                p_tgt, z_tgt, z_tgt_2, tuple_xt = src_net(x_tgt, mode='train')

                p_tgt2, z_tgt2, z_tgt2_2, tuple_xt2 = src_net(x2_tgt, mode='train')

                tgt_cls_loss = cls_criterion(p_tgt, y)

                src = tuple_x['Embedding']
                tar = tuple_xt['Embedding']
                tar2 = tuple_xt2['Embedding']


                zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1)
                zall_cont = torch.cat([z_tgt_2.unsqueeze(1), zsrc_2], dim=1)

                con_loss = (con_criterion(zall, adv=False) + con_criterion(zall_cont, adv=False)) / 2

                loss = src_cls_loss + w_tgt * con_loss + w_tgt * tgt_cls_loss

                src_opt.zero_grad()
                if flag_fixG:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)

                if flag_fixG:
                    con_loss_adv = torch.tensor(0)
                    div_loss = torch.tensor(0)
                    cyc_loss = torch.tensor(0)
                else:

                    zsrc_0 = torch.cat([z_tgt.unsqueeze(1), z1_src.unsqueeze(1).detach()], dim=1)
                    p_tgt, z_tgt, z_tgt_2, tuple_xt = src_net(x_tgt, mode='train')
                    p_tgt2, _, _, tuple_xt2 = src_net(x2_tgt, mode='train')
                    tgt_cls_loss = cls_criterion(p_tgt, y)


                    idx = np.random.randint(0, zsrc.size(1))
                    con_loss_adv = torch.FloatTensor([0]).cuda()
                    zall = torch.cat([z_tgt.unsqueeze(1), zsrc[:, idx:idx + 1].detach()], dim=1)
                    con_loss_adv = con_criterion(zall, adv=True)


                    if gen in ['cnn', 'hr']:
                        div_loss = (x_tgt - x2_tgt).abs().mean([1, 2, 3]).clamp(max=div_thresh).mean()

                        src = tuple_x['Embedding']
                        tar = tuple_xt['Embedding']
                        tar2 = tuple_xt2['Embedding']


                        amplitude_x, phase_x = get_amplitude_phase(x)
                        amplitude_x_rec, phase_x_rec = get_amplitude_phase(x_tgt)

                        complex_tensor = amplitude_x_rec * torch.exp(1j * phase_x_rec)
                        reconstructed_x_tgt = torch.fft.ifft2(complex_tensor, dim=(-2, -1), norm = "ortho").real

                        complex_tensor_x = amplitude_x_rec * torch.exp(1j * phase_x)
                        reconstructed_x = torch.fft.ifft2(complex_tensor_x, dim=(-2, -1), norm = "ortho").real

                        cyc_loss = F.mse_loss(reconstructed_x, reconstructed_x_tgt)

                        div_loss_phase = 0
                    loss = tgt_cls_loss + w_info * con_loss_adv - w_div * div_loss + cyc_loss

                    g1_opt.zero_grad()
                    loss.backward()
                    g1_opt.step()


                src_opt.step()

                loss_list.append(
                    [src_cls_loss.item(), tgt_cls_loss.item(), con_loss.item(), con_loss_adv.item(), div_loss.item(),
                     cyc_loss.item()])
            src_cls_loss, tgt_cls_loss, con_loss, con_loss_adv, div_loss, cyc_loss = np.mean(loss_list, 0)

            # 测试
            src_net.eval()
            if data in ['mnist', 'mnist_t', 'mnistvis']:
                teacc = evaluate(src_net, teloader)

            if best_acc <= teacc:
                best_acc = teacc
                torch.save({'cls_net': src_net.state_dict()}, os.path.join(svroot, f'{i_tgt}-best.pkl'))

            t2 = time.time()

            # 保存日志
            print(
                f'epoch {epoch}, time {t2 - t1:.2f}, src_cls {src_cls_loss:.4f} tgt_cls {tgt_cls_loss:.4f} con {con_loss:.4f} con_adv {con_loss_adv:.4f} div_loss {div_loss:.4f}  cyc_loss {cyc_loss:.4f} /// teacc {teacc:2.2f}')


            g1_all = g1_list + [g1_net]

        # 保存训练好的G1
        torch.save({'g1': g1_net.state_dict()}, os.path.join(g1root, f'{i_tgt}.pkl'))
        g1_list.append(g1_net)
        from main_test_digit import evaluate_digit
        if data == 'mnist':
            pklpath = f'{svroot}/{i_tgt}-best.pkl'
            df_best, best = evaluate_digit(gpu, pklpath, pklpath + '.test', df_best, best)
            print(df_best)



    writer.close()

if __name__ == '__main__':
    experiment()