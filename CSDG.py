import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import os
import click
import time
import numpy as np
import pandas as pd
from con_losses import SupConLoss
from network import GIN
import data_loader
from main_base import evaluate
from network import mnist_net


HOME = os.environ['HOME']
@click.command()
@click.option('--gpu', type=str, default='1', help='选择gpu')
@click.option('--data', type=str, default='PACS', help='数据集名称')
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
@click.option('--w_div', type=float, default=1.0, help='多形性loss权重')
@click.option('--div_thresh', type=float, default=0.1, help='div_loss 阈值')
@click.option('--w_tgt', type=float, default=1.0, help='target domain样本更新 tasknet 的强度控制')
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
            g1_net = GIN.cnnGenerator(imdim=imdim, imsize=imsize).cuda()
        return g1_net

    def generate_pseudo_feature_map(x):
        batch_size, channels, height, width = x.size()
        control_points = torch.rand(batch_size, channels, height // 4, width // 4, device=x.device)
        grid_x = torch.linspace(0, 1, steps=width, device=x.device) * 2 - 1
        grid_y = torch.linspace(0, 1, steps=height, device=x.device) * 2 - 1
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y)
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, height * width, 2)
        grid = grid.expand(batch_size, 1, height * width, 2)
        pseudo_feature_map = F.grid_sample(control_points, grid, mode='bilinear', align_corners=True)
        return pseudo_feature_map.reshape(batch_size, channels, height, width)

    g1_list = []
    if data in ['mnist', 'mnist_t']:
        src_net = mnist_net.ConvNet().cuda()
        src_opt = optim.Adam(src_net.parameters(), lr=lr)
        saved_weight = torch.load(ckpt)
        src_net.load_state_dict(saved_weight['cls_net'])


    cls_criterion = nn.CrossEntropyLoss()

    con_criterion = SupConLoss()
    dist_fn = torch.nn.MSELoss()
    kldiv = nn.KLDivLoss(reduction='batchmean')
    # 开始训练
    global_best_acc = 0
    columns_best = ['photo', 'art_painting', 'sketch', 'ave', 'best']
    df_best = pd.DataFrame()
    best = 0
    dummy_w = torch.nn.Parameter(torch.tensor([1.])).cuda()

    g1_net = get_generator(gen)
    best_acc = 0

    for epoch in range(100):
        t1 = time.time()
        loss_list = []
        src_net.train()
        for i, (x, y) in enumerate(trloader):
            x, y = x.cuda(), y.cuda()
            if gen in ['cnn', 'hr']:
                x_tgt = g1_net(x)
                x2_tgt = g1_net(x)

            p1_src, z1_src, z1_src_2, tuple_x = src_net(x, mode='train')

            src_cls_loss = cls_criterion(p1_src, y)


            src = tuple_x['Embedding']


            map = generate_pseudo_feature_map(x)
            x_tgt_new = x_tgt * map + x2_tgt * (1 - map)
            x_tgt2_new = x2_tgt * map + x_tgt * (1 - map)

            p_tgt, z_tgt, z_tgt_2, tuple_xt = src_net(x_tgt_new, mode='train')
            p_tgt2, z_tgt2, z_tgt_2_2, tuple_xt2 = src_net(x_tgt2_new, mode='train')

            tgt_cls_loss = cls_criterion(p_tgt, y) + cls_criterion(p_tgt2, y)

            pt1 = F.softmax(p_tgt, dim=1)
            pt2 = F.softmax(p_tgt2, dim=1)
            p_mixture = torch.clamp((pt1 + pt2) / 2., 1e-7, 1).log()
            kl_loss = 1.0 * (
                    F.kl_div(p_mixture, pt1 + 1e-7, reduction='batchmean') +
                    F.kl_div(p_mixture, pt2 + 1e-7, reduction='batchmean')
            ) / 2.


            loss = src_cls_loss + tgt_cls_loss + 1 * kl_loss


            src_opt.zero_grad()

            loss.backward()

            src_opt.step()


            if lr_scheduler in ['cosine']:
                scheduler.step()



        src_net.eval()

        if data in ['mnist', 'mnist_t', 'mnistvis','PACS']:
            teacc = evaluate(src_net, teloader)
        if best_acc < teacc:
            best_acc = teacc
            torch.save({'cls_net': src_net.state_dict()}, os.path.join(svroot, f'{i_tgt}-best.pkl'))

        t2 = time.time()

        # 保存日志
        print(
            f'epoch {epoch}, time {t2 - t1:.2f},   src_cls {src_cls_loss:.4f} tgt_cls {tgt_cls_loss:.4f}  /// teacc {teacc:2.2f}')

        g1_all = g1_list + [g1_net]

        # 保存训练好的G1
        torch.save({'g1': g1_net.state_dict()}, os.path.join(g1root, f'{i_tgt}.pkl'))
        g1_list.append(g1_net)

        from main_test_digit import evaluate_digit
        # 保存最好的i_tgt模型泛化效果
        if data == 'mnist':
            pklpath = f'{svroot}/{i_tgt}-best.pkl'
            df_best, best = evaluate_digit(gpu, pklpath, pklpath + '.test', df_best, best)
            print(df_best)
    writer.close()

if __name__ == '__main__':
    experiment()
