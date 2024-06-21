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



from network import mnist_net
import data_loader
from main_base import evaluate



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
@click.option('--w_info', type=float, default=1.0, help='infomin项权重')
@click.option('--w_desc', type=float, default=1.0, help='entropy_desc项权重')
@click.option('--w_div', type=float, default=1.0, help='多形性loss权重')
@click.option('--div_thresh', type=float, default=0.1, help='div_loss 阈值')
@click.option('--interpolation', type=str, default='img', help='在源域和生成域之间插值得到新的域，两种方式：img/pixel')

def experiment(gpu, data, ntr, gen, gen_mode, \
               n_tgt, tgt_epochs, tgt_epochs_fixg, nbatch, batchsize, lr, lr_scheduler, svroot, ckpt, \
               w_info, w_div, w_desc, div_thresh, interpolation):
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

    # 加载数据集
    imdim = 3  # 默认3通道
    if data in ['mnist', 'mnist_t', 'mnistvis']:
        if data in ['mnist', 'mnistvis']:
            trset = data_loader.load_mnist('train', ntr=ntr)
            teset = data_loader.load_mnist('test')

        elif data == 'mnist_t':
            trset = data_loader.load_mnist_t('train', ntr=ntr)
            teset = data_loader.load_mnist('test')
        imsize = [32, 32]

    trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                          sampler=RandomSampler(trset, True, nbatch * batchsize))
    trloader2 = DataLoader(trset, batch_size=1, num_workers=8, \
                          sampler=RandomSampler(trset, True, nbatch * 1))
    # trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
    #                       sampler=RandomSampler(trset, True, nbatch * batchsize))

    teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)






    def ACE(featureidx, numsamples, classifier, feature, targetm, outdim):
        bs, zdim = feature.shape
        zdo = torch.randn(numsamples, bs, zdim).cuda()
        zdo[:, :, featureidx] = feature[:, featureidx]
        sample = classifier(zdo.view(numsamples * bs, zdim))
        ACEdo = sample.view(numsamples, bs, -1).mean(0)

        zrand = torch.randn(numsamples, bs, zdim).cuda()
        sample = classifier(zrand.view(numsamples * bs, zdim))
        ACEbaseline = sample.view(numsamples, bs, -1).mean(0)
        ace = ACEbaseline - ACEdo
        return (ace)

    def contrastive_ace(numsamples, classifier, feature, targetm, outdim, anchorbs):
        numfeature = feature.shape[1]
        ace = []
        for i in range(numfeature):
            ace.append(ACE(i, numsamples, classifier, feature, targetm, outdim))
        acematrix = torch.stack(ace, dim=1) / (
                    torch.stack(ace, dim=1).norm(dim=1).unsqueeze(1) + 1e-8)  # [bs, num_feature]
        acematrix_test = acematrix.argmax(dim=-1)
        anchor = acematrix[:anchorbs] / acematrix[:anchorbs].norm(1)
        neighbor = acematrix[anchorbs:2 * anchorbs] / acematrix[anchorbs:2 * anchorbs].norm(1)
        distant = acematrix[2 * anchorbs:] / acematrix[2 * anchorbs:].norm(1)
        margin = 0.02
        pos = (torch.abs(anchor - neighbor)).sum()
        neg = (torch.abs(anchor - distant)).sum()
        contrastive_loss = F.relu(pos - neg + margin)
        return contrastive_loss


    g1_list = []

    if data in ['mnist', 'mnist_t']:
        src_net = mnist_net.ConvNet().cuda()
        src_opt = optim.Adam(src_net.parameters(), lr=lr)
        saved_weight = torch.load('saved-digit/base_run0/best.pkl')
        src_net.load_state_dict(saved_weight['cls_net'])

    cls_criterion = nn.CrossEntropyLoss()



    con_criterion = SupConLoss()


    global_best_acc = 0
    columns_best = ['mnist_m', 'usps', 'svhn', 'syndigit', 'ave', 'best']
    df_best = pd.DataFrame()
    best = 0
    best_acc = 0

    class_specific_train_dataset = []
    for i in range(10):
        class_specific_train_dataset.append([])
    print('          num classes :     {}'.format(10))

    for i, (x, y) in enumerate(trloader2):
        data = x
        target = int(y)
        class_specific_train_dataset[target].append([data, target])



    if lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(src_opt, tgt_epochs * 10)

    for epoch in range(30):
        t1 = time.time()
        loss_list = []
        time_list = []
        src_net.train()
        for i, (x, y) in enumerate(trloader):
            lable_num = np.unique(y)
            number = y.shape[0]
            x, y = x.cuda(), y.cuda()
            anchor_label = y
            neighbour_list = []
            distant_list = []
            label_dist_list = []

            bs = int(x.shape[0])

            for ibatch in range(bs):
                neighbour_label = anchor_label[ibatch]
                neighbour_idx = np.random.randint(len(class_specific_train_dataset[neighbour_label]), size=1)[0]
                neighbour = class_specific_train_dataset[neighbour_label][neighbour_idx][0]
                distant_label = np.random.randint(10, size=1)[0]
                while distant_label == anchor_label[ibatch]:
                    distant_label = np.random.randint(10, size=1)[0]
                distant_idx = np.random.randint(len(class_specific_train_dataset[distant_label]), size=1)[0]
                distant = class_specific_train_dataset[distant_label][distant_idx][0]

                label_dist_list.append(distant_label)
                neighbour_list.append(neighbour)
                distant_list.append(distant)

            neighbour_batch = torch.stack(neighbour_list, 0).cuda()
            distant_batch = torch.stack(distant_list, 0).cuda()
            label_dist_list = torch.tensor(label_dist_list).cuda()

            triplet_minibatches_device = []
            for i in range(bs):
                triplet_minibatches_device.append(
                    [x[i].cuda(), torch.tensor([anchor_label[i]]).cuda()])
            for i in range(bs):
                triplet_minibatches_device.append(
                    [neighbour_batch[i].cuda(), torch.tensor([anchor_label[i]]).cuda()])
            for i in range(bs):
                triplet_minibatches_device.append(
                    [distant_batch[i].cuda(), torch.tensor([label_dist_list[i]]).cuda()])
            data_list = [xi for xi, _ in triplet_minibatches_device]
            target_list = [yi for _, yi in triplet_minibatches_device]

            data_list = [torch.squeeze(tensor, dim=0) if tensor.size(0) == 1 else tensor for tensor in data_list]
            data = torch.stack(data_list, 0)


            target = torch.tensor(target_list).cuda()
            bs = int(data.shape[0] / 3)

            tripleoutput, _, _, tuple = src_net(data, mode='train')


            triplefeatures = tuple['Embedding']
            classifier = tuple['cls']
            objective = F.cross_entropy(tripleoutput, target)

            penalty = contrastive_ace(2, classifier, triplefeatures, target, 10, bs)

            src_opt.zero_grad()
            loss = objective + (1 * penalty)

            loss.backward()
            src_opt.step()

        # teacc = 0
        # 测试
        src_net.eval()

        teacc = evaluate(src_net, teloader)
        print(teacc)

        torch.save({'cls_net': src_net.state_dict()}, os.path.join(svroot, f'best.pkl'))

        t2 = time.time()

        # 保存日志
        print(
            f'time {t2 - t1:2.2f} teacc {teacc:2.2f}')


    # 测试 i_tgt 模型的泛化效果
        from main_test_digit import evaluate_digit
    # 保存最好的i_tgt模型泛化效果
        pklpath = f'{svroot}/best.pkl'
        df_best, best = evaluate_digit(gpu, pklpath, pklpath + '.test', df_best, best)
        print(df_best)

    writer.close()

if __name__ == '__main__':
    experiment()
