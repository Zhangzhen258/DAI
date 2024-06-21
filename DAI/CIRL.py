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
from tools.classifier import Masker
import pandas as pd
from tools.CIRL_tools import *


from network import mnist_net
import data_loader

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
# @click.option('--w_cls', type=float, default=1.0, help='cls项权重')
@click.option('--w_info', type=float, default=1.0, help='infomin项权重')
# @click.option('--w_cyc', type=float, default=10.0, help='cycleloss项权重')
@click.option('--w_desc', type=float, default=1.0, help='entropy_desc项权重')
@click.option('--w_div', type=float, default=1.0, help='多形性loss权重')
@click.option('--div_thresh', type=float, default=0.1, help='div_loss 阈值')
# @click.option('--w_tgt', type=float, default=1.0, help='target domain样本更新 tasknet 的强度控制')
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
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((32, 32), (0.8, 1.0)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = preprocess


    # 加载数据集
    imdim = 3  # 默认3通道
    if data in ['mnist', 'mnist_t', 'mnistvis']:
        if data in ['mnist', 'mnistvis']:
            trset = data_loader.load_mnist('train', ntr=ntr)
            teset = data_loader.load_mnist('test')

            # trset = data_loader.load_mnist('train', ntr=ntr, translate=train_transform)
            # teset = data_loader.load_mnist('test', ntr=ntr, translate=test_transform)

        elif data == 'mnist_t':
            trset = data_loader.load_mnist_t('train', ntr=ntr)
            teset = data_loader.load_mnist('test')
        imsize = [32, 32]

    trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                          sampler=RandomSampler(trset, True, nbatch * batchsize))
    # trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
    #                       sampler=RandomSampler(trset, True, nbatch * batchsize))

    teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)

    def do_eval(encoder, classifier, loader):
        correct, count = 0, 0
        ps = []
        ys = []
        encoder.eval()
        classifier.eval()
        for i, (x1, y1) in enumerate(loader):
            with torch.no_grad():
                x1 = x1.cuda()
                _, _, _, tuplex = encoder(x1,mode='train')
                features = tuplex['Embedding']
                p1 = classifier(features)
                p1 = p1.argmax(dim=1)
                ps.append(p1.detach().cpu().numpy())
                ys.append(y1.numpy())
        ps = np.concatenate(ps)
        ys = np.concatenate(ys)
        acc = np.mean(ys == ps) * 100
        return acc

    def evaluate_digit_CIRL(encoder, classifier, df_best, best, channels=3):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        encoder.eval()
        classifier.eval()
        # 测试
        str2fun = {
            'mnist': data_loader.load_mnist,
            'mnist_m': data_loader.load_mnist_m,
            'usps': data_loader.load_usps,
            'svhn': data_loader.load_svhn,
            'syndigit': data_loader.load_syndigit,
        }
        columns = ['mnist_m', 'usps', 'svhn', 'syndigit']
        columns_ave = ['mnist_m', 'usps', 'svhn', 'syndigit', 'ave']
        columns_best = ['mnist_m', 'usps', 'svhn', 'syndigit', 'ave', 'best']
        rst = []
        for data in columns:
            teset = str2fun[data]('test', channels=channels)
            teloader = DataLoader(teset, batch_size=128, num_workers=8)
            teacc = do_eval(encoder,classifier, teloader)
            rst.append(teacc)
        ave = np.array(rst).mean()
        rst.append(ave)
        df = pd.DataFrame([rst], columns=columns_ave)
        if best < ave:
            best = ave
            rst.append(0)
            df_best = pd.DataFrame([rst], columns=columns_best)
        print(df)
        return df_best, best




    g1_list = []

    if data in ['mnist', 'mnist_t']:
        encoder = mnist_net.ConvNet().cuda()
        saved_weight = torch.load('saved-digit/base_run0/best.pkl')
        encoder.load_state_dict(saved_weight['cls_net'])

        classifier = nn.Linear(1024, 10).cuda()
        classifier_ad = nn.Linear(1024, 10).cuda()
        dim=1024
        # masker = Masker(in_dim=dim, num_classes=dim, middle=4 * dim, k=self.config["k"]).to(device)
        masker = Masker(in_dim=dim, num_classes=dim, middle=4 * dim, k=1000).cuda()

        encoder_optim = optim.Adam(encoder.parameters(), lr=lr)
        classifier_optim = optim.Adam(classifier.parameters(), lr=lr)
        classifier_ad_optim = optim.Adam(classifier_ad.parameters(), lr=lr)
        masker_optim = optim.Adam(masker.parameters(), lr=lr)

    # 开始训练
    global_best_acc = 0
    columns_best = ['mnist_m', 'usps', 'svhn', 'syndigit', 'ave', 'best']
    df_best = pd.DataFrame()
    best = 0
    best_acc = 0

    for epoch in range(50):
        t1 = time.time()
        encoder.train()
        classifier.train()
        classifier_ad.train()
        masker.train()
        criterion = nn.CrossEntropyLoss()
        for i, (x, y) in enumerate(trloader):
            x, y = x.cuda(), y.cuda()
            encoder_optim.zero_grad()
            classifier_optim.zero_grad()
            classifier_ad_optim.zero_grad()
            masker_optim.zero_grad()
            loss_dict = {}
            correct_dict = {}
            num_samples_dict = {}
            total_loss = 0.0

            _, _, _, tuple_x = encoder(x, mode='train')
            features = tuple_x['Embedding']
            masks_sup = masker(features.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            if epoch <= 5:
                masks_sup = torch.ones_like(features.clone().detach())
                masks_inf = torch.ones_like(features.detach())
            features_sup = features * masks_sup
            features_inf = features * masks_inf
            scores_sup = classifier(features_sup)
            scores_inf = classifier_ad(features_inf)

            assert x.size(0) % 2 == 0
            split_idx = int(x.size(0) / 2)
            features_ori, features_aug = torch.split(features, split_idx)
            assert features_ori.size(0) == features_aug.size(0)

            loss_cls_sup = criterion(scores_sup, y)
            loss_dict["sup"] = loss_cls_sup.item()
            correct_dict["sup"] = calculate_correct(scores_sup, y)
            num_samples_dict["sup"] = int(scores_sup.size(0))

            # classification loss for inf feature
            loss_cls_inf = criterion(scores_inf, y)
            loss_dict["inf"] = loss_cls_inf.item()
            correct_dict["inf"] = calculate_correct(scores_inf, y)
            num_samples_dict["inf"] = int(scores_inf.size(0))

            # factorization loss for features between ori and aug
            loss_fac = factorization_loss(features_ori, features_aug)
            loss_dict["fac"] = loss_fac.item()

            const_weight = get_current_consistency_weight(epoch=epoch,
                                                          weight=5.0,
                                                          rampup_length=5,
                                                          rampup_type="sigmoid")

            total_loss = 0.5 * loss_cls_sup + 0.5 * loss_cls_inf + const_weight * loss_fac
            loss_dict["total"] = total_loss.item()

            # backward
            total_loss.backward()

            # update
            encoder_optim.step()
            classifier_optim.step()
            classifier_ad_optim.step()

            ## ---------------------------------- step2: update masker------------------------------
            masker_optim.zero_grad()
            _, _, _, tuple_x = encoder(x, mode='train')
            features = tuple_x['Embedding']
            masks_sup = masker(features.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            features_sup = features * masks_sup
            features_inf = features * masks_inf
            scores_sup = classifier(features_sup)
            scores_inf = classifier_ad(features_inf)

            loss_cls_sup = criterion(scores_sup, y)
            loss_cls_inf = criterion(scores_inf, y)
            total_loss = 0.5 * loss_cls_sup - 0.5 * loss_cls_inf
            total_loss.backward()
            masker_optim.step()

        encoder.eval()
        classifier.eval()
        masker.eval()
        classifier_ad.eval()


        teacc = do_eval(encoder, classifier, teloader)
        t2 = time.time()
        # 保存日志
        print(f'time {t2 - t1:2.2f} teacc {teacc:2.2f} ')
        df_best, best = evaluate_digit_CIRL(encoder, classifier, df_best, best)
        print(df_best)

    writer.close()

if __name__ == '__main__':
    experiment()
