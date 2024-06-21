import torch
import torch.nn as nn
import torch.nn.functional as F
class AdvStylemodule(nn.Module):
    '''
    input: source
    images
    gt: ground - truth
    labels
    net: segmentation
    network
    optim: optimizer
    of
    net
    adv_lr: learning
    rate
    of
    AdvStyle
    '''
    def __init__(self, concentration_coeff, base_style_num):
        super().__init__()
        self.register_buffer("proto_mean", torch.zeros((base_style_num, base_style_num), requires_grad=False))
        self.register_buffer("proto_std", torch.zeros((base_style_num, base_style_num), requires_grad=False))


    def AdvStyle(input, gt, net, optim, adv_lr):
        ### Adversarial Style Learning
        # Get style feature and normalized image
        B = input.size(0)
        mu = input.mean(dim=[2, 3], keepdim=True)
        var = input.var(dim=[2, 3], keepdim=True)
        sig = (var + 1e-5).sqrt()
        mu, sig = mu.detach(), sig.detach()
        input_normed = (input - mu) / sig
        input_normed = input_normed.detach().clone()
        # Set learnable style feature and adv optimizer
        adv_mu, adv_sig = mu, sig
        adv_mu.requires_grad_(True)
        adv_sig.requires_grad_(True)
        adv_optim = torch.optim.SGD(params=[adv_mu, adv_sig], lr=adv_lr, momentum=0, weight_decay=0)
        # Optimize adversarial style feature
        adv_optim.zero_grad()
        adv_input = input_normed * adv_sig+ adv_mu
        adv_output = net(adv_input)
        adv_loss = torch.nn.functional.cross_entropy(adv_output, gt)
        (- adv_loss).backward()
        adv_optim.step()
        ### Robust Model Training
        net.train()
        optim.zero_grad()
        adv_input = input_normed * adv_sig + adv_mu
        inputs = torch.cat((input, adv_input), dim=0)
        gt = torch.cat((gt, gt), dim=0)
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, gt)
        loss.backward()
        optim.step()