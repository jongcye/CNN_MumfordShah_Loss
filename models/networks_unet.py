import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np


###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            # lr_l = 10 ** (opt.lr_first-((abs(opt.lr_last)-abs(opt.lr_first)) * epoch /(opt.niter-1)))
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netG == 'subpixelUnet':
        netG = SubpixelUnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                     gpu_ids=gpu_ids)
    elif which_model_netG == 'originalUnet':
        netG = OriginalUnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                     gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

# Sub_pixel U-Net
class SubpixelUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SubpixelUnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = SubpixelUnetSkipConnectionBlock(ngf * 8, ngf * 16, input_nc=None, submodule=None,
                                                     norm_layer=norm_layer, innermost=True)
        unet_block = SubpixelUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer)
        unet_block = SubpixelUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer)
        unet_block = SubpixelUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer)
        unet_block = SubpixelUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                     norm_layer=norm_layer, outermost=True)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class subpixelPool(nn.Module):
    def __init__(self, input_nc):
        super(subpixelPool, self).__init__()
        self.input_nc = input_nc
        self.output_nc = input_nc*4

    def forward(self, input):
        output1 = input[:, :, ::2, ::2]
        output2 = input[:, :, ::2, 1::2]
        output3 = input[:, :, 1::2, ::2]
        output4 = input[:, :, 1::2, 1::2]
        return torch.cat([output1, output2, output3, output4], dim=1)

class unSubpixelPool(nn.Module):
    def __init__(self, input_nc):
        super(unSubpixelPool, self).__init__()
        self.input_nc = input_nc
        self.output_nc = int(input_nc/4)

    def forward(self, input):
        output = Variable(torch.cuda.FloatTensor(input.shape[0], self.output_nc, input.shape[2]*2, input.shape[3]*2).zero_())
        output[:, :, ::2, ::2] = input[:, :int(self.input_nc/4), :, :]
        output[:, :, ::2, 1::2] = input[:, int(self.input_nc/4):int(self.input_nc/2), :, :]
        output[:, :, 1::2, ::2] = input[:, int(self.input_nc/2):int(self.input_nc*3/4), :, :]
        output[:, :, 1::2, 1::2] = input[:, int(self.input_nc*3/4):, :, :]
        return output

class SubpixelUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SubpixelUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = subpixelPool(input_nc)

        if outermost:
            C1 = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            C1 = nn.Conv2d(inner_nc*2, inner_nc, kernel_size=3, stride=1, padding=1, bias=False)
        B1 = norm_layer(inner_nc)
        R1 = nn.ReLU(True)
        # R1 = nn.LeakyReLU(0.2, True)
        if innermost:
            C2 = nn.Conv2d(inner_nc, inner_nc*2, kernel_size=3, stride=1, padding=1, bias=False)
            B2 = norm_layer(inner_nc*2)
        else:
            C2 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1, bias=False)
            B2 = norm_layer(inner_nc)
        R2 = nn.ReLU(True)
        CBR1CBR2 = [C1, B1, R1, C2, B2, R2]

        C1u = nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1, bias=False)
        B1u = norm_layer(inner_nc)
        R1u = nn.ReLU(True)
        if outermost:
            C2u = nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1, bias=False)
            B2u = norm_layer(inner_nc)
        else:
            C2u = nn.Conv2d(inner_nc, inner_nc*2, kernel_size=3, stride=1, padding=1, bias=False)
            B2u = norm_layer(inner_nc*2)
        R2u = nn.ReLU(True)
        CBR1CBR2u = [C1u, B1u, R1u, C2u, B2u, R2u]

        if outermost:
            Cend = nn.Conv2d(inner_nc, outer_nc, kernel_size=1, stride=1, padding=0, bias=True)

            down = CBR1CBR2
            up = CBR1CBR2u + [Cend] + [nn.Softmax2d()]
            model = down + [submodule] + up
        elif innermost:
            upconv = unSubpixelPool(inner_nc*2)
            down = [downconv] + CBR1CBR2
            up = [upconv]
            model = down + up
        else:
            upconv = unSubpixelPool(inner_nc*2)
            down = [downconv] + CBR1CBR2
            up = CBR1CBR2u + [upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

########################################################################################################################
# U-Net
class OriginalUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(OriginalUnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = OriginalUnetSkipConnectionBlock(ngf * 8, ngf * 16, input_nc=None, submodule=None,
                                                     norm_layer=norm_layer, innermost=True)
        unet_block = OriginalUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer)
        unet_block = OriginalUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer)
        unet_block = OriginalUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer)
        unet_block = OriginalUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                     norm_layer=norm_layer, outermost=True)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class OriginalUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(OriginalUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=2,
                             stride=2, padding=0, bias=use_bias)
        if outermost:
            C1 = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            C1 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1, bias=False)
        B1 = norm_layer(inner_nc)
        R1 = nn.ReLU(True)
        # R1 = nn.LeakyReLU(0.2, True)
        C2 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1, bias=False)
        B2 = norm_layer(inner_nc)
        R2 = nn.ReLU(True)
        CBR1CBR2 = [C1, B1, R1, C2, B2, R2]

        C1u = nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1, bias=False)
        B1u = norm_layer(inner_nc)
        R1u = nn.ReLU(True)
        C2u = nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1, bias=False)
        B2u = norm_layer(inner_nc)
        R2u = nn.ReLU(True)
        CBR1CBR2u = [C1u, B1u, R1u, C2u, B2u, R2u]

        # downrelu = nn.LeakyReLU(0.2, True)
        # downnorm = norm_layer(inner_nc)
        # uprelu = nn.ReLU(True)
        # upnorm = norm_layer(outer_nc)
        if outermost:
            Cend = nn.Conv2d(inner_nc, outer_nc, kernel_size=1, stride=1, padding=0, bias=True)

            down = CBR1CBR2
            up = CBR1CBR2u + [Cend]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0, bias=use_bias)
            down = [downconv] + CBR1CBR2
            up = [upconv]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0, bias=use_bias)
            down = [downconv] + CBR1CBR2
            up = CBR1CBR2u + [upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)