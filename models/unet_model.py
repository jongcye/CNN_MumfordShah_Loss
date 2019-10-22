import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from .base_model import BaseModel
from . import networks_unet
from .loss import *
from . import loss
import sys
import ipdb


class UNetModel(BaseModel):
    def name(self):
        return 'UNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, int(opt.output_nc/2), size, size).long()
        # load/define networks
        self.net = networks_unet.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.net, 'G_A', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionCE = torch.nn.CrossEntropyLoss()
            self.criterionLS = levelsetLoss()
            self.criterionTV = gradientLoss2d()

            # initialize optimizers
            if opt.optim == 'Adam':
                self.optimizer_ = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'SGD':
                self.optimizer_ = torch.optim.SGD(self.net.parameters(), lr=opt.lr)
            elif opt.optim == 'RMS':
                self.optimizer_ = torch.optim.RMSprop(self.net.parameters(), lr=opt.lr)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_)
            for optimizer in self.optimizers:
                self.schedulers.append(networks_unet.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks_unet.print_network(self.net)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B'].long()
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']

    def set_input_test(self, input):
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.net(real_A)
        fake_B2 = torch.clamp(fake_B[:, 0:2], 1e-10, 1.0)

        self.fake_B2 = fake_B2.data
        self.fake_B = fake_B.data

        # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        fake_B = self.net(self.real_A)
        fake_B2 = torch.clamp(fake_B[:, 0:2], 1e-10, 1.0)
        
        loss_C = 0
        numch = 0
        for ibatch in range(self.real_B.shape[0]):
            if torch.max(self.real_B[ibatch, 0]) != 0:
                realB = self.real_B[ibatch, 0].unsqueeze(0)
                fakeB = fake_B2[ibatch, :].unsqueeze(0)
                loss_C += self.criterionCE(fakeB, realB) # * 100
                numch += 1.0
        if numch > 0:
            loss_C = loss_C / numch
            self.loss_C = loss_C.item()

        else:
            self.loss_C = 0

        loss_L = self.criterionLS(fake_B2, self.real_A)
        loss_A = self.criterionTV(fake_B2) *0.001
        loss_LS = (loss_L + loss_A) * self.opt.lambda_A
        
        loss_tot = loss_C+loss_LS
        loss_tot.backward()

        self.fake_B2= fake_B2.data
        self.loss_LS = loss_LS.item()

    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer_.zero_grad()
        self.backward_G()
        self.optimizer_.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('C', self.loss_C), ('LS', self.loss_LS)])
        return ret_errors

    def get_current_visuals(self):
        real_A1 = util.tensor2im(self.input_A[:, 0])
        real_A2 = util.tensor2im(self.input_A[:, 1])
        real_A3 = util.tensor2im(self.input_A[:, 2])

        real_B2 = util.tensor2im(self.input_B[:, 0])

        fake_B0 = util.tensor2im(self.fake_B2[:, 0])
        fake_B1 = util.tensor2im(self.fake_B2[:, 1])

        ret_visuals = OrderedDict([('real_A1', real_A1),('real_A2', real_A2), ('real_A3', real_A3),
                                   ('real_B2', real_B2), ('fake_B0', fake_B0), ('fake_B1', fake_B1)])
        return ret_visuals

    def get_current_data(self):
        ret_visuals = OrderedDict([('real_A2', self.input_A[:, 1]), ('real_B2', self.input_B[:, 1]), ('fake_B2', self.fake_B2)])
        return ret_visuals

    def get_current_data_seg(self):
        ret_visuals = OrderedDict([('real_A2', self.input_A[:, 1]), ('fake_B2', self.fake_B2)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.net, 'G_A', label, self.gpu_ids)
