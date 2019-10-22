import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable
from itertools import repeat
import numpy as np

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class levelsetLoss(nn.Module):
    def __init__(self):
        super(levelsetLoss, self).__init__()

    def forward(self, output, target):
        # input size = batch x 1 (channel) x height x width
        outshape = output.shape
        tarshape = target.shape
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:,ich], 1)
            target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2,3))/torch.sum(output, (2,3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss
		
class gradientLoss2d(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss2d, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW

        loss = torch.sum(dH) + torch.sum(dW)
        return loss

