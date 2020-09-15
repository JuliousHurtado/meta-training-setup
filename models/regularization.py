import numpy as np
import copy
import random
from torch import nn
import torch
import torch.nn.functional as F


class MaskRegularization(nn.Module):
    def __init__(self, c_theta = 0):
        super(MaskRegularization, self).__init__()
        self.c_theta = c_theta

    def __call__(self, mask):
        loss_reg = 0
        for j,param in enumerate(mask):
            loss_reg += param.abs().sum()/param.numel()

        return loss_reg*self.c_theta


class DiffRegularization(nn.Module):
    def __init__(self, c_theta = 0):
        super(DiffRegularization, self).__init__()
        self.c_theta = c_theta

    def __call__(self, s_feat, p_feat):
        loss_reg = 0
        loss_reg += (s_feat*p_feat).sum()

        return loss_reg*self.c_theta