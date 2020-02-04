import numpy as np
import copy
import random
from torch import nn
import torch
import torch.nn.functional as F

class FilterSparseReg(nn.Module):
    def __init__(self, c_theta = 0, i_beta = 0.1, f_beta = 1, num_layer = 4):
        super(FilterReg, self).__init__()
        self.beta = np.linspace(i_beta, f_beta, num = num_layer)
        self.c_theta = c_theta

    def __call__(self, model):
        loss_reg = 0
        self.convLayer = []
        self.getConvLayer(model)
        for j,param in enumerate(self.convLayer):
            temp = param.weight.view(param.weight.size(0),-1)

            temp = temp.norm(2,1)
            loss_reg += temp.sum()*(self.beta[j]/param.weight.size(0)) #l2,1

        return loss_reg*self.c_theta

    def getConvLayer(self, network):
        for layer in network.children():
            if list(layer.children()) != []: # if sequential layer, apply recursively to layers in sequential layer
                self.getConvLayer(layer)
            if isinstance(layer, nn.Conv2d):
                self.convLayer.append(layer)

class FilterReg(nn.Module):
    def __init__(self, c_theta = 0, i_beta = 0.1, f_beta = 1, num_layer = 4):
        super(FilterReg, self).__init__()
        self.beta = np.linspace(i_beta, f_beta, num = num_layer)
        self.c_theta = c_theta

    def __call__(self, model):
        loss_reg = 0
        self.convLayer = []
        self.getConvLayer(model)
        for j,param in enumerate(self.convLayer):
            dim_0 = param.weight.size(0)*param.weight.size(1)
            temp = param.weight.view(dim_0,-1)

            temp = temp.norm(2,1)
            loss_reg += temp.sum()*(self.beta[j]/dim_0) #l1,1,2

        return loss_reg*self.c_theta

    def getConvLayer(self, network):
        for layer in network.children():
            if list(layer.children()) != []: # if sequential layer, apply recursively to layers in sequential layer
                self.getConvLayer(layer)
            if isinstance(layer, nn.Conv2d):
                self.convLayer.append(layer)

class LinearReg(nn.Module):
    def __init__(self, c_omega = 0):
        super(LinearReg, self).__init__()
        self.c_omega = c_omega

    def __call__(self, model):
        loss_reg = 0
        weight = list(model.parameters())[-2]

        for i in range(weight.size(0)):
            for j in range(32):
                loss_reg += weight[i][(25*25)*j:(25*25)*(j+1)].norm(2)

        return loss_reg*self.c_omega