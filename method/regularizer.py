import numpy as np
import copy
import random
from torch import nn
import torch
import torch.nn.functional as F

class FilterSparseReg(nn.Module):
    def __init__(self, c_theta = 0, i_beta = 0.1, f_beta = 1, num_layer = 4):
        super(FilterSparseReg, self).__init__()
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
                loss_reg += weight[i][(5*5)*j:(5*5)*(j+1)].norm(2)

        loss_reg /= weight.size(0)

        return loss_reg*self.c_omega



class GroupMask(object):
    """docstring for GroupMask"""
    def __init__(self, c_theta, threshold=5e-4, num_layers=4, i_beta=0.1, f_beta=0.8):
        super(GroupMask, self).__init__()
        self.threshold = threshold
        self.beta = np.linspace(i_beta, f_beta, num = num_layers+1)
        self.c_theta = c_theta
        self.masks = None

    def __call__(self, model):
        reg_theta = 0
        for j,layer in enumerate(model.model.base):

            reg_theta += layer.conv.weight.norm(2)*max(0.5,(1 - self.beta[j]))/2 #weight decay
        
            sizes = layer.conv.weight.size()
            temp = layer.conv.weight.view(sizes[0],-1)

            temp = temp.norm(2,1)
            #reg_theta += temp.sum()*(self.beta[j])
            reg_theta += temp[ temp > self.threshold ].sum()*(self.beta[j]) #l2,1

        return reg_theta * self.c_theta

    def setMasks(self, model):
        masks = []
        #counts_filter = []
        for elem in model.model.base:
            sizes = elem.conv.weight.size()
            temp = elem.conv.weight.view(sizes[0],-1)
            temp = temp.norm(2,1)

            mask = (temp <= self.threshold).type(torch.FloatTensor).to(temp.device)
            masks.append(mask)

        self.masks = masks

    def getZerosMasks(self, model):
        masks = []
        for elem in model.model.base:
            sizes = elem.conv.weight.size()
            temp = elem.conv.weight.view(sizes[0],-1)
            temp = temp.norm(2,1)
            masks.append(torch.zeros_like(temp))

        return masks

    def setGradZero(self, model):
        if self.masks is None:
            return None

        for j, elem in enumerate(model.model.base):
            elem.conv.weight.grad.mul_(self.masks[j].view(-1,1,1,1))
            elem.conv.bias.grad.mul_(self.masks[j])

            elem.normalize.weight.grad.mul_(self.masks[j])
            elem.normalize.bias.grad.mul_(self.masks[j])
        