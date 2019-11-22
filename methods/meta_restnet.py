#!/usr/bin/env python3

import torch as th
from torch import nn
from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module


def maml_update(model, lr, grads=None):
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            print(msg)
        for p, g in zip(params, grads):
            p.grad = g

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - lr * p.grad

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None:
            model._buffers[buffer_key] = buff - lr * buff.grad

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = maml_update(model._modules[module_key],
                                                 lr=lr,
                                                 grads=None)
    return model

class MetaRestNet(BaseLearner):
    def __init__(self, model, lr, adaptation_steps = 1, device = 'cpu', 
                        first_order=False, num_freeze_layers = 1):
        super(MetaRestNet, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.adaptation_steps = adaptation_steps
        self.device = device
        self.num_freeze_layers = num_freeze_layers

        self.freeze_layers(num_freeze_layers)

    def freeze_layers(self, num_layers):
        self.module.conv1.required_grad = False
        self.module.bn1.required_grad = False

        layers = [self.module.layer1,self.module.layer2,
                    self.module.layer3,self.module.layer4]

        for i in range(num_layers):
            for param in layers[i].parameters():
                param.required_grad = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def meta_train(self, adaptation_data, evaluation_data, loss):
        for step in range(self.adaptation_steps):
            data = [d for d in adaptation_data]
            X = th.cat([d[0].unsqueeze(0) for d in data], dim=0).to(self.device)
            # X = th.cat([d[0] for d in data], dim=0).to(device)
            y = th.cat([th.tensor(d[1]).view(-1) for d in data], dim=0).to(self.device)
            train_error = loss(self.forward(X), y)
            train_error /= len(adaptation_data)
            self.adapt(train_error)

    def adapt(self, loss, first_order=None):
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order
        gradients = grad(loss,
                         self.module.fc.parameters(),
                         retain_graph=second_order,
                         create_graph=second_order)
        self.module.fc = maml_update(self.module.fc, self.lr, gradients)

    def clone(self, first_order=None):
        if first_order is None:
            first_order = self.first_order
        return MetaRestNet(clone_module(self.module),
                    lr=self.lr,
                    adaptation_steps = self.adaptation_steps, 
                    device = self.device,
                    first_order=first_order,
                    num_freeze_layers=self.num_freeze_layers)

    def setLinear(self, num_dataset, device):
        self.module.setLinear(num_dataset, device)

    def printParam(self):
        for i, param in enumerate(self.module.parameters()):
            print("{}_{}".format(i, param.sum()))