#!/usr/bin/env python3

import torch as th
from torch import nn
from torch.autograd import grad
import copy

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module


def maml_update(model, lr, grads=None):
    """
    **Description**
    Performs a MAML update on model using grads and lr.
    The function re-routes the Python object, thus avoiding in-place
    operations.
    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.
    **Arguments**
    * **model** (Module) - The model to update.
    * **lr** (float) - The learning rate used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.
    **Example**
    ~~~python
    maml = l2l.algorithms.MAML(Model(), lr=0.1)
    model = maml.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    maml_update(model, lr=0.1, grads)
    ~~~
    """
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


class TMAML(BaseLearner):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)
    **Description**
    High-level implementation of *Model-Agnostic Meta-Learning*.
    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt`
    methods.
    For the first-order version of MAML (i.e. FOMAML), set the `first_order` flag to `True`
    upon initialization.
    **Arguments**
    * **model** (Module) - Module to be wrapped.
    * **lr** (float) - Fast adaptation learning rate.
    * **first_order** (bool, *optional*, default=False) - Whether to use the
    **References**
    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.”
    **Example**
    ~~~python
    linear = l2l.algorithms.MAML(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(self, model, lr, adaptation_steps = 1, min_used = 0, 
                    device = 'cpu', first_order=False):
        super(TMAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.adaptation_steps = adaptation_steps
        self.device = device

        self.previous_grad = None
        self.grad_batches = None
        self.mask = None

        self.min_used = min_used

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
        """
        **Description**
        Updates the clone parameters in place using the MAML update.
        **Arguments**
        * **loss** (Tensor) - Loss to minimize upon update.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        """
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order
        gradients = grad(loss,
                         self.module.parameters(),
                         retain_graph=second_order,
                         create_graph=second_order)
        self.module = maml_update(self.module, self.lr, gradients)

    def clone(self, first_order=None):
        """
        **Description**
        Returns a `MAML`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.
        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().
        **Arguments**
        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        """
        if first_order is None:
            first_order = self.first_order
        return TMAML(clone_module(self.module),
                    lr=self.lr,
                    adaptation_steps = self.adaptation_steps, 
                    min_used = self.min_used,
                    device = self.device,
                    first_order=first_order)

    def setLinear(self, num_dataset, device):
        self.module.setLinear(num_dataset, device)

    def getGradients(self):
        if self.previous_grad is None:
            self.previous_grad = []
            self.grad_batches = {}
            for i, p in enumerate(self.parameters()):
                self.previous_grad.append(p.grad.data)
                self.grad_batches[i] = [p.grad.data]
        else:
            temp = []
            for p, p_g, i in zip(self.parameters(), self.previous_grad, range(len(self.previous_grad))):
                temp.append(p.grad.data)
                self.grad_batches[i].append(p.grad.data - p_g)
                #print(i,"\t",self.grad_batches[i].sum())
            self.previous_grad = copy.deepcopy(temp)

    def createMask(self):
        self.mask = []
        for i in range(len(self.grad_batches)):
            m = th.zeros_like(self.grad_batches[i][0])
            for elem in self.grad_batches[i]:
                m += (elem > 0).float()

            m_f = ( m >= self.min_used).float().to(m.device)
            self.mask.append(m_f)

    def setMask(self):
        self.createMask()
        for i, p in enumerate(self.parameters()):
            #p.grad.data.mul_(th.ones_like(p))
            p.grad.data.mul_(self.mask[i])

        self.previous_grad = None
        self.mask = None
        self.grad_batches = None

    # def selectGradient(self,grad_pi):
    #     for i, j, m in zip(self.sum_grads_pi, grad_pi, self.mask):
    #         m += ((i>0)*(j>0) + (i<0)*(j<0)).float()

    # def updateGradientOuter(self, loss):
    #     grad_pi = grad(loss,
    #                      self.module.parameters(),
    #                      create_graph=False)

    #     if self.sum_grads_pi is None:
    #         self.sum_grads_pi = grad_pi
    #         self.mask = []
    #         for i in range(len(grad_pi)):
    #             self.mask.append(th.zeros_like(grad_pi[i]))
    #     else:  # accumulate all gradients from different episode learner
    #         self.selectGradient(grad_pi)
    #         self.sum_grads_pi = [th.add(i, j) for i, j in zip(self.sum_grads_pi, grad_pi)]

    # def setMask(self):
    #     for m in self.mask:
    #         m = ( m >= self.min_used).float()

    # def write_grads(self, generator, optimizer, loss, shots, meta_b_s):
    #     adaptation_data = generator.sample(shots=shots)

    #     data = [d for d in adaptation_data]
    #     x_dummy = th.cat([d[0].unsqueeze(0) for d in data], dim=0).to(self.device)
    #     y_dummy = th.cat([th.tensor(d[1]).view(-1) for d in data], dim=0).to(self.device)
        
    #     dummy_loss = loss(self.forward(x_dummy), y_dummy)

    #     self.setMask()

    #     optimizer.zero_grad()
    #     dummy_loss.backward()

    #     for i,elem in enumerate(self.module.parameters()):
    #         print(i)
    #         print("Max: ", max(self.mask[i]))
    #         print("Min: ", min(self.mask[i]))
    #         elem.grad = self.sum_grads_pi[i].detach()*self.mask[i]/meta_b_s
        
    #     optimizer.step()

    #     self.sum_grads_pi = None
    #     self.mask = None