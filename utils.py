#!/usr/bin/env python3

import argparse
import torch
from collections import defaultdict

from torch.nn import functional as F

from models.TaskModel import TaskManager

from method.maml import MAML
from method.regularizer import FilterReg, LinearReg, FilterSparseReg

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getArguments():
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')


    #--------------------------------Meta Train-----------------------------------------#
    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--fast-lr', type=float, default=0.5)
    parser.add_argument('--first-order', type=str2bool, default=False)
    parser.add_argument('--meta-iterations', type=int, default=5000)
    parser.add_argument('--meta-batch-size', type=int, default=16)
    parser.add_argument('--adaptation-steps', type=int, default=5)
    parser.add_argument('--meta-train', type=str2bool, default=False)


    #--------------------------------Model----------------------------------------------#
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--layers', type=int, default=4)


    #--------------------------------Training-------------------------------------------#
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dataset', type=str, default='multi', metavar='C',
                        help='[multi, pmnist]')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)


    #--------------------------------Extra----------------------------------------------#
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-model', type=str2bool, default=False)
    parser.add_argument('--use-load-model', type=str2bool, default=False)
    parser.add_argument('--load-model', type=str, default='./results/temp.pth')


    #---------------------------------Regularization-------------------------------------#
    parser.add_argument('--filter-reg', type=str2bool, default=False)
    parser.add_argument('--sparse-reg', type=str2bool, default=False)
    parser.add_argument('--cost-theta', type=float, default=0.01)
    parser.add_argument('--linear-reg', type=str2bool, default=False)
    parser.add_argument('--cost-omega', type=float, default=0.01)


    return parser

def saveValues(name_file, results, model, args):
    torch.save({
            'results': results,
            'args': args,
            'checkpoint': model.state_dict()
            }, name_file)

def getModel(cls_per_task, hidden_size=32, layers=4,device='cpu', channels=3):
    return TaskManager(cls_per_task, channels, hidden_size, layers, device).to(device)

def getMetaAlgorithm(model, fast_lr, first_order):
    return MAML(model, lr=fast_lr, first_order=first_order)

def getRegularizer(convFilter, c_theta, linearReg, c_omega, sparseFilter):
    regularizator = []

    if convFilter:
        regularizator.append(FilterReg(c_theta))
    if linearReg:
        regularizator.append(LinearReg(c_omega))
    if sparseFilter:
        regularizator.append(FilterSparseReg(c_theta))

    return regularizator