#!/usr/bin/env python3

import argparse
import torch
from collections import defaultdict

from torch.nn import functional as F

from models.TaskModel import TaskManager

from method.maml import MAML
from method.regularizer import FilterReg, Lnorm

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
    parser.add_argument('--task-with-meta', type=int, default=-1)


    #--------------------------------Model----------------------------------------------#
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--layers', type=int, default=4)


    #--------------------------------Training-------------------------------------------#
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dataset', type=str, default='multi', metavar='C',
                        help='[multi, pmnist]')
    parser.add_argument('--num-tasks', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--only-linear', type=str2bool, default=False)
    parser.add_argument('--task-linear', type=int, default=-1)

    #--------------------------------Extra----------------------------------------------#
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-model', type=str2bool, default=False)
    parser.add_argument('--use-load-model', type=str2bool, default=False)
    parser.add_argument('--load-model', type=str, default='./results/temp.pth')


    #---------------------------------Regularization-------------------------------------#
    parser.add_argument('--cost-theta', type=float, default=0.001)
    parser.add_argument('--regularization',default='', type=str, required=False,
                        choices=['','1','2','1,2'])


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

def getRegularizer(norm, c_theta):
    regularizator = None

    if norm == '1':
        regularizator = Lnorm(c_theta,1)
    elif norm == '2':
        regularizator = Lnorm(c_theta,2)
    elif norm == '1,2':
        regularizator = FilterReg(c_theta)

    return regularizator