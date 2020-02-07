#!/usr/bin/env python3

import argparse
import torch
from collections import defaultdict

from learn2learn.vision.models import OmniglotCNN, MiniImagenetCNN

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

    parser.add_argument('--ways', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=5, metavar='N',
                        help='number of shots (default: 5)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=8, metavar='N',
                        help='tasks per step (default: 8)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=5, metavar='N',
                        help='steps per fast adaption (default: 5)')

    parser.add_argument('--iterations', type=int, default=15000, metavar='N',
                        help='number of iterations (default: 15000)')

    parser.add_argument('--dataset', type=str, default='Omniglot', metavar='C',
                        help='[Omniglot, MiniImagenet, cifar10, SVHN]')
    parser.add_argument('--input-channel', type=int, default=1, metavar='C',
                        help='1 for using black and white image and 3 for RGB')

    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--fast-lr', type=float, default=0.5, metavar='LR',
                        help='learning rate for MAML (default: 0.5)')
    parser.add_argument('--first-order', type=str2bool, default=False, metavar='LR',
                        help='Using First order MAML')
    parser.add_argument('--freeze-layer', nargs='+', type=int, metavar='LR',
                        help='List of frozen layers')
    parser.add_argument('--only-linear', type=str2bool, default=False,
                        help='train only classifier (default False)')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument('--algorithm', type=str, default='MAML',
                        help='[MAML, ANIL, FT]')
    parser.add_argument('--filter-reg', type=str2bool, default=False,
                        help='Using or not sparse-group regularization in conv filter (default False)')
    parser.add_argument('--cost-theta', type=float, default=0.01,
                        help='cost value of filter reg (default 0.01)')
    parser.add_argument('--linear-reg', type=str2bool, default=False,
                        help='Using or not sparse-group regularization in linear layer (default False)')
    parser.add_argument('--cost-omega', type=float, default=0.01,
                        help='cost value of linear reg (default 0.01)')
    parser.add_argument('--sparse-reg', type=str2bool, default=False,
                        help='Using or not sparse-group regularization in conv filter (default False)')

    parser.add_argument('--save-model', type=str2bool, default=True, metavar='LR',
                        help='save model (default True)')
    parser.add_argument('--load-model', type=str, default='', metavar='LR',
                        help='Model to Load, ./results/models/name_file')

    return parser

def saveValues(name_file, results, model, args):
    torch.save({
            'results': results,
            'args': args,
            'checkpoint': model.state_dict()
            }, name_file)

def getModel(input_channels, ways = 5, device = 'cpu'):
    if input_channels == 1:
        return OmniglotCNN(ways).to(device)
    elif input_channels == 3:
        return MiniImagenetCNN(ways).to(device)
    else:
        raise Exception('Input Channels must be 1 or 3, not: {}'.format(input_channels))

def getAlgorithm(algorithm, model, fast_lr, first_order, freeze_layer):
    if algorithm == 'MAML':
        return MAML(model, lr=fast_lr, first_order=first_order)
    elif algorithm == 'ANIL':
        return MAML(model, lr=fast_lr, first_order=first_order, freeze_layer=freeze_layer) 
    elif algorithm == 'FT':
        return model
    else:
        raise Exception('Algorithm {} not supported'.format(algorithm))

def getRegularizer(convFilter, c_theta, linearReg, c_omega, sparseFilter):
    regularizator = []

    if convFilter:
        regularizator.append(FilterReg(c_theta))
    if linearReg:
        regularizator.append(LinearReg(c_omega))
    if sparseFilter:
        regularizator.append(FilterSparseReg(c_theta))

    return regularizator


def create_bookkeeping(dataset):
    """
    Iterates over the entire dataset and creates a map of target to indices.
    Returns: A dict with key as the label and value as list of indices.
    """

    assert hasattr(dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

    labels_to_indices = defaultdict(list)
    indices_to_labels = defaultdict(int)
    for i in range(len(dataset)):
        try:
            label = dataset[i][1]
            # if label is a Tensor, then take get the scalar value
            if hasattr(label, 'item'):
                label = dataset[i][1].item()
        except ValueError as e:
            raise ValueError(
                'Requires scalar labels. \n' + str(e))

        labels_to_indices[label].append(i)
        indices_to_labels[i] = label

    dataset.labels_to_indices = labels_to_indices
    dataset.indices_to_labels = indices_to_labels
    dataset.labels = list(dataset.labels_to_indices.keys())
