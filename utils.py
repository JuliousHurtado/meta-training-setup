#!/usr/bin/env python3

import argparse
import torch
from collections import defaultdict

from torch.nn import functional as F
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

    parser.add_argument('--use-ewc', type=str2bool, default=False,
                        help='Use EWC')
    parser.add_argument('--ewc-importance', type=float, default=100, metavar='LR',
                        help='learning rate for MAML (default: 0.5)')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument('--algorithm', type=str, default='MAML',
                        help='[MAML, ANIL, FT]')
    parser.add_argument('--remap-label', type=str2bool, default=True, metavar='LR',
                        help='Used in original MAML, to remap Label to amount of ways')

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
    parser.add_argument('--load-head', type=str, default='', metavar='LR',
                        help='Head of the modelo Load from , ./results/models/name_file')
    parser.add_argument('--init-ways', type=int, default=200, metavar='N',
                        help='number of ways (default: 5)')

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

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def fast_adapt(batch, learner, regs, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = torch.zeros(data.size(0), dtype=torch.bool)#.byte()
    adaptation_indices[torch.arange(shots*ways) * 2] = 1
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices].long()
    evaluation_data, evaluation_labels = data[~adaptation_indices], labels[~adaptation_indices].long()

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)

    if len(regs) > 0:
        for reg in regs:
            valid_error += reg(learner)

    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def train_normal(data_loader, learner, loss, optimizer, regs, device, ewc = None):
    learner.train()
    running_loss = 0
    running_corrects = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        out = learner(inputs)
        _, preds = torch.max(out, 1)
        l = loss(out, labels)

        if len(regs) > 0:
            for reg in regs:
                l += reg(learner)

        if ewc is not None:
            l += ewc.penalty(learner)

        l.backward()
        optimizer.step()

        running_loss += l.item()
        running_corrects += torch.sum(preds == labels.data)

    return running_loss / len(data_loader), running_corrects / len(data_loader)

def test_normal(model, data_loader, device):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = input.to(device), target.long().to(device)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.item() / len(data_loader.dataset)