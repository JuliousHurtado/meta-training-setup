#!/usr/bin/env python3

import os
import random
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from learn2learn.vision.models import OmniglotCNN, MiniImagenetCNN

from method.maml import MAML

from legacy.utils import getRandomDataset

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def saveValues(name_file, results, model, args):
    th.save({
            'results': results,
            'args': args,
            'checkpoint': model.state_dict()
            }, name_file)

def getModel(input_channels, ways, device):
    if input_channels == 1:
        return OmniglotCNN(ways).to(device)
    elif input_channels == 3:
        return MiniImagenetCNN(ways).to(device)
    else:
        raise Exception('Input Channels must be 1 or 3, not: {}'.format(input_channels))

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def getDataset(name_dataset, ways, shots):
    generators = {}
    if name_dataset == 'Omniglot':
        omniglot = l2l.vision.datasets.FullOmniglot(root='./data',
                                                transform=transforms.Compose([
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=True)
        dataset = l2l.data.MetaDataset(omniglot)
        classes = list(range(1623))
        random.shuffle(classes)

        for mode, num_tasks, split in zip(['train','validation','test'],
                                        [20000,1024,1024],[[0,1100],[1100,1200],[1200,-1]]):

            transforms = [
                l2l.data.transforms.FilterLabels(dataset, classes[split[0]:split[1]]),
                l2l.data.transforms.NWays(dataset, ways),
                l2l.data.transforms.KShots(dataset, 2*shots),
                l2l.data.transforms.LoadData(dataset),
                l2l.data.transforms.RemapLabels(dataset),
                l2l.data.transforms.ConsecutiveLabels(dataset),
                l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
            ]

            generators[mode] = l2l.data.TaskDataset(dataset,
                                       task_transforms=transforms,
                                       num_tasks=num_tasks)

    elif name_dataset == 'MiniImagenet':
        train_dataset = None
        for mode, num_tasks in zip(['train','validation','test'],[20000,600,600]):
            dataset = l2l.data.MetaDataset(l2l.vision.datasets.MiniImagenet(root='./data', mode=mode))

            if train_dataset is None:
                train_dataset = dataset

            transforms = [
                    NWays(dataset, ways),
                    KShots(dataset, 2*shots),
                    LoadData(dataset),
                    RemapLabels(dataset),
                    ConsecutiveLabels(train_dataset),
                ]

            generators[mode] = l2l.data.TaskDataset(dataset,
                                       task_transforms=transforms,
                                       num_tasks=num_tasks)

    else:
        raise Exception('Dataset {} not supported'.format(dataset))

    return generators

def getAlgorithm(algorithm, model, fast_lr, first_order, freeze_layer):
    if algorithm == 'MAML':
        return MAML(model, lr=fast_lr, first_order=first_order)
    elif algorithm == 'ANIL':
        return MAML(model, lr=fast_lr, first_order=first_order, freeze_layer=freeze_layer) 
    else:
        raise Exception('Algorithm {} not supported'.format(algorithm))


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = torch.zeros(data.size(0), dtype=torch.bool)#.byte()
    adaptation_indices[torch.arange(shots*ways) * 2] = 1
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[~adaptation_indices], labels[~adaptation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        meta_alg,
        data_generators,
        ways,
        shots,
        device,
        lr=0.003,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
        parser=None,
):
    opt = optim.Adam(meta_alg.parameters(), lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    results = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
        'test_acc': [],
        'test_loss': [],

    }

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = meta_alg.clone()
            batch = data_generators['train'].sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            #meta_alg.printValue()
            #learner.printValue()
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = meta_alg.clone()
            batch = data_generators['validation'].sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = meta_alg.clone()
            batch = data_generators['test'].sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        if iteration % 100 == 0:
            # Print some metrics
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / meta_batch_size)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            print('Meta Valid Error', meta_valid_error / meta_batch_size)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
            print('Meta Test Error', meta_test_error / meta_batch_size)
            print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)

        results['train_loss'].append(meta_train_error / meta_batch_size)
        results['train_acc'].append(meta_train_accuracy / meta_batch_size)
        results['val_loss'].append(meta_valid_error / meta_batch_size)
        results['val_acc'].append(meta_valid_accuracy / meta_batch_size)
        results['test_loss'].append(meta_test_error / meta_batch_size)
        results['test_acc'].append(meta_test_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in meta_alg.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    name_file = 'results/{}_{}'.format(str(time.time()),parser.algorithm)
    args = vars(parser.parse_args())
    saveValues(name_file, results, meta_alg.module, args)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')

    parser.add_argument('--ways', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=5, metavar='N',
                        help='number of shots (default: 5)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=4, metavar='N',
                        help='tasks per step (default: 4)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=10, metavar='N',
                        help='steps per fast adaption (default: 10)')

    parser.add_argument('--iterations', type=int, default=30000, metavar='N',
                        help='number of iterations (default: 30000)')

    parser.add_argument('--dataset', type=str, default='Omniglot', metavar='C',
                        help='[Omniglot, MiniImagenet]')
    parser.add_argument('--input-channel', type=int, default=1, metavar='C',
                        help='1 for using black and white image and 3 for RGB')

    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--fast-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for MAML (default: 0.5)')
    parser.add_argument('--first-order', type=str2bool, default=True, metavar='LR',
                        help='Using First order MAML')
    parser.add_argument('--freeze-layer', nargs='+', type=int, metavar='LR',
                        help='List of frozen layers')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument('--download-location', type=str, default="/tmp/mnist", metavar='S',
                        help='download location for train data (default : /tmp/mnist')

    parser.add_argument('--algorithm', type=str, default='MAML',
                        help='[MAML, ANIL]')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    model = getModel(args.input_channel, args.ways, device)
    meta_model = getAlgorithm(args.algorithm, model, args.fast_lr, args.first_order, args.freeze_layer)
    
    # print(len(list(meta_model.getParams())))

    #print(model)
    data_generators = getDataset(args.dataset, args.ways, args.shots)
    # data_generators = getRandomDataset(args.ways, False)

    main(meta_model,
         data_generators,
         ways=args.ways,
         shots=args.shots,
         device=device,
         lr=args.lr,
         meta_batch_size=args.tasks_per_step,
         adaptation_steps=args.fast_adaption_steps,
         num_iterations=args.iterations,
         seed=args.seed,
         parser=parser)