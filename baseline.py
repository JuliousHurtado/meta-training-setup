#!/usr/bin/env python3

import os
import random
import time

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
from method.regularizer import FilterReg, LinearReg

from utils import saveValues, getArguments, getModel, getAlgorithm, getRegularizer, \
                    test_normal, fast_adapt, train_normal

#from legacy.utils import getRandomDataset

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

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
    elif name_dataset == 'Tiny-FT':
        for mode in ['train','validation','test']:
            dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode=mode)    
            generators[mode] = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    else:
        raise Exception('Dataset {} not supported'.format(dataset))

    return generators

def addResults(model, data_generators, results, iteration, train_error, train_accuracy, batch_size, device):
    valid_accuracy = test_normal(model, data_generators['validation'], device)
    test_accuracy = test_normal(model, data_generators['test'], device)

    # Print some metrics
    print('\n')
    print('Iteration', iteration)
    print('Meta Train Error', train_error / batch_size)
    print('Meta Train Accuracy', train_accuracy / batch_size)
    print('Meta Valid Accuracy', valid_accuracy)
    print('Meta Test Accuracy', test_accuracy)
    print('\n', flush=True)

    results['train_loss'].append(train_error / batch_size)
    results['train_acc'].append(train_accuracy / batch_size)
    results['val_acc'].append(valid_accuracy)
    results['test_acc'].append(test_accuracy)

def main(
        meta_alg,
        data_generators,
        regs,
        ways,
        shots,
        device,
        lr=0.003,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
        args=None,
        fine_tuning=False,

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

    # for iteration in range(num_iterations):
    #     meta_train_error = 0.0
    #     meta_train_accuracy = 0.0
    #     meta_valid_error = 0.0
    #     meta_valid_accuracy = 0.0
    #     meta_test_error = 0.0
    #     meta_test_accuracy = 0.0
    #     if fine_tuning:
    #         evaluation_error, evaluation_accuracy = train_normal(data_generators['train'], meta_alg, loss, opt, regs, device)
    #         meta_train_error = evaluation_error
    #         meta_train_accuracy = evaluation_accuracy.item()

    #         addResults(meta_alg, data_generators, results, iteration, meta_train_error, meta_train_accuracy, 1, device)
    #     else:
    #         opt.zero_grad()
    #         for task in range(meta_batch_size):
    #             # Compute meta-training loss
    #             learner = meta_alg.clone()
    #             batch = data_generators['train'].sample()
    #             evaluation_error, evaluation_accuracy = fast_adapt(batch,
    #                                                                learner,
    #                                                                regs,
    #                                                                loss,
    #                                                                adaptation_steps,
    #                                                                shots,
    #                                                                ways,
    #                                                                device)
    #             #meta_alg.printValue()
    #             #learner.printValue()
    #             evaluation_error.backward()
    #             meta_train_error += evaluation_error.item()
    #             meta_train_accuracy += evaluation_accuracy.item()

    #             # Compute meta-validation loss
    #             learner = meta_alg.clone()
    #             batch = data_generators['validation'].sample()
    #             evaluation_error, evaluation_accuracy = fast_adapt(batch,
    #                                                                learner,
    #                                                                regs,
    #                                                                loss,
    #                                                                adaptation_steps,
    #                                                                shots,
    #                                                                ways,
    #                                                                device)
    #             meta_valid_error += evaluation_error.item()
    #             meta_valid_accuracy += evaluation_accuracy.item()

    #             # Compute meta-testing loss
    #             learner = meta_alg.clone()
    #             batch = data_generators['test'].sample()
    #             evaluation_error, evaluation_accuracy = fast_adapt(batch,
    #                                                                learner,
    #                                                                regs,
    #                                                                loss,
    #                                                                adaptation_steps,
    #                                                                shots,
    #                                                                ways,
    #                                                                device)
    #             meta_test_error += evaluation_error.item()
    #             meta_test_accuracy += evaluation_accuracy.item()

    #         if iteration % 500 == 0:
    #             # Print some metrics
    #             print('\n')
    #             print('Iteration', iteration)
    #             print('Meta Train Error', meta_train_error / meta_batch_size)
    #             print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
    #             print('Meta Valid Error', meta_valid_error / meta_batch_size)
    #             print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
    #             print('Meta Test Error', meta_test_error / meta_batch_size)
    #             print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)
    #             print('\n', flush=True)

    #         results['train_loss'].append(meta_train_error / meta_batch_size)
    #         results['train_acc'].append(meta_train_accuracy / meta_batch_size)
    #         results['val_loss'].append(meta_valid_error / meta_batch_size)
    #         results['val_acc'].append(meta_valid_accuracy / meta_batch_size)
    #         results['test_loss'].append(meta_test_error / meta_batch_size)
    #         results['test_acc'].append(meta_test_accuracy / meta_batch_size)

    #         # Average the accumulated gradients and optimize
    #         for p in meta_alg.parameters():
    #             p.grad.data.mul_(1.0 / meta_batch_size)
    #         opt.step()

    if args['save_model']:
        name_file = 'results/{}_{}'.format(str(time.time()),args['algorithm'])
        saveValues(name_file, results, meta_alg.module, args)

if __name__ == '__main__':
    parser = getArguments()
    args = parser.parse_args()

    fine_tuning = False
    if args.algorithm == 'FT':
        fine_tuning = True

    use_cuda = torch.cuda.is_available()

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
    regs = getRegularizer( 
                    args.filter_reg, args.cost_theta,
                    args.linear_reg, args.cost_omega,
                    args.sparse_reg)

    # print(len(list(meta_model.getParams())))

    #print(model)
    data_generators = getDataset(args.dataset, args.ways, args.shots)
    #data_generators = getRandomDataset(args.ways, False)

    main(meta_model,
         data_generators,
         regs,
         ways=args.ways,
         shots=args.shots,
         device=device,
         lr=args.lr,
         meta_batch_size=args.tasks_per_step,
         adaptation_steps=args.fast_adaption_steps,
         num_iterations=args.iterations,
         seed=args.seed,
         args=vars(parser.parse_args()),
         fine_tuning=fine_tuning)