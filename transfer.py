#!/usr/bin/env python3

import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder, SVHN, CIFAR10

import learn2learn as l2l
from learn2learn.data.transforms import KShots, LoadData, RemapLabels, ConsecutiveLabels
from method.meta_transform import NWays

from utils import saveValues, getArguments, getModel, getAlgorithm, getRegularizer, create_bookkeeping

#from legacy.utils import getRandomDataset

import traceback
import warnings
import sys

base_path = 'results/models'

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def train_fine_tuning(data_loader, learner, loss, optimizer, regs, device):
    model.train()
    running_loss = 0
    running_corrects = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        out = learner(inputs)
        _, preds = torch.max(out, 1)
        l = loss(out, labels)

        if len(regs) > 0:
            for reg in regs:
                l += reg(learner)

        l.backward()
        optimizer.step()

        running_loss += l.item()
        running_corrects += torch.sum(preds == labels.data)

    return running_loss / len(data_loader), running_corrects / len(data_loader)

def test(model, data_loader):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.item() / len(data_loader.dataset)

def getDataset(name_dataset, ways, shots, fine_tuning):
    generators = {}
    transform_data = transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if name_dataset == 'SVHN':
        dataset = SVHN('./data/', split='train', transform=transform_data, download=True)
        
        if fine_tuning:
            generators['train'] = torch.utils.data.DataLoader(dataset, batch_size=64)
        else:
            create_bookkeeping(dataset)

            meta_transforms = [
                    l2l.data.transforms.NWays(dataset, ways),
                    l2l.data.transforms.KShots(dataset, 2*shots),
                    l2l.data.transforms.LoadData(dataset),
                ]
            generators['train'] = l2l.data.TaskDataset(l2l.data.MetaDataset(dataset),
                                           task_transforms=meta_transforms)

        dataset = SVHN('./data/', split='train', transform=transform_data, download=True)
        generators['validation'] = torch.utils.data.DataLoader(dataset, batch_size=64)

        dataset = SVHN('./data/', split='test', transform=transform_data, download=True)
        generators['test'] = torch.utils.data.DataLoader(dataset, batch_size=64)

    elif name_dataset == 'cifar10':
        dataset_train = CIFAR10('./data/', train=True, transform=transform_data, download=True)
        dataset_test = CIFAR10('./data/', train=False, transform=transform_data, download=True)

        if fine_tuning:
            generators['train'] = torch.utils.data.DataLoader(dataset_train, batch_size=64)
        else:
            create_bookkeeping(dataset_train)

            meta_transforms = [
                        NWays(dataset_train, ways),
                        KShots(dataset_train, 2*shots),
                        LoadData(dataset_train),
                    ]

            generators['train'] = l2l.data.TaskDataset(l2l.data.MetaDataset(dataset_train), 
                                                task_transforms=meta_transforms, num_tasks=20000)
        generators['validation'] = torch.utils.data.DataLoader(dataset_train, batch_size=64)
        generators['test'] = torch.utils.data.DataLoader(dataset_test, batch_size=64)

    else:
        raise Exception('Dataset {} not supported'.format(name_dataset))

    return generators

def fast_adapt(batch, learner, regs, loss, adaptation_steps, shots, ways, device):
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

    if len(regs) > 0:
        for reg in regs:
            valid_error += reg(learner)

    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def loadModel(file_name, model, device, ways):
    checkpoint = torch.load(os.path.join(base_path,file_name), map_location=device)

    model.load_state_dict(checkpoint['checkpoint'])
    model.linear = nn.Linear(model.linear.weight.size(1), ways, bias=True).to(device)

    return model

def addResults(model, data_generators, results, iteration, train_error, train_accuracy, batch_size):
    valid_accuracy = test(model, data_generators['validation'])
    test_accuracy = test(model, data_generators['test'])

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

    for iteration in range(num_iterations):
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        if fine_tuning:
            evaluation_error, evaluation_accuracy = train_fine_tuning(data_generators['train'], meta_alg, loss, opt, regs, device)
            meta_train_error += evaluation_error
            meta_train_accuracy += evaluation_accuracy.item()

            addResults(meta_alg, data_generators, results, iteration, meta_train_error, meta_train_accuracy, 1)
        else:
            opt.zero_grad()
            for task in range(meta_batch_size):
                # Compute meta-training loss
                learner = meta_alg.clone()
                batch = data_generators['train'].sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   regs,
                                                                   loss,
                                                                   adaptation_steps,
                                                                   shots,
                                                                   ways,
                                                                   device)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

            # Average the accumulated gradients and optimize
            for p in meta_alg.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()
        
            if iteration % int(num_iterations/60):
                addResults(meta_alg, data_generators, results, iteration, meta_train_error, meta_train_accuracy, meta_batch_size)

    if args['save_model']:
        name_file = '{}/{}_{}_{}'.format(base_path,str(time.time()),args['algorithm'], args['dataset'])
        if fine_tuning:
            saveValues(name_file, results, meta_alg, args)
        else:
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
    
    model = getModel(args.input_channel, device = device)
    model = loadModel(args.load_model, model, device, args.ways)

    meta_model = getAlgorithm(args.algorithm, model, args.fast_lr, args.first_order, args.freeze_layer)
    regs = getRegularizer( 
                    args.filter_reg, args.cost_theta,
                    args.linear_reg, args.cost_omega)

    #print(model)
    data_generators = getDataset(args.dataset, args.ways, args.shots, fine_tuning)
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