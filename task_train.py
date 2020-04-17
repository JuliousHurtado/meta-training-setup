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
from torchvision.datasets import SVHN, CIFAR10

from models.task_model import TaskModel
from utils import saveValues, getArguments, train_normal, test_normal

base_path = 'results'

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def getDataset(name_dataset):
    generators = {}
    transform_data = transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    bs = 4
    if name_dataset == 'SVHN':
        dataset = SVHN('./data/', split='train', transform=transform_data, download=True)        
        generators['train'] = torch.utils.data.DataLoader(dataset, batch_size=bs)

        dataset = SVHN('./data/', split='train', transform=transform_data, download=True)
        generators['validation'] = torch.utils.data.DataLoader(dataset, batch_size=bs)

        dataset = SVHN('./data/', split='test', transform=transform_data, download=True)
        generators['test'] = torch.utils.data.DataLoader(dataset, batch_size=bs)

    elif name_dataset == 'cifar10':
        dataset_train = CIFAR10('./data/', train=True, transform=transform_data, download=True)
        dataset_test = CIFAR10('./data/', train=False, transform=transform_data, download=True)

        generators['train'] = torch.utils.data.DataLoader(dataset_train, batch_size=bs)
        generators['validation'] = torch.utils.data.DataLoader(dataset_train, batch_size=bs)
        generators['test'] = torch.utils.data.DataLoader(dataset_test, batch_size=bs)

    else:
        raise Exception('Dataset {} not supported'.format(name_dataset))

    return generators

def addResults(model, data_generators, results, iteration, train_error, train_accuracy, device):
    valid_accuracy = test_normal(model, data_generators['validation'], device)
    test_accuracy = test_normal(model, data_generators['test'], device)

    # Print some metrics
    print('\n')
    print('Iteration', iteration)
    print('Meta Train Error', train_error)
    print('Meta Train Accuracy', train_accuracy)
    print('Meta Valid Accuracy', valid_accuracy)
    print('Meta Test Accuracy', test_accuracy)
    print('\n', flush=True)

    results['train_loss'].append(train_error)
    results['train_acc'].append(train_accuracy)
    results['val_acc'].append(valid_accuracy)
    results['test_acc'].append(test_accuracy)

def main(model, data_generators, device, lr=0.003, args=None):
    opt = optim.Adam(model.task_model.parameters(), lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    results = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
        'test_acc': [],
        'test_loss': [],

    }

    for iteration in range(args['iterations']):        
        train_error, train_accuracy = train_normal(data_generators['train'], model, loss, opt, [],device)
        addResults(model, data_generators, results, iteration, train_error, train_accuracy, device)

    if args['save_model']:
        name_file = '{}/{}_{}_{}'.format(base_path,str(time.time()),args['algorithm'], args['dataset'])
        saveValues(name_file, results, model, args)
        
if __name__ == '__main__':
    parser = getArguments()
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = TaskModel(os.path.join('./results', args.load_model), 0.5, device).to(device)
    model.setLinear(0, 10)
    data_generators = getDataset(args.dataset)

    main(model,
         data_generators,
         device=device,
         lr=args.lr,
         args=vars(parser.parse_args()))