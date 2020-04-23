#!/usr/bin/env python3

import os
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import SVHN, CIFAR10

from models.task_model import TaskModel
from utils import saveValues, getArguments, train_normal, test_normal
from method.ewc import EWC

base_path = 'results'

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def get_sample(dataset, sample_size):
    sample_idx = random.sample(range(len(dataset)), sample_size)
    temp = []
    for img in dataset.data[sample_idx]:
        if dataset.transform:
            img = dataset.transform(Image.fromarray(img)).unsqueeze(0)
        temp.append(img)
    return temp

def getDataset(name_dataset, ewc=False):
    generators = {}
    transform_data = transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    bs = 64
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

    if ewc:
        dataset_train = CIFAR10('./data/', train=True, transform=transform_data, download=True)
        generators['sample'] = get_sample(dataset_train,300)

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

def loadModel(model, path_file, lr, device):
    checkpoint = torch.load(os.path.join(path_file), map_location=device)

    model.load_state_dict(checkpoint['checkpoint'])
    opt = optim.Adam(model.meta_model.linear.parameters(), lr)

    return model, opt

def main(model, data_generators, device, lr=0.003, args=None):
    if args['use_load_model']:
        model, opt = loadModel(model, args['load_model'], lr, device)

        ewc = None
        if args['use_ewc']:
            ewc = EWC(model, data_generators['sample'], args['ewc_importance'], model.getTaskParameters())
        
        if args['train_task_parameters'] or args['use_ewc']:
            opt = optim.Adam(model.getTaskParameters(), lr)

    else:
        opt = optim.Adam(model.getTaskParameters(), lr)
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
        train_error, train_accuracy = train_normal(data_generators['train'], model, loss, opt, [],device, ewc)
        addResults(model, data_generators, results, iteration, train_error, train_accuracy, device)

    if args['save_model']:
        name_file = '{}/{}_{}_{}_{}'.format(base_path,str(time.time()),str(args['percentage_new_filter']), str(args['split_batch']), args['dataset'])
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
    
    model = TaskModel(os.path.join('./results', args.load_model), args.percentage_new_filter, args.split_batch, device, not args.use_load_model).to(device)
    model.setLinear(0, 10)
    data_generators = getDataset(args.dataset, args.use_ewc)

    #print(model)
    main(model,
         data_generators,
         device=device,
         lr=args.lr,
         args=vars(parser.parse_args()))