#!/usr/bin/env python3
import random
import os
import copy

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.datasets import ImageFolder, SVHN, CIFAR10
from torchvision import transforms
from torch.nn import functional as F

import learn2learn as l2l

from models.task_model import TaskModel
from utils import getArguments

from models.l2l_models import MiniImagenetCNN

base_path = 'results'

def test_normal(model, data_loader, device):
    #model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = input.to(device), target.long().to(device)
        o = model[0].forwardNoHead(input)
        output = model[1].forwardOnlyHead(o)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.item() / len(data_loader.dataset)

def getDataset(name_dataset):
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

    return generators

def loadModel(args, file_name, file_head, device):
    model_body = TaskModel(os.path.join('./results', args.load_model), args.percentage_new_filter, args.split_batch, device, not args.use_load_model).to(device)
    model_body.setLinear(0, 10)
    
    model_head = TaskModel(os.path.join('./results', args.load_model), args.percentage_new_filter, args.split_batch, device, not args.use_load_model).to(device)
    model_head.setLinear(0, 10)

    checkpoint1 = torch.load(file_name, map_location=device)
    model_body.load_state_dict(checkpoint1['checkpoint'])

    checkpoint2 = torch.load(file_head, map_location=device)
    model_head.load_state_dict(checkpoint2['checkpoint'])

    return model_body, model_head

def main(model, data_generators, device, args):
    loss = nn.CrossEntropyLoss(reduction='mean')

    tests = []
    tests.append(test_normal(model, data_generators['test'], device))

    print(args)
    print(np.mean(tests))


if __name__ == '__main__':
    parser = getArguments()
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if use_cuda:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    model = loadModel(args, args.load_model, args.load_head, device)
    data_generators = getDataset(args.dataset)

    main(model,
         data_generators,
         device=device,
         args=vars(parser.parse_args()))