#!/usr/bin/env python3

import os
import copy

import numpy as np
import torch
from torch import nn
from torchvision import transforms

import learn2learn as l2l

from utils import getArguments, getModel, getAlgorithm, fast_adapt, test_normal

base_path = 'results'

def getDataset(name_dataset, ways, shots):
    generators = {}

    if name_dataset == 'MiniImagenet':
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

    elif name_dataset == 'cifar10':
        transform_data = transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset_train = CIFAR10('./data/', train=True, transform=transform_data, download=True)
        dataset_test = CIFAR10('./data/', train=False, transform=transform_data, download=True)

        generators['train'] = torch.utils.data.DataLoader(dataset_train, batch_size=64)
        generators['validation'] = torch.utils.data.DataLoader(dataset_train, batch_size=64)
        generators['test'] = torch.utils.data.DataLoader(dataset_test, batch_size=64)

    else:
        raise Exception('Dataset {} not supported'.format(name_dataset))

    return generators

def loadModel(file_name, model, file_head, model_head, device):
    checkpoint = torch.load(os.path.join(base_path,file_name), map_location=device)
    model.load_state_dict(checkpoint['checkpoint'])

    checkpoint = torch.load(os.path.join(base_path,file_head), map_location=device)
    model_head.load_state_dict(checkpoint['checkpoint'])

    model.linear = copy.deepcopy(model_head.linear)

    return model

def getModel(input_channels, ways = 5, device = 'cpu'):
    if input_channels == 1:
        return OmniglotCNN(ways).to(device)
    elif input_channels == 3:
        return MiniImagenetCNN(ways).to(device)
    else:
        raise Exception('Input Channels must be 1 or 3, not: {}'.format(input_channels))

def test_normal(model, data_loader, device):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = input.to(device), target.long().to(device)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.item() / len(data_loader.dataset)

def main(model, data_generators, ways, shots, device, adaptation_steps, args, fine_tuning):
    loss = nn.CrossEntropyLoss(reduction='mean')

    if fine_tuning:
        test_accuracy = test_normal(model, data_generators['test'], device)
    else:
        tests = []
        for iteration in range(100):
            learner = model.clone()
            batch = data_generators['test'].sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   [],
                                                                   loss
                                                                   adaptation_steps,
                                                                   shots,
                                                                   ways,
                                                                   device)

    print(args)
    print(np.mean(tests))


if __name__ == '__main__':
    parser = getArguments()
    args = parser.parse_args()

    fine_tuning = False
    if args.algorithm == 'FT':
        fine_tuning = True

    use_cuda = torch.cuda.is_available()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if use_cuda:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    model = getModel(args.input_channel, ways=args.init_ways, device=device)
    model_head = getModel(args.input_channel, ways=args.ways, device=device)
    model = loadModel(args.load_model, model, args.load_head, model_head, device)

    meta_model = getAlgorithm(args.algorithm, model, args.fast_lr, args.first_order, args.freeze_layer)
    
    data_generators = getDataset(args.dataset, args.ways, args.shots)

    main(meta_model,
         data_generators,
         ways=args.ways,
         shots=args.shots,
         device=device,
         adaptation_steps=args.fast_adaption_steps,
         args=vars(parser.parse_args()),
         fine_tuning=fine_tuning)