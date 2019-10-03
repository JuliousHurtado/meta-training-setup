import os
import random

import numpy as np
import torch as th
from PIL.Image import LANCZOS

from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data.datasets.full_omniglot import FullOmniglot
from model.omniglot_cnn import OmniglotCNN

from copy import deepcopy

import learn2learn as l2l

device = th.device("cpu" if th.cuda.is_available() else "cpu")

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(adaptation_data, evaluation_data, learner, loss, adaptation_steps):
    for step in range(adaptation_steps):
        data = [d for d in adaptation_data]
        X = th.cat([d[0].unsqueeze(0) for d in data], dim=0).to(device)
        # X = th.cat([d[0] for d in data], dim=0).to(device)
        y = th.cat([th.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
        train_error = loss(learner(X), y)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)
    data = [d for d in evaluation_data]
    X = th.cat([d[0].unsqueeze(0) for d in data], dim=0).to(device)
    # X = th.cat([d[0] for d in data], dim=0).to(device)
    y = th.cat([th.tensor(d[1]).view(-1) for d in data], dim=0).to(device)
    predictions = learner(X)
    valid_error = loss(predictions, y)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, y)
    return valid_error, valid_accuracy

def getDatasets(dataset, ways):
    tasks_list = [20000, 1024, 1024]
    generators = {'train': None, 'validation': None, 'test': None}
    if dataset == 'mini-imagenet':
        for mode, tasks in zip(['train','validation','test'], tasks_list):
            dataset = l2l.vision.datasets.MiniImagenet(root='./data/data', mode=mode, 
                                transform = transforms.Compose([
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]))

            dataset = l2l.data.MetaDataset(dataset)
            generators[mode] = l2l.data.TaskGenerator(dataset=dataset, ways=ways, tasks=tasks)
    else:
        omniglot = FullOmniglot(root='./data/data',
                                                transform=transforms.Compose([
                                                    l2l.vision.transforms.RandomDiscreteRotation(
                                                        [0.0, 90.0, 180.0, 270.0]),
                                                    transforms.Resize(84, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=False, to_color = True)

        omniglot = l2l.data.MetaDataset(omniglot)
        classes = list(range(1623))
        random.shuffle(classes)
        generators['train'] = l2l.data.TaskGenerator(dataset=omniglot,
                                                 ways=ways,
                                                 classes=classes[:1100],
                                                 tasks=20000)
        generators['validation'] = l2l.data.TaskGenerator(dataset=omniglot,
                                                 ways=ways,
                                                 classes=classes[1100:1200],
                                                 tasks=1024)
        generators['test'] = l2l.data.TaskGenerator(dataset=omniglot,
                                                ways=ways,
                                                classes=classes[1200:],
                                                tasks=1024)

    return generators['train'], generators['validation'], generators['test']

def main(
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=4,
        adaptation_steps=1,
        num_iterations=6,
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed(seed)

    # Create Datasets
    generators = {'mini-imagenet': None, 'omniglot': None}

    generators['mini-imagenet'] = getDatasets('mini-imagenet', ways)
    generators['omniglot'] = getDatasets('omniglot', ways)
    
    # Create model
    model = OmniglotCNN(ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(size_average=True, reduction='mean')

    results = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
        'test_acc': [],
        'test_loss': [],

    }
    for dataset in ['mini-imagenet', 'omniglot']:
        train_generator = generators[dataset][0]
        valid_generator = generators[dataset][1]
        test_generator = generators[dataset][2]
        
        for iteration in range(num_iterations):
            opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            for task in range(meta_batch_size):
                # Compute meta-training loss
                learner = maml.clone()
                adaptation_data = train_generator.sample(shots=shots)
                evaluation_data = train_generator.sample(shots=shots,
                                                         task=adaptation_data.sampled_task)
                evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                                   evaluation_data,
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps,
                                                                   device)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = maml.clone()
                adaptation_data = valid_generator.sample(shots=shots)
                evaluation_data = valid_generator.sample(shots=shots,
                                                         task=adaptation_data.sampled_task)
                evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                                   evaluation_data,
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps,
                                                                   device)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            err = []
            acc = []
            for dataset2 in ['mini-imagenet', 'omniglot']:
                test_generator = generators[dataset2][2]
                # Compute meta-testing loss
                learner = maml.clone()
                adaptation_data = test_generator.sample(shots=shots)
                evaluation_data = test_generator.sample(shots=shots,
                                                            task=adaptation_data.sampled_task)
                evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                                       evaluation_data,
                                                                       learner,
                                                                       loss,
                                                                       adaptation_steps,
                                                                       device)
                err.append(evaluation_error.item())
                acc.append(evaluation_accuracy.item())

            # Print some metrics
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / meta_batch_size)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            results['train_loss'].append(meta_train_error / meta_batch_size)
            results['train_acc'].append(meta_train_accuracy / meta_batch_size)

            print('Meta Valid Error', meta_valid_error / meta_batch_size)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
            results['val_loss'].append(meta_valid_error / meta_batch_size)
            results['val_acc'].append(meta_valid_accuracy / meta_batch_size)

            results['test_loss'].append(err)
            results['test_acc'].append(acc)

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()

if __name__ == '__main__':
    main()