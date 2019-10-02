import os
import random

import numpy as np
import torch as th
from PIL.Image import LANCZOS

from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data.datasets import full_omniglot

from copy import deepcopy

import learn2learn as l2l


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(adaptation_data, evaluation_data, learner, loss, adaptation_steps, device):
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
            print(dataset[14][0].size())
            print(dataset[14][0].sum())
            dataset = l2l.data.MetaDataset(dataset)
            generators[mode] = l2l.data.TaskGenerator(dataset=dataset, ways=ways, tasks=tasks)
    else:
        omniglot = full_omniglot.FullOmniglot(root='./data/data',
                                                transform=transforms.Compose([
                                                    l2l.vision.transforms.RandomDiscreteRotation(
                                                        [0.0, 90.0, 180.0, 270.0]),
                                                    transforms.Resize(84, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=False, to_color = True)
        print(omniglot[12][0].size())
        print(omniglot[12][0].sum())
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
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    device = th.device('cpu')
    if cuda:
        th.cuda.manual_seed(seed)
        device = th.device('cuda')

    # Create Datasets
    generators = {'mini-imagenet': None, 'omniglot': None}

    generators['mini-imagenet'] = getDatasets('mini-imagenet', ways)
    generators['omniglot'] = getDatasets('omniglot', ways)
    
    # Create model
    # model = l2l.vision.models.OmniglotCNN(ways)
    # model.to(device)
    # maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    # opt = optim.Adam(maml.parameters(), meta_lr)
    # loss = nn.CrossEntropyLoss(size_average=True, reduction='mean')

    # for iteration in range(num_iterations):
    #     opt.zero_grad()
    #     meta_train_error = 0.0
    #     meta_train_accuracy = 0.0
    #     meta_valid_error = 0.0
    #     meta_valid_accuracy = 0.0
    #     meta_test_error = 0.0
    #     meta_test_accuracy = 0.0
    #     for task in range(meta_batch_size):
    #         # Compute meta-training loss
    #         learner = maml.clone()
    #         adaptation_data = train_generator.sample(shots=shots)
    #         evaluation_data = train_generator.sample(shots=shots,
    #                                                  task=adaptation_data.sampled_task)
    #         evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
    #                                                            evaluation_data,
    #                                                            learner,
    #                                                            loss,
    #                                                            adaptation_steps,
    #                                                            device)
    #         evaluation_error.backward()
    #         meta_train_error += evaluation_error.item()
    #         meta_train_accuracy += evaluation_accuracy.item()

    #         # Compute meta-validation loss
    #         learner = maml.clone()
    #         adaptation_data = valid_generator.sample(shots=shots)
    #         evaluation_data = valid_generator.sample(shots=shots,
    #                                                  task=adaptation_data.sampled_task)
    #         evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
    #                                                            evaluation_data,
    #                                                            learner,
    #                                                            loss,
    #                                                            adaptation_steps,
    #                                                            device)
    #         meta_valid_error += evaluation_error.item()
    #         meta_valid_accuracy += evaluation_accuracy.item()

    #         # Compute meta-testing loss
    #         learner = maml.clone()
    #         adaptation_data = test_generator.sample(shots=shots)
    #         evaluation_data = test_generator.sample(shots=shots,
    #                                                 task=adaptation_data.sampled_task)
    #         evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
    #                                                            evaluation_data,
    #                                                            learner,
    #                                                            loss,
    #                                                            adaptation_steps,
    #                                                            device)
    #         meta_test_error += evaluation_error.item()
    #         meta_test_accuracy += evaluation_accuracy.item()

    #     # Print some metrics
    #     print('\n')
    #     print('Iteration', iteration)
    #     print('Meta Train Error', meta_train_error / meta_batch_size)
    #     print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
    #     print('Meta Valid Error', meta_valid_error / meta_batch_size)
    #     print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
    #     print('Meta Test Error', meta_test_error / meta_batch_size)
    #     print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)

    #     # Average the accumulated gradients and optimize
    #     for p in maml.parameters():
    #         p.grad.data.mul_(1.0 / meta_batch_size)
    #     opt.step()

if __name__ == '__main__':
    main()
