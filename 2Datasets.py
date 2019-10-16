import os
import random
import argparse

import numpy as np
import torch as th
from PIL.Image import LANCZOS

from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data.datasets.full_omniglot import FullOmniglot
from model.omniglot_cnn import OmniglotCNN

from methods.maml import MAML
from methods.meta_sgd import MetaSGD

from copy import deepcopy

import learn2learn as l2l

device = th.device("cuda" if th.cuda.is_available() else "cpu")

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(adaptation_data, evaluation_data, learner, loss, adaptation_steps):
    if args['algorithm'] != 'sgd':
        learner.meta_train(adaptation_data, evaluation_data, loss)

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

def saveValues(name_file, acc, loss, args):
    th.save({
            'acc': acc,
            'loss': loss,
            'args': args
            }, name_file)

def getMetaAlgorithm(args, model):
    if args['algorithm'] == 'maml':
        meta_model = MAML(model, lr=args['fast_lr'], first_order=args['first_order'])
    elif args['algorithm'] == 'meta-sgd':
        meta_model = MetaSGD(model, lr=args['fast_lr'], first_order=args['first_order'])
    else:
        meta_model = model

    return meta_model

def main(args):
    # Create Datasets
    generators = {'mini-imagenet': None, 'omniglot': None}

    generators['mini-imagenet'] = getDatasets('mini-imagenet', args['ways'])
    generators['omniglot'] = getDatasets('omniglot', args['ways'])
    
    # Create model
    model = OmniglotCNN(args['ways'])
    model.to(device)

    meta_model = getMetaAlgorithm(args, model)
    
    opt = optim.Adam(meta_model.parameters(), args['meta_lr'])
    loss = nn.CrossEntropyLoss(size_average=True, reduction='mean')

    results = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
        'test_acc': [],
        'test_loss': [],

    }
    for dataset in ['omniglot']: #'mini-imagenet', 
        train_generator = generators[dataset][0]
        valid_generator = generators[dataset][1]
        test_generator = generators[dataset][2]
        
        for iteration in range(args['num_iterations']):
            opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            for task in range(args['meta_batch_size']):
                # Compute meta-training loss
                learner = meta_model.clone()
                adaptation_data = train_generator.sample(shots=args['shots'])
                evaluation_data = train_generator.sample(shots=args['shots'],
                                                         task=adaptation_data.sampled_task)
                evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                                   evaluation_data,
                                                                   learner,
                                                                   loss,
                                                                   args['adaptation_steps'])
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = meta_model.clone()
                adaptation_data = valid_generator.sample(shots=args['shots'])
                evaluation_data = valid_generator.sample(shots=args['shots'],
                                                         task=adaptation_data.sampled_task)
                evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                                   evaluation_data,
                                                                   learner,
                                                                   loss,
                                                                   args['adaptation_steps'])
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            err = []
            acc = []
            for dataset2 in ['mini-imagenet', 'omniglot']:
                test_generator = generators[dataset2][2]
                # Compute meta-testing loss
                learner = meta_model.clone()
                adaptation_data = test_generator.sample(shots=args['shots'])
                evaluation_data = test_generator.sample(shots=args['shots'],
                                                            task=adaptation_data.sampled_task)
                evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                                       evaluation_data,
                                                                       learner,
                                                                       loss,
                                                                       args['adaptation_steps'])
                err.append(evaluation_error.item())
                acc.append(evaluation_accuracy.item())

            # Print some metrics
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / args['meta_batch_size'])
            print('Meta Train Accuracy', meta_train_accuracy / args['meta_batch_size'])
            results['train_loss'].append(meta_train_error / args['meta_batch_size'])
            results['train_acc'].append(meta_train_accuracy / args['meta_batch_size'])

            print('Meta Valid Error', meta_valid_error / args['meta_batch_size'])
            print('Meta Valid Accuracy', meta_valid_accuracy / args['meta_batch_size'])
            results['val_loss'].append(meta_valid_error / args['meta_batch_size'])
            results['val_acc'].append(meta_valid_accuracy / args['meta_batch_size'])

            results['test_loss'].append(err)
            results['test_acc'].append(acc)

            # Average the accumulated gradients and optimize
            for p in meta_model.parameters():
                p.grad.data.mul_(1.0 / args['meta_batch_size'])
            opt.step()

    file_path = 'results/2datasets_{}_{}_{}_{}.pth'.format(args['algorithm'], args['shots'], args['ways'], args['first_order'])
    saveValues(file_path,results['test_acc'],results['test_loss'], args)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #Global
    parser.add_argument('--ways', default=5, type=int)
    parser.add_argument('--shots', default=5, type=int)
    parser.add_argument('--meta_lr', default=0.003, type=float)
    parser.add_argument('--fast_lr', default=0.5, type=float)
    parser.add_argument('--meta_batch_size', default=32, type=int)
    parser.add_argument('--adaptation_steps', default=5, type=int)
    parser.add_argument('--num_iterations', default=6000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--algorithm', choices=['maml', 'meta-sgd','sgd'], type=str)

    #MAML
    parser.add_argument('--first_order', default=False, type=str2bool)

    # ProtoNet
    parser.add_argument('--distance', default='l2')
    args = parser.parse_args()

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    th.manual_seed(args['seed'])
    if device == 'cuda':
        th.cuda.manual_seed(args['seed'])

    main(args)
