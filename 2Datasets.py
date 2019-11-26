import random
import argparse
import time

import numpy as np
import torch as th
from torch import nn
from torch import optim

from model.adam import Adam

from model.omniglot_cnn import OmniglotCNN
from model.resnet import resnet18

from utils import getDatasets, saveValues, str2bool, getMetaAlgorithm, getRandomDataset

from copy import deepcopy

device = th.device("cuda" if th.cuda.is_available() else "cpu")

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def fast_adapt(adaptation_data, evaluation_data, learner, loss, adaptation_steps):
    if args['algorithm'] == 'protonet':
        y_support = th.LongTensor(adaptation_data.label).to(device)
        valid_error, y_pred = learner.meta_train(adaptation_data, evaluation_data, loss)
        valid_accuracy = learner.categorical_accuracy(y_support, y_pred)
        return valid_error, valid_accuracy

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

def adaptationProcess(args, generator, learner, loss):
    adaptation_data = generator.sample(shots=args['shots'])
    evaluation_data = generator.sample(shots=args['shots'],task=adaptation_data.sampled_task)

    evaluation_error, evaluation_accuracy = fast_adapt(adaptation_data,
                                                                   evaluation_data,
                                                                   learner,
                                                                   loss,
                                                                   args['adaptation_steps'])
    return evaluation_error, evaluation_accuracy

def cloneModel(args, model):
    if args['algorithm'] in ['maml', 'meta-sgd', 'tmaml']:
        return model.clone()
    return model

def main(args):
    # Create Datasets
    generators = {'mini-imagenet': None, 'omniglot': None}
    
    # Create model
    print("Creating Model", flush=True)
    if True:
        model = resnet18(pretrained = args['pretrained'])
        model.createLineals(2, args['ways'])
    else:
        model = OmniglotCNN(args['ways'])
    model.to(device)

    print("Getting Meta Algorithm", flush=True)
    meta_model = getMetaAlgorithm(args, model, device)
    
    print("Obtaining optimizer", flush=True)
    opt = Adam(meta_model.parameters(), args['meta_lr'])
    if args['algorithm'] == 'protonet':
        loss = nn.NLLLoss()
    else:
        loss = nn.CrossEntropyLoss(reduction='mean')

    print("Reading datasets", flush=True)
    generators['mini-imagenet'] = getDatasets('omniglot', args['ways'])
    generators['omniglot'] = getDatasets('omniglot', args['ways'])

    #generators['mini-imagenet'] = getRandomDataset(args['ways'])
    #generators['omniglot'] = getRandomDataset(args['ways'])

    results = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
        'test_acc': [],
        'test_loss': [],

    }
    for i,dataset in enumerate(['mini-imagenet','omniglot']): # 
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
                meta_model.setLinear(i, device)
                learner = cloneModel(args, meta_model)

                evaluation_error, evaluation_accuracy = adaptationProcess(args, train_generator, learner, loss)
                #meta_model.printParam()
                
                evaluation_error.backward()
                if args['algorithm'] == 'tmaml':
                    meta_model.getGradients()

                if args['algorithm'] in ['sgd', 'protonet']:
                    opt.step()
                    opt.zero_grad()

                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = cloneModel(args, meta_model)
                evaluation_error, evaluation_accuracy = adaptationProcess(args, valid_generator, learner, loss)

                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            # Average the accumulated gradients and optimize
            #    meta_model.write_grads(valid_generator, opt, loss, args['shots'], args['meta_batch_size'])

            if args['algorithm'] not in ['sgd', 'protonet']:
                if args['algorithm'] == 'tmaml':
                    meta_model.setMask()
                for p in meta_model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(1.0 / args['meta_batch_size'])

                opt.step()
                #meta_model.printParam()

            if iteration % 20 == 0:
                err = []
                acc = []
                for j,dataset2 in enumerate(['mini-imagenet', 'omniglot']):
                    test_generator = generators[dataset2][2]
                    # Compute meta-testing loss
                    meta_model.setLinear(j, device)
                    learner = cloneModel(args, meta_model)
                    evaluation_error, evaluation_accuracy = adaptationProcess(args, test_generator, learner, loss)

                    err.append(evaluation_error.item())
                    acc.append(evaluation_accuracy.item())
                results['test_loss'].append(err)
                results['test_acc'].append(acc)

            # Print some metrics
            if iteration % 20 == 0:
                print('\n')
                print('Iteration', iteration)
                print('Meta Train Error', meta_train_error / args['meta_batch_size'])
                print('Meta Train Accuracy', meta_train_accuracy / args['meta_batch_size'])

                print('Meta Valid Error', meta_valid_error / args['meta_batch_size'])
                print('Meta Valid Accuracy', meta_valid_accuracy / args['meta_batch_size'], flush=True)
        
            results['train_loss'].append(meta_train_error / args['meta_batch_size'])
            results['train_acc'].append(meta_train_accuracy / args['meta_batch_size'])

            results['val_loss'].append(meta_valid_error / args['meta_batch_size'])
            results['val_acc'].append(meta_valid_accuracy / args['meta_batch_size'])

    file_path = 'results/2datasets_{}_{}_{}_{}_{}_{}.pth'.format(str(time.time()), args['algorithm'], 
                                                                args['shots'], args['ways'], 
                                                                args['first_order'], args['min_used'])
    saveValues(file_path, results, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #Global
    parser.add_argument('--ways', default=5, type=int)
    parser.add_argument('--shots', default=5, type=int)
    parser.add_argument('--meta_lr', default=0.003, type=float)
    parser.add_argument('--fast_lr', default=0.5, type=float)
    parser.add_argument('--meta_batch_size', default=32, type=int)
    parser.add_argument('--adaptation_steps', default=5, type=int)
    parser.add_argument('--num_iterations', default=20000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--freeze_block', default=1, type=int)
    parser.add_argument('--algorithm', choices=['maml', 'meta-sgd','sgd', 'protonet', 'tmaml', 'meta-resnet'], type=str)
    parser.add_argument('--pretrained', default=True, type=str2bool)

    #MAML
    parser.add_argument('--first_order', default=True, type=str2bool)

    #Transfer
    parser.add_argument('--min_used', default=0.0, type=float)

    # ProtoNet
    parser.add_argument('--distance', default='l2')
    args = vars(parser.parse_args())

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    th.manual_seed(args['seed'])
    if device == 'cuda':
        th.cuda.manual_seed(args['seed'])

    main(args)
