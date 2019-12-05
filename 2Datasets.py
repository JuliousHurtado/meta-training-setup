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

from utils import getDatasets, saveValues, str2bool, getMetaAlgorithm, getRandomDataset, getMetaTrainingSet

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
    if args['algorithm'] in ['maml', 'meta-sgd', 'meta-transfer']:
        return model.clone()
    return model

def train_process(args, meta_model, loss, opt, train_generator, valid_generator, task_ind):
    opt.zero_grad()
    meta_train_error = 0.0
    meta_train_accuracy = 0.0
    meta_valid_error = 0.0
    meta_valid_accuracy = 0.0

    for task in range(args['meta_batch_size']):
        # Compute meta-training loss
        meta_model.setLinear(task_ind, device)
        learner = cloneModel(args, meta_model)

        evaluation_error, evaluation_accuracy = adaptationProcess(args, train_generator, learner, loss)
        # meta_model.printParam()
                
        evaluation_error.backward()
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
        for p in meta_model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / args['meta_batch_size'])

        opt.step()
        #meta_model.printParam()

    return meta_train_error, meta_valid_error, meta_train_accuracy, meta_valid_accuracy

def test_process(args, meta_model, generators, loss):
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

    return err, acc

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
    generators['mini-imagenet'] = getDatasets('mini-imagenet', args['ways'])
    generators['omniglot'] = getDatasets('omniglot', args['ways'])

    #generators['mini-imagenet'] = getRandomDataset(args['ways'], False)
    #generators['omniglot'] = getRandomDataset(args['ways'], False)

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
        
        if args['meta_training']:
            for iteration in range(args['num_set_train']-1):
                meta_train_generator, meta_valid_generator, _ = getMetaTrainingSet(dataset, args['ways'], args['num_new_cls'])
                #meta_train_generator, meta_valid_generator, _ = getRandomDataset(args['ways'], args['meta_training'], args['num_new_cls'])

                for iteration in range(int(args['num_iterations']/args['num_set_train'])):
                    t_error, v_error, t_acc, v_acc = train_process(args, meta_model, loss, opt, meta_train_generator, meta_valid_generator, i)

            for iteration in range(args['num_iterations']):
                t_error, v_error, t_acc, v_acc = train_process(args, meta_model, loss, opt, train_generator, valid_generator, i)

                if iteration % 10 == 0:
                    err, acc = test_process(args, meta_model, generators, loss)
                    results['test_loss'].append(err)
                    results['test_acc'].append(acc)

                # Print some metrics
                if iteration % 50 == 0:
                    print('\n')
                    print('Iteration', iteration)
                    print('Meta Train Error', t_error / args['meta_batch_size'])
                    print('Meta Train Accuracy', t_acc / args['meta_batch_size'])

                    print('Meta Valid Error', v_error / args['meta_batch_size'])
                    print('Meta Valid Accuracy', v_acc / args['meta_batch_size'], flush=True)
            
                results['train_loss'].append(t_error / args['meta_batch_size'])
                results['train_acc'].append(t_acc / args['meta_batch_size'])

                results['val_loss'].append(v_error / args['meta_batch_size'])
                results['val_acc'].append(v_acc / args['meta_batch_size'])

        else:
            for iteration in range(args['num_iterations']):
                t_error, v_error, t_acc, v_acc = train_process(args, meta_model, loss, opt, train_generator, valid_generator, i)

                if iteration % 10 == 0:
                    err, acc = test_process(args, meta_model, generators, loss)
                    results['test_loss'].append(err)
                    results['test_acc'].append(acc)

                # Print some metrics
                if iteration % 50 == 0:
                    print('\n')
                    print('Iteration', iteration)
                    print('Meta Train Error', t_error / args['meta_batch_size'])
                    print('Meta Train Accuracy', t_acc / args['meta_batch_size'])

                    print('Meta Valid Error', v_error / args['meta_batch_size'])
                    print('Meta Valid Accuracy', v_acc / args['meta_batch_size'], flush=True)
            
                results['train_loss'].append(t_error / args['meta_batch_size'])
                results['train_acc'].append(t_acc / args['meta_batch_size'])

                results['val_loss'].append(v_error / args['meta_batch_size'])
                results['val_acc'].append(v_acc / args['meta_batch_size'])

    file_path = 'results/2datasets_{}_{}_{}_{}_{}.pth'.format(str(time.time()), args['algorithm'], 
                                                                args['shots'], args['ways'], 
                                                                args['first_order'])
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
    parser.add_argument('--num_iterations', default=500, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--algorithm', choices=['maml', 'meta-sgd','sgd', 'protonet', 'meta-transfer'], type=str)
    parser.add_argument('--pretrained', default=True, type=str2bool)

    #Meta Transfer
    parser.add_argument('--freeze_block', default=1, type=int)

    #Meta Train
    parser.add_argument('--meta_training', default=False, type=str2bool)
    parser.add_argument('--num_set_train', default=5, type=int)
    parser.add_argument('--num_new_cls', default=30, type=int)

    #MAML
    parser.add_argument('--first_order', default=True, type=str2bool)

    # ProtoNet
    parser.add_argument('--distance', default='l2')
    args = vars(parser.parse_args())

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    th.manual_seed(args['seed'])
    if device == 'cuda':
        th.cuda.manual_seed(args['seed'])

    main(args)
