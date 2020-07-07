import torch
import random
import time
import copy
import numpy as np
from torch import nn
from torch import optim

from utils import getArguments, getModel, saveValues, getRegularizer, getMetaAlgorithm

from train_process.meta_training import trainingProcessMeta
from train_process.task_training import trainingProcessTask

from dataloader.multi_dataset import DatasetGen as multi_cls

def adjustModelTask(model, task, lr, num_cls):
    model.setLinearLayer(task, num_cls)
    
    return optim.SGD(model.parameters(), lr)

def main(args, data_generators, model, device):
    lr = args.lr

    meta_regs = []

    results = {}
    for i in range(data_generators.num_task):
        results[i] = {
            'meta_loss': [],
            'meta_acc': [],
            'train_acc': [],
            'train_loss': [],
            'valid_acc': [],
            'test_acc': [],
            'final_acc': []
        }

        task_dataloader = data_generators.get(i)
        num_cls_i = data_generators.taskcla[i]
        
        if args.meta_train:
            opti_meta = adjustModelTask(model, i, args.fast_lr, num_cls_i) 
            for e in range(args.meta_iterations):
                loss_meta, acc_meta = trainingProcessMeta(args, model, opti_meta, 
                        task_dataloader[i]['meta'], meta_regs, 
                        num_cls_i, device)
                
                results[i]['meta_loss'].append(loss_meta)
                results[i]['meta_acc'].append(acc_meta)
                print('Meta: Task {4} Epoch [{0}/{1}] \t Train Loss: {2:1.4f} \t Train Acc {3:3.2f} %'.format(e, args.meta_iterations, loss_meta, acc_meta*100, i+1))
        
        if args.meta_train:
            opti = optim.SGD(model.getLinearParameters(), lr)
        else:
            opti = optim.SGD(model.parameters(), lr)

        for e in range(args.epochs):
            loss_task, acc_task = trainingProcessTask(task_dataloader[i]['train'], 
                    model, opti, False, device) 
            
            results[i]['train_loss'].append(loss_task)
            results[i]['train_acc'].append(acc_task)            
            print('Task: Task {4} Epoch [{0}/{1}] \t Train Loss: {2:1.4f} \t Train Acc {3:3.2f} %'.format(e, args.epochs, loss_task, acc_task*100, i+1), flush=True)

            test_acc = test_normal(model, task_dataloader[i]['test'], device)
            results[i]['test_acc'].append(test_acc)   

        for j in range(i):
            model.setLinearLayer(j, -1)
            test_acc = test_normal(model, task_dataloader[j]['test'], device)
            results[j]['final'].append(test_acc)  
            
        if args.save_model:
            results[i]['parameters'] = model.state_dict()
            results[i]['linear'] = model.task[i].state_dict()
            name_file = '{}/basic_{}_Meta_{}'.format('results', args.dataset, args.meta_train)
            saveValues(name_file, results, model, args)

    for i in range(data_generators.num_task):
        print(results[i]['final_acc'])

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

    if args.dataset == 'multi':
        data_generators = multi_cls(args)
        args.dataset_order = data_generators.datasets_names

    cls_per_task = data_generators.taskcla

    model = getModel(cls_per_task[0], args.hidden_size , args.layers ,device)
    meta_model = getMetaAlgorithm(model, args.fast_lr, args.first_order)

    main(args, data_generators, meta_model, device)
