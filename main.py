import os,argparse,time
import numpy as np
from omegaconf import OmegaConf
import time

import torch

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import utils

from models.conv import Net
from models.hat import HatNet
from approach import test, training_procedure
from utils import getMasks

def run(args, run_id):
    # Args -- Experiment
    if args.experiment=='pmnist':
        from dataloaders import pmnist as datagenerator
    elif args.experiment=='mnist5':
        from dataloaders import mnist5 as datagenerator
    elif args.experiment=='cifar100':
        from dataloaders import cifar100 as datagenerator
    elif args.experiment=='miniimagenet':
        from dataloaders import miniimagenet as datagenerator
    elif args.experiment=='multidatasets':
        from dataloaders import mulitidatasets as datagenerator
    elif args.experiment=='imagenet':
        from dataloaders import imagenet as datagenerator
    else:
        raise NotImplementedError

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader
    print('Instantiate data generators and model...')
    dataloader = datagenerator.DatasetGen(args)
    args.taskcla, args.inputsize = dataloader.taskcla, dataloader.inputsize
    if args.experiment == 'multidatasets': args.lrs = dataloader.lrs

    # Model
    net = Net(args, device)
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    net.print_model_size()

    acc=np.zeros((len(args.taskcla),len(args.taskcla)),dtype=np.float32)
    lss=np.zeros((len(args.taskcla),len(args.taskcla)),dtype=np.float32)

    total_res = {}
    change = { 0: {} }

    for n,p in net.shared.named_parameters():
        change[0][n] = p.to('cpu')


    masks = {'train': {}, 'test': {}}
    feats = {'train': {}, 'test': {}}
    args.train_f_representation = True
    # args.pre_train_shared = True
    args.train_meta = True
    for _ in range(args.num_iter):
        for t,ncla in args.taskcla:
            print('*'*150)
            dataset = dataloader.get(t)
            print(' '*75, 'Dataset {:2d} ({:s})'.format(t+1,dataset[t]['name']))
            print('*'*150)

            if args.experiment == 'multidatasets':
                args.lr_task = dataloader.lrs[t][1]

            res_task = training_procedure(args, net, t, dataset[t], criterion, device)
            total_res[t] = res_task
            print('-'*150)
            print()

            if args.get_masks:
                change[t+1] = {}
                for n,p in net.shared.named_parameters():
                    change[t+1][n] = p.to('cpu') - change[0][n]

                masks['train'][t] = getMasks(net, t, dataset[t]['train'], device)
            #     feats['train'][t] = get_feature(net, t, dataset[t]['train'], device)

            for u in range(t+1):
                if args.use_last_pri:
                    test_res = test(net, u, dataset[u]['test'], criterion, device, t)
                else:
                    test_res = test(net, u, dataset[u]['test'], criterion, device)
                print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u+1, dataset[u]['name'],
                                                                                            test_res[1],
                                                                                            test_res[0]))

                acc[t, u] = test_res[0]
                lss[t, u] = test_res[1]
            
            if args.save_model:
                torch.save({
                        'args': args,
                        'checkpoint': net.state_dict()
                        }, 'models/{}_use_meta_{}_only_share_{}_task_{}.pth'.format(args.experiment, args.use_meta, args.only_shared, t))

        avg_acc, gem_bwt = utils.print_log_acc_bwt(args.taskcla, acc, lss, output_path=args.checkpoint, run_id=run_id)
        args.train_f_representation = False
        args.pre_train_shared = False
        args.train_meta = False

    # for t1,ncla in args.taskcla:
    #     masks['test'][t1] = {}
    #     feats['test'][t1] = {}
    #     for t2,ncla in args.taskcla:
    #         masks['test'][t1][t2] = getMasks(net, t2, dataset[t1]['test'], device)
    #         feats['test'][t1][t2] = get_feature(net, t2, dataset[t1]['test'], device)

    if args.get_masks:
        torch.save({ 'change_param': change, 'mean_mask': masks, 'args': args }, 
                'masks/{}_{}_{}_{}_{}_{}.pth'.format(args.experiment, run_id, args.meta_epochs, args.resnet18, args.use_meta, args.only_shared))

    # avg_acc, gem_bwt = utils.print_log_acc_bwt(args.taskcla, acc, lss, output_path=args.checkpoint, run_id=run_id)
    return avg_acc, gem_bwt, total_res

def main(args):
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    tstart=time.time()
    accuracies, forgetting, results = [], [], []
    for n in range(args.num_runs):
        args.seed += n 
        args.output = 'results/{}_{}_tasks_seed_{}.pth'.format(args.experiment_name, args.ntasks, args.seed)
        print ("args.output: ", args.output)
        
        print (" >>>> Run #", n)
        acc, bwt, total_res = run(args, n)
        accuracies.append(acc)
        forgetting.append(bwt)
        results.append(total_res)

    print('*' * 100)
    print ("Average over {} runs: ".format(args.num_runs))
    print ('AVG ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean(), np.array(accuracies).std()))
    print ('AVG BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean(), np.array(forgetting).std()))

    print ("All Done! ")
    print('[Elapsed time = {:.1f} min]'.format((time.time()-tstart)/(60)))
    utils.print_time()

    torch.save({
            'results': results,
            'acc': accuracies,
            'bwt': bwt,
            'args': args,
            }, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Continual Learning...')
    # Load the config file
    parser.add_argument('--config', type=str, default='./configs/config_mnist5.yml')
    parser.add_argument('--mini-tasks', type=int, default=-1)
    parser.add_argument('--inner-loop', type=int, default=-1)
    parser.add_argument('--feats-epochs', type=int, default=50)
    parser.add_argument('--meta-epochs', type=int, default=10)

    parser.add_argument('--num-masks', type=int, default=-1)
    parser.add_argument('--dist-masks', type=str, default='')
    parser.add_argument('--mask-dist-p', type=int, default=-1)
    parser.add_argument('--ntasks', type=int, default=-1)

    parser.add_argument('--pre-train-shared', type=int, default=1)
    parser.add_argument('--random-f', type=int, default=0)
    parser.add_argument('--num-iter', type=int, default=1)

    parser.add_argument('--only-shared', type=int, default=0)
    parser.add_argument('--save-model', type=int, default=0)
    parser.add_argument('--use-meta', type=int, default=1)
    parser.add_argument('--use-relu', type=int, default=1)
    parser.add_argument('--test-every-epoch', type=int, default=0)

    flags =  parser.parse_args()
    args = OmegaConf.load(flags.config)

    args.feats_epochs = flags.feats_epochs
    args.num_iter = flags.num_iter

    if flags.mini_tasks >= 0:
        args.mini_tasks = flags.mini_tasks
    if flags.inner_loop >= 0:
        args.inner_loop = flags.inner_loop
    
    if flags.pre_train_shared == 0:
        args.pre_train_shared = False
    if flags.random_f == 1:
        args.random_f = True
        args.use_one_representation = True

    if flags.only_shared == 1:
        args.only_shared = True
    if flags.use_meta == 0:
        args.use_meta = False

    if flags.save_model == 1:
        args.save_model = True

    if flags.use_relu == 0:
        args.use_relu = False
    
    if flags.test_every_epoch == 1:
        args.test_every_epoch = True

    args.meta_epochs = flags.meta_epochs


    main(args)

