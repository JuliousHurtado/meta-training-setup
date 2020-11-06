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
from approach import train, test, trainAll, prueba, prueba2

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
    for t,ncla in args.taskcla:
        print('*'*150)
        dataset = dataloader.get(t)
        print(' '*75, 'Dataset {:2d} ({:s})'.format(t+1,dataset[t]['name']))
        print('*'*150)

        if args.experiment == 'multidatasets':
            args.lr_task = dataloader.lrs[t][1]
        # Train
        #if args.train_first and (t == 0 or not (args.use_share and args.use_private)):
        #    res_task = trainAll(args, net, t, dataset[t], criterion, device)
        #else:
        #    res_task = train(args, net, t, dataset[t], criterion, device)
        
        res_task = prueba2(args, net, t, dataset[t], criterion, device)
        total_res[t] = res_task
        print('-'*150)
        print()

        for u in range(t+1):
            test_res = test(net, u, dataset[u]['test'], criterion, device)

            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u+1, dataset[u]['name'],
                                                                                          test_res[1],
                                                                                          test_res[0]))

            acc[t, u] = test_res[0]
            lss[t, u] = test_res[1]

    avg_acc, gem_bwt = utils.print_log_acc_bwt(args.taskcla, acc, lss, output_path=args.checkpoint, run_id=run_id)

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
    flags =  parser.parse_args()
    args = OmegaConf.load(flags.config)

    # for m_task in [1,5,10,20,30,40]:
    #     for i_loop in [1,5,10,20,35,50]:
    #         args.mini_tasks = m_task
    #         args.inner_loop = i_loop
    main(args)