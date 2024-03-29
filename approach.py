import copy
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch import optim

from utils import init_grads_out, get_diff_weights, set_grads, printSum


def train_representation(args, net, dataloader, task_id, criterion, device):
    params = []

    for p in net.private.conv[task_id].parameters():
        params.append(p)

    clfs = torch.nn.Linear(net.private.num_ftrs, net.taskcla[task_id][1]).to(device)
    for p in clfs.parameters():
        params.append(p)

    opti = optim.SGD(params, args.lr_task, weight_decay=0.01, momentum=0.9)

    for e in range(args.feats_epochs):
        correct, loss = 0.0, 0.0
        total = 0
        for i, batch in enumerate(dataloader['train']):
            opti.zero_grad()
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            outs = net.private.conv[task_id](inputs)
            outs = clfs(outs)
            _, preds = outs.max(1)

            l = criterion(outs, labels)
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.private.conv[task_id].parameters(),0.5)
            torch.nn.utils.clip_grad_norm_(clfs.parameters(),0.5)

            opti.step()

            correct += preds.eq(labels.view_as(preds)).sum().item()
            loss += l.item()

            total += inputs.size(0)

    correct_test = 0.0
    total_test = 0
    for i, batch in enumerate(dataloader['valid']):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        outs = net.private.conv[task_id](inputs)
        outs = clfs(outs)
        _, preds = outs.max(1)
        correct_test += preds.eq(labels.view_as(preds)).sum().item()
        total_test += inputs.size(0)

    print("Feature Training: Train loss: {:.4f} \t Acc Train: {:.4f} \t Acc Val: {:.4f}".format(loss/len(dataloader), correct/total, correct_test/total_test))

    for p in net.private.conv[task_id].parameters():
        p.requires_grad = False

def test(net, task_id, dataloader, criterion, device, task_pri = None):
    net.eval()
    correct, loss = 0.0, 0.0
    total = 0
    for i, batch in enumerate(dataloader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        inputs_feats = batch[2].to(device)

        outs, _ = net(inputs, task_id, inputs_feats, task_pri=task_pri)
        _, preds = outs.max(1)

        correct += preds.eq(labels.view_as(preds)).sum().item()

        l = criterion(outs, labels) 
        loss += l.item()

        total += inputs.size(0)
    net.train()
    return correct/total, loss/len(dataloader)

def train_meta_batch(net, opti, criterion, batch, inner_loop, task_id, task_pri, device):
    running_loss = 0.0
    net.train()
    inputs = batch[0].to(device)
    labels = batch[1].to(device)
    inputs_feats = batch[2].to(device)

    if opti is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opti,factor=0.5, patience=5, threshold=0.001)

    for _ in range(inner_loop):
        if opti is not None:
            opti.zero_grad()

        outs, _ = net(inputs, task_id, inputs_feats, True, task_pri)
        _, preds = torch.max(outs, 1)

        correct = preds.eq(labels.clone().view_as(preds)).sum().item()

        if criterion is not None:
            l = criterion(outs, labels.clone())
            l.backward()
            # torch.nn.utils.clip_grad_norm_(net.shared.parameters(),0.5)
            # torch.nn.utils.clip_grad_norm_(net.shared_clf.parameters(),0.5)
            running_loss = l.item()

        if opti is not None:
            opti.step()
            scheduler.step(running_loss)

    return running_loss, correct/inputs.size(0)

def meta_training(args, net, loader, task_id, opti_shared, criterion, device):
    grads_acc = {'grads': [], 'acc': []}
    loss_mini_task = 0.0
    total_loss = 0.0
    net#.to('cpu')
    for k in range(args.mini_tasks):
        t_net = copy.deepcopy(net).to(device)
        t_net.shared_clf = torch.nn.Linear(net.latent_dim, net.taskcla[task_id][1]).to(device)

        params = []
        for p in t_net.shared.parameters():
            params.append(p)
        for p in t_net.shared_clf.parameters():
            params.append(p)
        opti_shared_task = optim.SGD(params, args.lr_meta, weight_decay=0.1, momentum=0.9) #

        try:
            batch = next(iter_data_train)
        except:
            iter_data_train = iter(loader)
            batch = next(iter_data_train)

        batch_task_id = task_id
        temp_loss, acc_mini = train_meta_batch(t_net, opti_shared_task, criterion, batch, args.inner_loop, task_id, batch_task_id, device)

        loss_mini_task += temp_loss
        grads_acc['grads'].append(get_diff_weights(net, t_net))
       
        _, acc_mini= train_meta_batch(t_net, None, None, batch, 1, task_id, batch_task_id, device)
        grads_acc['acc'].append(acc_mini)

    # print(grads_acc['acc'])
    net.to(device)
    save_grads = init_grads_out(net)
    weights = torch.tensor(grads_acc['acc'])/sum(grads_acc['acc'])
    
    for i, g in enumerate(grads_acc['grads']):
        for n in g:
            save_grads[n] += g[n].to(device)*(weights[i])/(args.mini_tasks) # 

    set_grads(net, save_grads, task_id)

    opti_shared.step()
    opti_shared.zero_grad()

    return np.mean(grads_acc['acc']), loss_mini_task/args.mini_tasks

def traditional_training(args, net, loader_train, val_loader, task_id, opti_shared, criterion, device):
    val_acc = []
    for e in range(args.feats_epochs):
        correct = 0.0
        total = 0.0
        loss = 0.0
        for i, batch in enumerate(loader_train):
            opti_shared.zero_grad()
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            inputs_feats = batch[2].to(device)

            outs, _  = net(inputs, task_id, inputs_feats)
            _, preds = outs.max(1)
            l = criterion(outs, labels)
            l.backward()

            opti_shared.step()

            correct += preds.eq(labels.clone().view_as(preds)).sum().item()
            total += inputs.size(0)
            loss += l.item()

    #print("[{}|{}]Pre Acc: {:.4f}".format(e+1,args.feats_epochs,correct/total))
    return correct/total, loss/len(loader_train), val_acc

def training_procedure(args, net, task_id, dataloader, criterion, device):
    results_val = []

    # Train input representation
    if args.train_f_representation:
        if args.use_one_representation:
            if args.random_f:
                mask_lr = args.lr_task
            elif task_id == 0 and not args.resnet18:
                train_representation(args, net, dataloader, task_id, criterion, device)
                mask_lr = args.lr_task
            else:
                mask_lr = args.lr_task*0.1
        else:
            if not args.only_shared:
                if not args.resnet18:
                    train_representation(args, net, dataloader, task_id, criterion, device)
                    mask_lr = args.lr_task
                else:
                    if args.use_pca:
                        net.private.train_pca(dataloader['train'], task_id, device)
                    mask_lr = args.lr_task*0.1
            else:
                mask_lr = args.lr_task*0.1
    else:
        mask_lr = args.lr_task*0.01

    # Train shared weights in a traditional way only in first task
    if task_id == 0 and args.pre_train_shared:
        params = []
        for p in net.shared.parameters():
            params.append(p)
        for p in net.head[task_id].parameters():
            params.append(p)
        opti_shared_task = optim.SGD(params, args.lr_task, weight_decay=0.01, momentum=0.9)
        net.only_shared = True
        acc_train, loss_train, _ = traditional_training(args, net, dataloader['train'], dataloader['valid'], task_id, opti_shared_task, criterion, device)
        net.only_shared = args.only_shared
        acc_valid, _ = test(net, task_id, dataloader['valid'], criterion, device)
        print("Train Shared: Train loss: {:.4f} \t Acc Train: {:.4f} \t Acc Val: {:.4f}".format(loss_train, acc_train, acc_valid))


    # Learning to Reuse previous knowledge, training mask and classifier
    if not args.only_shared:
        params = []
        for p in net.private.linear[task_id].parameters():
            params.append(p)
        for p in net.head[task_id].parameters():
            params.append(p)
        opti_shared_mask = optim.SGD(params, mask_lr, weight_decay=0.01, momentum=0.9)
        acc_train, loss_train, hist_val = traditional_training(args, net, dataloader['train'], dataloader['valid'], task_id, opti_shared_mask, criterion, device)
        acc_valid, _ = test(net, task_id, dataloader['valid'], criterion, device)
        results_val.append(hist_val)
        print("Mask Training: Train loss: {:.4f} \t Acc Train: {:.4f} \t Acc Val: {:.4f}".format(loss_train, acc_train, acc_valid))



    # Training Share weights, via meta training or in a traditional way
    net.shared_clf = torch.nn.Linear(net.latent_dim, net.taskcla[task_id][1]).to(device)
    params = []
    for p in net.shared.parameters():
        params.append(p)
    if not args.use_meta:
        args.meta_epochs = 1
        for p in net.head[task_id].parameters():
            params.append(p)
    for e in range(args.meta_epochs):
        if args.use_meta:
            opti_shared = optim.SGD(params, args.lr_meta*0.1, weight_decay=0.1) #
            meta_acc, meta_loss = meta_training(args, net, dataloader['train'], task_id, opti_shared, criterion, device)
        else:
            opti_shared = optim.SGD(params, args.lr_meta*0.1, weight_decay=0.1) # 
            meta_acc, meta_loss, hist_val = traditional_training(args, net, dataloader['train'], dataloader['valid'], task_id, opti_shared, criterion, device)
            results_val.append(hist_val)
        acc_valid, _ = test(net, task_id, dataloader['valid'], criterion, device)
        print("[{}|{}]Meta Acc: {:.4f}\t Loss: {:.4f} Test Acc: {:.4f}\t".format(e+1,args.meta_epochs,meta_acc, meta_loss, acc_valid))



    # Consolidating Knowledge
    params = []
    if not args.only_shared:
        for p in net.private.linear[task_id].parameters():
            params.append(p)
    for p in net.head[task_id].parameters():
        params.append(p)
    opti_shared_mask = optim.SGD(params, mask_lr, weight_decay=0.01, momentum=0.9)
    acc_train, loss_train, hist_val = traditional_training(args, net, dataloader['train'], dataloader['valid'], task_id, opti_shared_mask, criterion, device)
    acc_valid, _ = test(net, task_id, dataloader['valid'], criterion, device)
    results_val.append(hist_val)
    print("Final Training: Train loss: {:.4f} \t Acc Train: {:.4f} \t Acc Val: {:.4f}".format(loss_train, acc_train, acc_valid))
