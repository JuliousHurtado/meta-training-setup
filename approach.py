import copy
import numpy as np

import torch
from torch import optim



def getOptimizer(shared, net, lr, task_id):
    params = []

    for p in net.private.conv[task_id].parameters():
        params.append(p)

    for p in net.private.linear[task_id].parameters():
        params.append(p)

    for p in net.private.last_em[task_id].parameters():
        params.append(p)

    for p in net.head[task_id].parameters():
        params.append(p)

    if shared:
        for p in net.shared.parameters():
            params.append(p)

    return optim.SGD(params, lr, weight_decay=0.01, momentum=0.9)

def train_dataset(net, opti, criterion, dataloader, epochs, task_id, device):
    net.train()
    for e in range(epochs):
        correct, loss = 0.0, 0.0
        total = 0
        for i, batch in enumerate(dataloader):
            opti.zero_grad()
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            outs = net(inputs.clone(), inputs.clone(), task_id)
            _, preds = outs.max(1)

            l = criterion(outs, labels) 
            l.backward()

            opti.step()

            correct += preds.eq(labels.view_as(preds)).sum().item()
            loss += l.item()

            total += inputs.size(0)

    return correct/total, loss/len(dataloader)

def train_batch(net, opti, criterion, batch, inner_loop, task_id, device, patience_inner, save_grad=None):
    running_loss = 0.0
    best_loss = np.inf
    best_model = copy.deepcopy(net)
    patience = patience_inner
    factor = 1
    net.train()
    inputs = batch[0].to(device)
    labels = batch[1].to(device)
    # print("New batch")
    for _ in range(inner_loop):
        outs = net(inputs.clone(), inputs.clone(), task_id)
        _, preds = torch.max(outs, 1)

        l = criterion(outs, labels)
        l.backward()

        if type(save_grad) == dict:
            for name, param in net.named_parameters():
                if param.grad is not None:
                    if name not in save_grad:
                        save_grad[name] = param.grad
                    else:
                        save_grad[name] += param.grad
        else:
            opti.step()
            opti.zero_grad()

        running_loss += l.item()

        if l.item() < best_loss:
            best_loss = l.item()
            best_model = copy.deepcopy(net)
        else:
            patience -= 1
            if patience <= 0:
                factor *= 0.5
                for param_group in opti.param_groups:
                    param_group['lr'] = param_group['lr']*factor
                
        net.load_state_dict(copy.deepcopy(best_model).state_dict())

    return running_loss/inner_loop

def train_mini_task(args, net, dataloader, task_id, criterion, device):
    iter_data_train = iter(dataloader['train'])
    #iter_data_val = iter(dataloader['train'])
    
    save_grads = {}
    loss_mini_task = 0.0
    total_loss = 0.0

    for k in range(args.mini_tasks):
        t_net = copy.deepcopy(net)
        opti_priv = getOptimizer(False, t_net, args.lr_inner, task_id)

        try:
            batch = next(iter_data_train)
        except:
            iter_data_train = iter(dataloader['train'])
            batch = next(iter_data_train)

        loss_mini_task += train_batch(t_net, opti_priv, criterion, batch, args.inner_loop, task_id, device, args.lr_patience_inner) 

        try:
            batch = next(iter_data_val)
        except:
            iter_data_val = iter(dataloader['train'])
            batch = next(iter_data_val)

        total_loss += train_batch(t_net, None, criterion, batch, 1, task_id, device, args.lr_patience_inner, save_grads)

    return save_grads, total_loss, loss_mini_task

def train(args, net, task_id, dataloader, criterion, device):
    opti_total = getOptimizer(True, net, args.lr_out, task_id)
    scheduler_shar = optim.lr_scheduler.ReduceLROnPlateau(opti_total, mode='min', 
                    factor=0.5, patience=args.lr_patience, min_lr=1e-5, eps=1e-08)

    opti_priv = getOptimizer(False, net, args.lr_priv, task_id)
    scheduler_priv = optim.lr_scheduler.ReduceLROnPlateau(opti_priv, mode='min', 
                    factor=0.5, patience=args.lr_patience, min_lr=1e-5, eps=1e-08)

    best_loss = np.inf
    best_model = copy.deepcopy(net)

    results = {
        'meta_loss': [],
        'mini_loss': [],
        'val_loss': [],
        'val_acc': [],
        'train_loss': [],
        'train_acc': []
    }
    for i in range(args.out_epochs):
        if args.mini_tasks > 0:
            save_grads, total_loss, loss_mini_task = train_mini_task(args, net, dataloader, task_id, criterion, device)
            print("Train: Total loss: {} \t Mini Task Loss: {}".format(total_loss/args.mini_tasks,loss_mini_task/args.mini_tasks))
            results['meta_loss'].append(total_loss/args.mini_tasks)
            results['mini_loss'].append(loss_mini_task/args.mini_tasks)

            for n, p in net.named_parameters():
                if n in save_grads and save_grads[n] is not None:   
                    p.grad = save_grads[n]/args.mini_tasks
                else:
                    p.grad = None
            opti_total.step()
            opti_total.zero_grad()

            scheduler_priv.step(total_loss)

        if i % args.val_iter == 0:
            res_train = train_dataset(net, opti_priv, criterion, dataloader['train'], args.pri_epochs, task_id, device)
            results['train_loss'].append(res_train[1])
            results['train_acc'].append(res_train[0])

            res_test = test(net, task_id, dataloader['valid'], criterion, device)
            print("Tain Loss: {} \t Valid loss: {} \t Accuracy: {}".format(res_train[1],res_test[1],res_test[0]))
            results['val_loss'].append(res_test[1])
            results['val_acc'].append(res_test[0])

            if res_test[1] < best_loss:
                best_loss = res_test[1]
                best_model = copy.deepcopy(net)

            scheduler_priv.step(res_test[1])
            net.load_state_dict(copy.deepcopy(best_model).state_dict())

    res_train = train_dataset(net, opti_priv, criterion, dataloader['train'], args.pri_epochs, task_id, device)
    results['train_loss'].append(res_train[1])
    results['train_acc'].append(res_train[0])

    res_test = test(net, task_id, dataloader['valid'], criterion, device)
    print("Tain Loss: {} \t Valid loss: {} \t Accuracy: {}".format(res_train[1],res_test[1],res_test[0]))
    results['val_loss'].append(res_test[1])
    results['val_acc'].append(res_test[0])

    if res_test[1] < best_loss:
        best_model = copy.deepcopy(net)

    net.load_state_dict(copy.deepcopy(best_model).state_dict())

    return results

def trainAll(args, net, task_id, dataloader, criterion, device):
    net.train()
    opti_total = getOptimizer(True, net, args.lr_out, task_id)

    best_loss = np.inf
    best_model = copy.deepcopy(net)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opti_total, mode='min', 
                    factor=0.5, patience=args.lr_patience, min_lr=1e-5, eps=1e-08)

    results = {
        'meta_loss': [],
        'mini_loss': [],
        'val_loss': [],
        'val_acc': [],
        'train_loss': [],
        'train_acc': []
    }
    for i in range(args.out_epochs):
        res_train = train_dataset(net, opti_total, criterion, dataloader['train'], 1, task_id, device)
        results['train_loss'].append(res_train[1])
        results['train_acc'].append(res_train[0])

        res_test = test(net, task_id, dataloader['valid'], criterion, device)
        results['val_loss'].append(res_test[1])
        results['val_acc'].append(res_test[0])

        print("Tain Loss: {} \t Valid loss: {} \t Accuracy: {}".format(res_train[1],res_test[1],res_test[0]))

        if res_test[1] < best_loss:
            best_loss = res_test[1]
            best_model = copy.deepcopy(net)

        scheduler.step(res_test[1])
        net.load_state_dict(copy.deepcopy(best_model).state_dict())

    return results

def test(net, task_id, dataloader, criterion, device):
    net.eval()
    correct, loss = 0.0, 0.0
    total = 0
    for i, batch in enumerate(dataloader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        outs = net(inputs.clone(), inputs.clone(), task_id)
        _, preds = outs.max(1)

        correct += preds.eq(labels.view_as(preds)).sum().item()

        l = criterion(outs, labels) 
        loss += l.item()

        total += inputs.size(0)

    return correct/total, loss/len(dataloader)