import copy

import torch
from torch import optim



def getOptimizer(shared, net, lr, task_id):
    params = []

    for p in net.private.parameters():
        params.append(p)

    for p in net.head[task_id].parameters():
        params.append(p)

    if shared:
        for p in net.shared.parameters():
            params.append(p)

    return optim.SGD(params, lr)

def train_dataset(net, opti, criterion, dataloader, epochs, task_id, device):
    net.train()
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

def train_batch(net, opti, criterion, batch, inner_loop, task_id, device, save_grad=None):
    running_loss = 0.0
    net.train()
    inputs = batch[0].to(device)
    labels = batch[1].to(device)
    for _ in range(inner_loop):
        outs = net(inputs.clone(), inputs.clone(), task_id)
        _, preds = torch.max(outs, 1)

        l = criterion(outs, labels)
        l.backward()

        if type(save_grad) == dict:
            for name, param in net.named_parameters():
                if name not in save_grad and param.grad is not None:
                    save_grad[name] = param.grad
                if param.grad is not None:
                    save_grad[name] += param.grad
        else:
            opti.step()
            opti.zero_grad()

        # print(l.item())
        running_loss += l.item()
        running_corrects = preds.eq(labels.view_as(preds)).sum().item()

    return running_loss/inner_loop

def train(args, net, task_id, dataloader, criterion, device):
    opti_total = getOptimizer(True, net, args.lr_out, task_id)

    for i in range(args.out_epochs):
        save_grads = {}
        loss_mini_task = 0.0
        total_loss = 0.0
        iter_data = iter(dataloader['train'])
        for k in range(args.mini_tasks):
            t_net = copy.deepcopy(net)
            opti_priv = getOptimizer(False, t_net, args.lr_inner, task_id)

            batch = next(iter_data)
            loss_mini_task += train_batch(t_net, opti_priv, criterion, batch, args.inner_loop, task_id, device) 
            
            batch = next(iter_data)
            total_loss += train_batch(t_net, None, criterion, batch, 1, task_id, device, save_grads) 

        if args.mini_tasks > 0:
            print("Train: Total loss: {} \t Mini Task Loss: {}".format(total_loss/args.mini_tasks,loss_mini_task/args.mini_tasks))

        for n, p in net.named_parameters():
            if n in save_grads and save_grads[n] is not None:   
                p.grad = save_grads[n]/args.mini_tasks
            else:
                p.grad = None
        opti_total.step()
        opti_total.zero_grad()

        if i % args.val_iter == 0:
            opti_priv = getOptimizer(False, net, args.lr_inner, task_id)
            res = train_dataset(net, opti_priv, criterion, dataloader['train'], args.pri_epochs, task_id, device)

            res = test(net, task_id, dataloader['valid'], criterion, device)
            print("Validation: Total loss: {} \t Accuracy: {}".format(res[1],res[0]))

    opti_priv = getOptimizer(False, net, args.lr_inner, task_id)
    res = train_dataset(net, opti_priv, criterion, dataloader['train'], args.pri_epochs, task_id, device)
    
    res = test(net, task_id, dataloader['valid'], criterion, device)
    print("Validation: Total loss: {} \t Accuracy: {}".format(res[1],res[0]))

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