import copy
import numpy as np

import torch
from torch import optim



def getOptimizer(shared, private, head, net, lr, task_id):
    params = []

    if private:
        try:
            for p in net.private.conv[task_id].parameters():
                params.append(p)
        except:
            pass

        for p in net.private.linear[task_id].parameters():
            params.append(p)

        try:
            for p in net.private.last_em[task_id].parameters():
                params.append(p)
        except:
            pass

    if head:
        for p in net.head[task_id].parameters():
            params.append(p)

    if shared:
        for p in net.shared.parameters():
            params.append(p)

    return optim.SGD(params, lr, weight_decay=0.01, momentum=0.9)

def init_grads_out(net):
    grads = {}

    for n,p in net.named_parameters():
        grads[n] = torch.zeros_like(p)

    return grads

def get_diff_weights(old_net, new_net):
    grads = {}

    for (_,p_new),(n,p_old) in zip(new_net.named_parameters(), old_net.named_parameters()):
        grads[n] = p_new - p_old

    return grads 

def print_sum_params(net, task_id):
    s = 0
    for n,p in net.shared.named_parameters():
        s += p.sum()
    print("Shared: ",s)

    s = 0
    try:
        for n,p in net.private.conv[task_id].named_parameters():
            s += p.sum()
    except:
        pass

    for n,p in net.private.linear[task_id].named_parameters():
        s += p.sum()

    try:
        for n,p in net.private.last_em[task_id].named_parameters():
            s += p.sum()
    except:
        pass
    print("Private: ",s)
    
    s = 0
    for n,p in net.head[task_id].named_parameters():
        s += p.sum()
    print("Head: ",s)

def set_grads(net, save_grads, task_id, num_mini_tasks):
    for n,p in enumerate(net.shared.named_parameters()):
        if n in save_grads:
            p.grad = save_grads[n]/num_mini_tasks
    try:
        for n,p in net.private.conv[task_id].named_parameters():
            if n in save_grads:
                p.grad = save_grads[n]/num_mini_tasks
    except:
        pass

    for n,p in net.private.linear[task_id].named_parameters():
        if n in save_grads:
            p.grad = save_grads[n]/num_mini_tasks

    try:
        for n,p in net.private.last_em[task_id].named_parameters():
            if n in save_grads:
                p.grad = save_grads[n]/num_mini_tasks
    except:
        pass

    for n,p in net.head[task_id].named_parameters():
        if n in save_grads:
            p.grad = save_grads[n]/num_mini_tasks

def train_dataset(net, opti, criterion, dataloader, epochs, task_id, device, use_only_share):
    net.train()
    for e in range(epochs):
        correct, loss = 0.0, 0.0
        total = 0
        for i, batch in enumerate(dataloader):
            opti.zero_grad()
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            inputs_feats = batch[2].to(device)

            outs, loss_reg = net(inputs, inputs_feats, task_id, use_only_share)
            _, preds = outs.max(1)

            l = criterion(outs, labels) + loss_reg
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
    inputs_feats = batch[2].to(device)
    scheduler = optim.lr_scheduler.StepLR(opti, step_size=15, gamma=0.5)
    for _ in range(inner_loop):
        if opti is not None:
            opti.zero_grad()

        outs, _ = net(inputs.clone(), inputs_feats.clone(), task_id, True)
        _, preds = torch.max(outs, 1)

        correct = preds.eq(labels.clone().view_as(preds)).sum().item()
        l = criterion(outs, labels.clone())
        l.backward()

        if type(save_grad) == dict:
            save_grad['acc'].append(correct/inputs.size(0))

            local_grad = init_grads_out(net)
            for name, param in net.named_parameters():
                if param.grad is not None:
                    local_grad[name] = param.grad.clone()

            save_grad['grads'].append(local_grad)

        else:
            opti.step()
            #scheduler.step()

        #print("Loss: {}\t Correct:{}\t Total:{}".format(l.item(),correct,inputs.size(0)))
        running_loss = l.item()

    return running_loss, correct/inputs.size(0)

def train_mini_task(args, net, dataloader, task_id, criterion, device):
    iter_data_train = iter(dataloader['train'])
    iter_data_val = iter(dataloader['train'])
    
    grads_acc = {'grads': [], 'acc': []}
    loss_mini_task = 0.0
    total_loss = 0.0
    for k in range(args.mini_tasks):
        t_net = copy.deepcopy(net)
        opti_priv = getOptimizer(args.shad_mini, args.priv_mini, args.head_mini, t_net, args.lr_mini, task_id)

        try:
            batch = next(iter_data_train)
        except:
            iter_data_train = iter(dataloader['train'])
            batch = next(iter_data_train)

        temp_loss, acc_mini = train_batch(t_net, opti_priv, criterion, batch, args.inner_loop, task_id, device) 
        loss_mini_task += temp_loss
        grads_acc['grads'].append(get_diff_weights(net, t_net))
        grads_acc['acc'].append(acc_mini)
        
        # for k in range(args.mini_tests):
        #     try:
        #         batch = next(iter_data_val)
        #     except:
        #         iter_data_val = iter(dataloader['train'])
        #         batch = next(iter_data_val)
       
        # temp_loss, _= train_batch(t_net, None, criterion, batch, 1, task_id, device, grads_acc)
        # total_loss += temp_loss

    save_grads = init_grads_out(net)
    weights = torch.tensor(grads_acc['acc'])/sum(grads_acc['acc'])
    for i, g in enumerate(grads_acc['grads']):
        for n in g:
            save_grads[n] += g[n]*weights[i]/args.inner_loop
    #Ponderacion
    return save_grads, total_loss, loss_mini_task, np.mean(grads_acc['acc'])

def train(args, net, task_id, dataloader, criterion, device):
    opti_shar = getOptimizer(args.shad_meta, args.priv_meta, args.head_meta, net, args.lr_meta, task_id)
    scheduler_shar = optim.lr_scheduler.ReduceLROnPlateau(opti_shar, mode='min', 
                    factor=0.5, patience=args.lr_patience, min_lr=1e-5, eps=1e-08)

    opti_priv = getOptimizer(args.shad_task, args.priv_task, args.head_task, net, args.lr_task, task_id)
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

    res_train = train_dataset(net, opti_priv, criterion, dataloader['train'], args.pri_epochs, task_id, device, False)
    results['train_loss'].append(res_train[1])
    results['train_acc'].append(res_train[0])

    res_test = test(net, task_id, dataloader['valid'], criterion, device)
    print("Tain loss: {}\t Acc: {}\t Valid loss: {}\t Acc: {}".format(res_train[1],res_train[0],res_test[1],res_test[0]))
    results['val_loss'].append(res_test[1])
    results['val_acc'].append(res_test[0])

    for i in range(args.out_epochs):
        if args.mini_tasks > 0:
            save_grads, total_loss, loss_mini_task, acc = train_mini_task(args, net, dataloader, task_id, criterion, device)
            print("Train: Total loss: {} \t Mini Task Loss: {} \t Acc: {}".format(total_loss/args.mini_tasks,loss_mini_task/args.mini_tasks, acc))
            results['meta_loss'].append(total_loss/args.mini_tasks)
            results['mini_loss'].append(loss_mini_task/args.mini_tasks)

            # print("Before opti step: ")
            # print_sum_params(net, task_id)
            set_grads(net, save_grads, task_id, args.mini_tasks)
            opti_shar.step()
            opti_shar.zero_grad()
            # print("After opti step: ")
            # print_sum_params(net, task_id)

            scheduler_shar.step(total_loss)

        if i % args.val_iter == 0:
            res_train = train_dataset(net, opti_priv, criterion, dataloader['train'], args.pri_epochs, task_id, device, False)
            results['train_loss'].append(res_train[1])
            results['train_acc'].append(res_train[0])

            res_test = test(net, task_id, dataloader['valid'], criterion, device)
            print("Train loss: {}\t Acc: {}\t Valid loss: {}\t Acc: {}".format(res_train[1],res_train[0],res_test[1],res_test[0]))
            results['val_loss'].append(res_test[1])
            results['val_acc'].append(res_test[0])

            if res_test[1] < best_loss:
                best_loss = res_test[1]
                best_model = copy.deepcopy(net)
            else:
                net.load_state_dict(copy.deepcopy(best_model).state_dict())
                opti_priv = getOptimizer(args.shad_task, args.priv_task, args.head_task, net, args.lr_task, task_id)

            scheduler_priv.step(res_test[1])

    res_train = train_dataset(net, opti_priv, criterion, dataloader['train'], args.pri_epochs, task_id, device, False)
    results['train_loss'].append(res_train[1])
    results['train_acc'].append(res_train[0])

    res_test = test(net, task_id, dataloader['valid'], criterion, device)
    print("Train loss: {}\t Acc: {}\t Valid loss: {}\t Acc: {}".format(res_train[1],res_train[0],res_test[1],res_test[0]))
    results['val_loss'].append(res_test[1])
    results['val_acc'].append(res_test[0])

    if res_test[1] < best_loss:
        best_model = copy.deepcopy(net)
    else:
        net.load_state_dict(copy.deepcopy(best_model).state_dict())

    return results

def trainAll(args, net, task_id, dataloader, criterion, device):
    net.train()
    opti_total = getOptimizer(args.use_share, args.use_private, True, net, args.lr_task, task_id)

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
    use_only_share = False
    for i in range(args.out_epochs):
        res_train = train_dataset(net, opti_total, criterion, dataloader['train'], 1, task_id, device, use_only_share)
        results['train_loss'].append(res_train[1])
        results['train_acc'].append(res_train[0])

        res_test = test(net, task_id, dataloader['valid'], criterion, device)
        results['val_loss'].append(res_test[1])
        results['val_acc'].append(res_test[0])

        print("Train loss: {}\t Acc: {}\t Valid loss: {}\t Acc: {}".format(res_train[1],res_train[0],res_test[1],res_test[0]))

        if res_test[1] < best_loss:
            best_loss = res_test[1]
            best_model = copy.deepcopy(net)
        else:
            net.load_state_dict(copy.deepcopy(best_model).state_dict())
            opti_total = getOptimizer(args.use_share, args.use_private, True, net, args.lr_task, task_id)

        scheduler.step(res_test[1])

        if i > args.out_epochs/2:
            use_only_share = False

    return results

def test(net, task_id, dataloader, criterion, device):
    net.eval()
    correct, loss = 0.0, 0.0
    total = 0
    for i, batch in enumerate(dataloader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        inputs_feats = batch[2].to(device)

        outs, _ = net(inputs, inputs_feats, task_id)
        _, preds = outs.max(1)

        correct += preds.eq(labels.view_as(preds)).sum().item()

        l = criterion(outs, labels) 
        loss += l.item()

        total += inputs.size(0)

    return correct/total, loss/len(dataloader)