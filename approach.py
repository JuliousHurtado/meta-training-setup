import copy
import numpy as np
import random

import torch
from torch import optim



def getOptimizer(shared, private, private_lin, head, net, lr, task_id):
    params = []

    if private:
        # try:
        #     for p in net.private.conv[task_id].parameters():
        #         params.append(p)
        # except:
        #     pass

        for p in net.private.linear[task_id].parameters():
            params.append(p)

    if private_lin:
        for p in net.private.last_em[task_id].parameters():
            params.append(p)

    if head:
        for p in net.head[task_id].parameters():
            params.append(p)

    if shared:
        for p in net.shared.parameters():
            params.append(p)

    return optim.SGD(params, lr, weight_decay=0.01, momentum=0.9)

def init_grads_out(net):
    grads = {}

    for n,p in net.shared.named_parameters():
        grads[n] = torch.zeros_like(p)

    return grads

def get_diff_weights(old_net, new_net, device='cuda'):
    grads = {}

    for (_,p_new),(n,p_old) in zip(new_net.shared.named_parameters(), old_net.shared.named_parameters()):
        grads[n] = (p_new - p_old)#.to(device)

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

def set_grads(net, save_grads, task_id):
    for n,p in enumerate(net.shared.named_parameters()):
        if n in save_grads:
            p.grad = save_grads[n]
    try:
        for n,p in net.private.conv[task_id].named_parameters():
            if n in save_grads:
                p.grad = save_grads[n]
    except:
        pass

    for n,p in net.private.linear[task_id].named_parameters():
        if n in save_grads:
            p.grad = save_grads[n]

    try:
        for n,p in net.private.last_em[task_id].named_parameters():
            if n in save_grads:
                p.grad = save_grads[n]
    except:
        pass

    for n,p in net.head[task_id].named_parameters():
        if n in save_grads:
            p.grad = save_grads[n]

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

def train_batch(net, opti, criterion, batch, inner_loop, task_id, device):
    running_loss = 0.0
    net.train()
    inputs = batch[0].to(device)
    labels = batch[1].to(device)
    inputs_feats = batch[2].to(device)
    if opti is not None:
        scheduler = optim.lr_scheduler.StepLR(opti, step_size=15, gamma=0.5)
    for _ in range(inner_loop):
        if opti is not None:
            opti.zero_grad()

        outs, _ = net(inputs.clone(), inputs_feats.clone(), task_id, True)
        _, preds = torch.max(outs, 1)

        correct = preds.eq(labels.clone().view_as(preds)).sum().item()

        if criterion is not None:
            l = criterion(outs, labels.clone())
            l.backward()
            
            #print("Loss: {}\t Correct:{}\t Total:{}".format(l.item(),correct,inputs.size(0)))
            running_loss = l.item()

        if opti is not None:
            opti.step()
            scheduler.step()

    return running_loss, correct/inputs.size(0)

def train_mini_task(args, net, dataloader, task_id, criterion, device):
    iter_data_train = iter(dataloader['train'])
    iter_data_val = iter(dataloader['train'])
    
    grads_acc = {'grads': [], 'acc': []}
    loss_mini_task = 0.0
    total_loss = 0.0
    for k in range(args.mini_tasks):
        t_net = copy.deepcopy(net)
        opti_priv = getOptimizer(args.shad_mini, args.priv_mini, False, args.head_mini, t_net, args.lr_mini, task_id)

        try:
            batch = next(iter_data_train)
        except:
            iter_data_train = iter(dataloader['train'])
            batch = next(iter_data_train)

        temp_loss, acc_mini = train_batch(t_net, opti_priv, criterion, batch, args.inner_loop, task_id, device) 
        loss_mini_task += temp_loss
        grads_acc['grads'].append(get_diff_weights(net, t_net))
        
        try:
            batch = next(iter_data_val)
        except:
            iter_data_val = iter(dataloader['train'])
            batch = next(iter_data_val)
       
        _, acc_mini= train_batch(t_net, None, None, batch, 1, task_id, device)
        grads_acc['acc'].append(acc_mini)

    save_grads = init_grads_out(net)
    weights = torch.tensor(grads_acc['acc'])/sum(grads_acc['acc'])
    for i, g in enumerate(grads_acc['grads']):
        for n in g:
            save_grads[n] += g[n]*weights[i]/args.inner_loop
    #Ponderacion
    return save_grads, total_loss, loss_mini_task, np.mean(grads_acc['acc'])

def train(args, net, task_id, dataloader, criterion, device):
    if args.use_share:
        opti_shar = getOptimizer(args.shad_meta, args.priv_meta, False, args.head_meta, net, args.lr_meta, task_id)
        scheduler_shar = optim.lr_scheduler.ReduceLROnPlateau(opti_shar, mode='min', 
                    factor=0.5, patience=args.lr_patience, min_lr=1e-5, eps=1e-08)

    opti_priv = getOptimizer(args.shad_task, args.priv_task, True, args.head_task, net, args.lr_task, task_id)
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

    train_features(args, net, dataloader, task_id, criterion, device)

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
                opti_priv = getOptimizer(args.shad_task, args.priv_task, True, args.head_task, net, args.lr_task, task_id)

            scheduler_priv.step(res_test[1])

    opti_priv = getOptimizer(False, False, True, True, net, args.lr_task, task_id)
    res_train = train_dataset(net, opti_priv, criterion, dataloader['train'], args.pri_epochs*3, task_id, device, False)
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
    train_features(args, net, dataloader['train'], task_id, criterion, device)

    net.train()
    opti_total = getOptimizer(args.use_share, args.use_private, True, True, net, args.lr_task, task_id)

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
            opti_total = getOptimizer(args.use_share, args.use_private, True, True, net, args.lr_task, task_id)

        scheduler.step(res_test[1])

        if i > args.out_epochs/2:
            use_only_share = False

    return results





def train_features(args, net, dataloader, task_id, criterion, device):
    params = []

    if not args.resnet18:
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
            inputs_feats = batch[2].to(device)

            if args.resnet18:
                outs = net.private.feat_extraction(inputs_feats).squeeze()
            else:
                outs = net.private.conv[task_id](inputs)
                # outs = net.private.avgpool(outs).squeeze()
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
        inputs_feats = batch[2].to(device)
        if args.resnet18:
            outs = net.private.feat_extraction(inputs_feats).squeeze()
        else:
            outs = net.private.conv[task_id](inputs)#.view(inputs.size(0),-1)
            # outs = net.private.avgpool(outs).squeeze()
        outs = clfs(outs)
        _, preds = outs.max(1)
        correct_test += preds.eq(labels.view_as(preds)).sum().item()
        total_test += inputs.size(0)

    print("Feature Training: Train loss: {:.4f} \t Acc Train: {:.4f} \t Acc Val: {:.4f}".format(loss/len(dataloader), correct/total, correct_test/total_test))

    if not args.resnet18:
        for p in net.private.conv[task_id].parameters():
            p.requires_grad = False

def test(net, task_id, dataloader, criterion, device):
    net.eval()
    correct, loss = 0.0, 0.0
    total = 0
    for i, batch in enumerate(dataloader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        inputs_feats = batch[2].to(device)

        if net.con_pri_shd:
            #outs, _ = net(inputs, inputs_feats, task_id)
            outs, _ = net.forward2(inputs, task_id, inputs_feats)
        else:
            outs, _ = net.forward5(inputs, task_id, inputs_feats)
        _, preds = outs.max(1)

        correct += preds.eq(labels.view_as(preds)).sum().item()

        l = criterion(outs, labels) 
        loss += l.item()

        total += inputs.size(0)
    net.train()
    return correct/total, loss/len(dataloader)

def trainPrueba(net, task_id, loader, opti, criterion, device):
    correct, loss = 0.0, 0.0
    total = 0

    for i, batch in enumerate(loader):
        opti.zero_grad()
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        inputs_feats = batch[2].to(device)

        outs, reg_loss = net.forward2(inputs, task_id, inputs_feats)
        _, preds = outs.max(1)
        l = criterion(outs, labels) + reg_loss*0.01
        l.backward()

        opti.step()

        correct += preds.eq(labels.view_as(preds)).sum().item()
        loss += l.item()

        total += inputs.size(0)

    return correct/total, loss/len(loader)

def trainBatchPrueba(net, opti, criterion, batch, inner_loop, task_id, device):
    running_loss = 0.0
    net.train()
    inputs = batch[0].to(device)
    labels = batch[1].to(device)
    inputs_feats = batch[2].to(device)

    if opti is not None:
        # scheduler = optim.lr_scheduler.StepLR(opti, step_size=30, gamma=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opti,factor=0.5, patience=5, threshold=0.001)

    for _ in range(inner_loop):
        if opti is not None:
            opti.zero_grad()

        outs, _ = net.forward3(inputs, task_id, inputs_feats)
        _, preds = torch.max(outs, 1)

        correct = preds.eq(labels.clone().view_as(preds)).sum().item()

        if criterion is not None:
            l = criterion(outs, labels.clone())
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.shared.parameters(),0.5)
            torch.nn.utils.clip_grad_norm_(net.shared_clf.parameters(),0.5)
            running_loss = l.item()

        if opti is not None:
            opti.step()
            scheduler.step(running_loss)

    return running_loss, correct/inputs.size(0)

def trainTaskPrueba(args, net, loader, task_id, opti_shared, criterion, device, memory):
    grads_acc = {'grads': [], 'acc': []}
    loss_mini_task = 0.0
    total_loss = 0.0
    for k in range(args.mini_tasks):
        t_net = copy.deepcopy(net)
        t_net.shared_clf = torch.nn.Linear(net.private.dim_embedding, net.taskcla[task_id][1]).to(device)

        params = []
        # for p in t_net.private.linear.parameters():
        #     params.append(p)
        for p in t_net.shared.parameters():
            params.append(p)
        for p in t_net.shared_clf.parameters():
            params.append(p)

        opti_shared_task = optim.SGD(params, args.lr_meta, weight_decay=0.1, momentum=0.9)
        # opti_shared_task = getOptimizer(args.shad_meta, args.priv_meta, args.priv_l_meta, args.head_meta, t_net, args.lr_meta, task_id)
        
        prob = random.uniform(0, 1)
        if args.use_memory and prob < args.prob_use_mem and task_id > 0:
            task_key = random.sample(memory.keys(), 1)[0]
            batch = random.sample(memory[task_key], 1)[0]
            batch_task_id = task_key
        else:
            try:
                batch = next(iter_data_train)
            except:
                iter_data_train = iter(loader)
                batch = next(iter_data_train)

            if task_id not in memory:
                memory[task_id] = []

            prob = random.uniform(0, 1)
            if len(memory[task_id]) < args.mem_size:
                memory[task_id].append(batch)
            elif prob < 0.7:
                idex = random.sample(list(range(len(memory[task_id]))), 1)[0]
                memory[task_id][idex] = batch

            batch_task_id = task_id

        temp_loss, acc_mini = trainBatchPrueba(t_net, opti_shared_task, criterion, batch, args.inner_loop, batch_task_id, device) 
        loss_mini_task += temp_loss

        grads_acc['grads'].append(get_diff_weights(net, t_net))
        
        # try:
        #     batch = next(iter_data_val)
        # except:
        #     iter_data_val = iter(loader)
        #     batch = next(iter_data_val)
       
        _, acc_mini= trainBatchPrueba(t_net, None, None, batch, 1, task_id, device)
        grads_acc['acc'].append(acc_mini)

    save_grads = init_grads_out(net)
    weights = torch.tensor(grads_acc['acc'])/sum(grads_acc['acc'])
    for i, g in enumerate(grads_acc['grads']):
        for n in g:
            save_grads[n] += g[n].to(device)*weights[i]/(args.inner_loop+args.mini_tasks)

    set_grads(net, save_grads, task_id)
    opti_shared.step()
    opti_shared.zero_grad()
    # print(grads_acc['acc'])
    # print(loss_mini_task)
    return np.mean(grads_acc['acc']), loss_mini_task/args.mini_tasks

def trainShared(args, net, loader, task_id, opti_shared, criterion, fun_forward, device):
    for e in range(args.feats_epochs):
        correct = 0.0
        total = 0.0
        loss = 0.0
        for i, batch in enumerate(loader):
            opti_shared.zero_grad()
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            inputs_feats = batch[2].to(device)

            outs, reg_loss  = fun_forward(inputs, task_id, inputs_feats)
            _, preds = outs.max(1)
            l = criterion(outs, labels) + reg_loss
            l.backward()

            opti_shared.step()

            correct += preds.eq(labels.clone().view_as(preds)).sum().item()
            total += inputs.size(0)
            loss += l.item()

    #print("[{}|{}]Pre Acc: {:.4f}".format(e+1,args.feats_epochs,correct/total))
    return correct/total, loss/len(loader)

def printSum(net, task_id):
    p_conv, p_lin, p_emb = 0, 0, 0
    for p in net.private.conv[task_id].parameters():
        p_conv += p.sum()

    for p in net.private.linear[task_id].parameters():
        p_lin += p.sum()

    for p in net.private.last_em[task_id].parameters():
        p_emb += p.sum()

    head, shared = 0, 0
    for p in net.head[task_id].parameters():
        head += p.sum()

    for p in net.shared.parameters():
        shared += p.sum()

    print("Private -> Conv: {} , Linear: {} , Embedding: {}\nHead -> {}\nShared -> {}".format(p_conv, 
                        p_lin, p_emb, head, shared))

def getMasks(net, task_id, dataloader, device):
    m_all = {'masks': {}, 'labels': []}
    for i, batch in enumerate(dataloader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        inputs_feats = batch[2].to(device)

        m_all['labels'].extend(labels)

        masks = net.get_masks(inputs, task_id, inputs_feats)

        for i, m in enumerate(masks):
            if i not in m_all['masks']:
                m_all['masks'][i] = []
            #m_all[i].append(m[0].squeeze().mean(dim=0).tolist())
            m_all['masks'][i].extend(m[0].squeeze().tolist())

    # for k,v in m_all.items():
    #     m_all[k] = np.mean(v, axis = 0)
    return m_all

def prueba(args, net, task_id, dataloader, criterion, device):
    results = {
        'meta_loss': [],
        'meta_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_loss': [],
        'train_acc': []
    }

    if not args.resnet18:
        train_features(args, net, dataloader, task_id, criterion, device)


    if task_id == 0:
        net.shared_clf = torch.nn.Linear(net.private.dim_embedding, net.taskcla[task_id][1]).to(device)
        params = []
        for p in net.shared.parameters():
            params.append(p)
        for p in net.shared_clf.parameters():
            params.append(p)
        opti_shared_task = optim.SGD(params, args.lr_task, weight_decay=0.01, momentum=0.9)
        trainShared(args, net, dataloader['train'], task_id, opti_shared_task, criterion, net.forward4, device)


    net.shared_clf = torch.nn.Linear(net.private.dim_embedding, net.taskcla[task_id][1]).to(device)
    params = []
    for p in net.private.linear[task_id].parameters():
        params.append(p)
    for p in net.shared_clf.parameters():
        params.append(p)
    opti_shared_mask = optim.SGD(params, args.lr_task, weight_decay=0.01, momentum=0.9)
    trainShared(args, net, dataloader['train'], task_id, opti_shared_mask, criterion, net.forward3, device)



    net.shared_clf = torch.nn.Linear(net.private.dim_embedding, net.taskcla[task_id][1]).to(device)
    params = []
    # for p in net.private.linear[task_id].parameters():
    #     params.append(p)
    for p in net.shared.parameters():
        params.append(p)
    # for p in net.shared_clf.parameters():
    #     params.append(p)
    opti_shared = optim.SGD(params, args.lr_meta*0.1,  weight_decay=0.9) # 
    #opti_shared = getOptimizer(args.shad_meta, args.priv_meta, args.priv_l_meta, args.head_meta, net, args.lr_meta, task_id)
    for e in range(args.meta_epochs):
        # trainShared(args, net, dataloader['train'], task_id, opti_shared, criterion, net.forward3, device)
        #res_train = trainPrueba(net, task_id, dataloader['train'], opti_shared, criterion, device)
        #meta_acc = res_train[0]
        meta_acc, meta_loss = trainTaskPrueba(args, net, dataloader['train'], task_id, opti_shared, criterion, device)
        print("[{}|{}]Meta Acc: {:.4f}\t Loss: {:.4f}".format(e+1,args.meta_epochs,meta_acc, meta_loss))



    params = []
    for p in net.private.last_em[task_id].parameters():
        params.append(p)
    for p in net.head[task_id].parameters():
        params.append(p)
    # for p in net.private.linear[task_id].parameters():
    #     params.append(p)
    opti = optim.SGD(params, args.lr_task, weight_decay=0.01, momentum=0.9)
    #opti = getOptimizer(args.shad_task, args.priv_task, args.priv_l_task, args.head_task, net, args.lr_task, task_id)
    best_loss = np.inf
    best_model = copy.deepcopy(net)

    for e in range(args.epochs):
        res_train = trainPrueba(net, task_id, dataloader['train'], opti, criterion, device)
        res_test = test(net, task_id, dataloader['valid'], criterion, device)
        print("[{}|{}]Train loss: {:.4f} \t Acc Train: {:.4f} \t Acc Val: {:.4f}".format(e+1,args.epochs,res_train[1], res_train[0], res_test[0]))

        if res_test[1] < best_loss:
            best_loss = res_test[1]
            best_model = copy.deepcopy(net)
        else:
            net.load_state_dict(copy.deepcopy(best_model).state_dict())
            opti_priv = getOptimizer(args.shad_task, args.priv_task, args.priv_l_task, args.head_task, net, args.lr_task, task_id)



def prueba2(args, net, task_id, dataloader, criterion, device, memory):
    results = {
        'meta_loss': [],
        'meta_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_loss': [],
        'train_acc': []
    }

    if not args.only_shared:
        if not args.resnet18:
            train_features(args, net, dataloader, task_id, criterion, device)
            mask_lr = args.lr_task
        else:
            mask_lr = args.lr_task*0.1
    else:
        mask_lr = args.lr_task



    if task_id == 0:
        params = []
        for p in net.shared.parameters():
            params.append(p)
        for p in net.head[task_id].parameters():
            params.append(p)
        opti_shared_task = optim.SGD(params, args.lr_task, weight_decay=0.01, momentum=0.9)
        acc_train, loss_train = trainShared(args, net, dataloader['train'], task_id, opti_shared_task, criterion, net.forward6, device)
        acc_valid, _ = test(net, task_id, dataloader['valid'], criterion, device)
        print("Train Shared: Train loss: {:.4f} \t Acc Train: {:.4f} \t Acc Val: {:.4f}".format(loss_train, acc_train, acc_valid))


    if not args.only_shared:
        params = []
        for p in net.private.linear[task_id].parameters():
            params.append(p)
        for p in net.head[task_id].parameters():
            params.append(p)
        opti_shared_mask = optim.SGD(params, mask_lr, weight_decay=0.01, momentum=0.9)
        acc_train, loss_train = trainShared(args, net, dataloader['train'], task_id, opti_shared_mask, criterion, net.forward5, device)
        acc_valid, _ = test(net, task_id, dataloader['valid'], criterion, device)
        print("Mask Training: Train loss: {:.4f} \t Acc Train: {:.4f} \t Acc Val: {:.4f}".format(loss_train, acc_train, acc_valid))




    net.shared_clf = torch.nn.Linear(net.private.dim_embedding, net.taskcla[task_id][1]).to(device)
    params = []
    for p in net.shared.parameters():
        params.append(p)
    if not args.use_meta:
        args.meta_epochs = 1
        for p in net.head[task_id].parameters():
            params.append(p)
    opti_shared = optim.SGD(params, args.lr_meta*0.1,  weight_decay=0.9) # 
    for e in range(args.meta_epochs):
        if args.use_meta:
            meta_acc, meta_loss = trainTaskPrueba(args, net, dataloader['train'], task_id, opti_shared, criterion, device, memory)
        else:
            meta_acc, meta_loss = trainShared(args, net, dataloader['train'], task_id, opti_shared, criterion, net.forward5, device)
        print("[{}|{}]Meta Acc: {:.4f}\t Loss: {:.4f}".format(e+1,args.meta_epochs,meta_acc, meta_loss))


    if args.only_shared:
        params = []
        for p in net.head[task_id].parameters():
            params.append(p)
        opti_shared_mask = optim.SGD(params, mask_lr, weight_decay=0.01, momentum=0.9)
        acc_train, loss_train = trainShared(args, net, dataloader['train'], task_id, opti_shared_mask, criterion, net.forward6, device)
        acc_valid, _ = test(net, task_id, dataloader['valid'], criterion, device)
        print("Final Training: Train loss: {:.4f} \t Acc Train: {:.4f} \t Acc Val: {:.4f}".format(loss_train, acc_train, acc_valid))
    else:
        params = []
        for p in net.private.linear[task_id].parameters():
            params.append(p)
        for p in net.head[task_id].parameters():
            params.append(p)
        opti_shared_mask = optim.SGD(params, mask_lr, weight_decay=0.01, momentum=0.9)
        acc_train, loss_train = trainShared(args, net, dataloader['train'], task_id, opti_shared_mask, criterion, net.forward5, device)
        acc_valid, _ = test(net, task_id, dataloader['valid'], criterion, device)
        print("Final Training: Train loss: {:.4f} \t Acc Train: {:.4f} \t Acc Val: {:.4f}".format(loss_train, acc_train, acc_valid))