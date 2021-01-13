import os
import numpy as np
import pickle

import torch

def print_log_acc_bwt(taskcla, acc, lss, output_path, run_id):

    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        print()

    avg_acc = np.mean(acc[acc.shape[0]-1,:])
    print ('ACC: {:5.4f}%'.format(avg_acc))
    print()
    print()
    # BWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
    gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)
    # BWT calculated based on UCB paper (https://arxiv.org/abs/1906.02425)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT: {:5.2f}%'.format(gem_bwt))
    # print ('BWT (UCB paper): {:5.2f}%'.format(ucb_bwt))

    print('*'*100)
    print('Done!')


    logs = {}
    # save results
    logs['name'] = output_path
    logs['taskcla'] = taskcla
    logs['acc'] = acc
    logs['loss'] = lss
    logs['gem_bwt'] = gem_bwt
    logs['ucb_bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]

    # pickle
    path = os.path.join(output_path, 'logs_run_id_{}.p'.format(run_id))
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    print ("Log file saved in ", path)
    return avg_acc, gem_bwt

def print_time():
    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Job finished at =", dt_string)

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

            m_all['masks'][i].extend(m[0].squeeze().tolist())

    return m_all

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