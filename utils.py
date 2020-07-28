import os
import numpy as np
import pickle

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
