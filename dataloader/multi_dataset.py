# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import sys
import copy

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import torch.utils.data
from torchvision import transforms

import learn2learn as l2l

from dataloader.datasets_utils import *

mean_datasets = {
    'CIFAR10': [x/255 for x in [125.3,123.0,113.9]],
    'notMNIST': (0.4254,),
    'MNIST': (0.1,) ,
    'SVHN':[0.4377,0.4438,0.4728] ,
    'FashionMNIST': (0.2190,),

}
std_datasets = {
    'CIFAR10': [x/255 for x in [63.0,62.1,66.7]],
    'notMNIST': (0.4501,),
    'MNIST': (0.2752,),
    'SVHN': [0.198,0.201,0.197],
    'FashionMNIST': (0.3318,)
}

classes_datasets = {
    'CIFAR10': 10,
    'notMNIST': 10,
    'MNIST': 10,
    'SVHN': 10,
    'FashionMNIST': 10,
}


gray_datasets = {
    'CIFAR10': False,
    'notMNIST': True,
    'MNIST': True,
    'SVHN': False,
    'FashionMNIST': True,
}


class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args):
        super(DatasetGen, self).__init__()

        self.batch_size = args.batch_size
        self.pc_valid = 0.15
        self.root = './data'

        self.num_task = args.num_tasks
        self.num_samples = 0

        self.inputsize = [3,32,32]

        self.num_workers = 4
        self.pin_memory = True

        #self.datasets_idx = list(np.random.permutation(self.num_task))
        self.datasets_idx = [1, 4, 2, 0, 3]
        self.datasets_idx = self.datasets_idx[:self.num_task]
        
        print('Task order =', [list(classes_datasets.keys())[item] for item in self.datasets_idx])
        self.datasets_names = [list(classes_datasets.keys())[item] for item in self.datasets_idx]

        self.taskcla = []

        for i in range(self.num_task):
            t = self.datasets_idx[i]
            self.taskcla.append(list(classes_datasets.values())[t])
        print('taskcla =', self.taskcla)

        self.train_set = {}
        self.train_split = {}
        self.test_set = {}

        self.args = args

        self.dataloaders = {}
        for i in range(self.num_task):
            self.dataloaders[i] = {}

        self.download = True

    def get_dataset(self, dataset_idx, task_num, num_samples_per_class=False, normalize=True):
        dataset_name = list(mean_datasets.keys())[dataset_idx]
        nspc = num_samples_per_class
        if normalize:
            transformation = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean_datasets[dataset_name],std_datasets[dataset_name])])
            mnist_transformation = transforms.Compose([
                transforms.Pad(padding=2, fill=0),
                transforms.ToTensor(),
                transforms.Normalize(mean_datasets[dataset_name], std_datasets[dataset_name])])
        else:
            transformation = transforms.Compose([transforms.ToTensor()])
            mnist_transformation = transforms.Compose([
                transforms.Pad(padding=2, fill=0),
                transforms.ToTensor(),
                ])

        # target_transormation = transforms.Compose([transforms.ToTensor()])
        target_transormation = None

        if dataset_idx == 0:
            trainset = CIFAR10_(root=self.root, task_num=task_num, num_samples_per_class=nspc, train=True, download=self.download, target_transform = target_transormation, transform=transformation)
            testset = CIFAR10_(root=self.root,  task_num=task_num, num_samples_per_class=nspc, train=False, download=self.download, target_transform = target_transormation, transform=transformation)

        if dataset_idx == 1:
            trainset = notMNIST_(root=self.root, task_num=task_num, num_samples_per_class=nspc, train=True, download=self.download, target_transform = target_transormation, transform=mnist_transformation)
            testset = notMNIST_(root=self.root,  task_num=task_num, num_samples_per_class=nspc, train=False, download=self.download, target_transform = target_transormation, transform=mnist_transformation)

        if dataset_idx == 2:
            trainset = MNIST_RGB(root=self.root, train=True, num_samples_per_class=nspc, task_num=task_num, download=self.download, target_transform = target_transormation, transform=mnist_transformation)
            testset = MNIST_RGB(root=self.root,  train=False, num_samples_per_class=nspc, task_num=task_num, download=self.download, target_transform = target_transormation, transform=mnist_transformation)

        if dataset_idx == 3:
            trainset = SVHN_(root=self.root,  train=True, num_samples_per_class=nspc, task_num=task_num, download=self.download, target_transform = target_transormation, transform=transformation)
            testset = SVHN_(root=self.root,  train=False, num_samples_per_class=nspc, task_num=task_num, download=self.download, target_transform = target_transormation, transform=transformation)

        if dataset_idx == 4:
            trainset = FashionMNIST_(root=self.root, num_samples_per_class=nspc, task_num=task_num, train=True, download=self.download, target_transform = target_transormation, transform=mnist_transformation)
            testset = FashionMNIST_(root=self.root,  num_samples_per_class=nspc, task_num=task_num, train=False, download=self.download, target_transform = target_transormation, transform=mnist_transformation)

        return trainset, testset


    def get(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        current_dataset_idx = self.datasets_idx[task_id]
        dataset_name = list(mean_datasets.keys())[current_dataset_idx]
        self.train_set[task_id], self.test_set[task_id] = self.get_dataset(current_dataset_idx,task_id)

        self.num_classes = classes_datasets[dataset_name]

        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id], [len(self.train_set[task_id]) - split, split])

        if self.args.meta_train:
            meta_dataset = copy.deepcopy(self.train_set[task_id])
            meta_loader = self.get_meta_loader(meta_dataset, task_id)

            self.dataloaders[task_id]['meta'] = meta_loader
        else:
            self.dataloaders[task_id]['meta'] = None

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory,shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=self.batch_size,
                                                   num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory,shuffle=True)

        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = '{} - {} classes - {} images'.format(dataset_name,
                                                                              classes_datasets[dataset_name],
                                                                              len(self.train_set[task_id]))
        self.dataloaders[task_id]['classes'] = self.num_classes


        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Validation set size: {} images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Train+Val  set size: {} images of {}x{}".format(len(valid_loader.dataset)+len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        return self.dataloaders

    def get_meta_loader(self, meta_dataset, task_id):
        create_bookkeeping(meta_dataset, self.taskcla[task_id])

        meta_transforms = [
                    l2l.data.transforms.NWays(meta_dataset, self.taskcla[task_id]),
                    l2l.data.transforms.KShots(meta_dataset, 2*self.args.shots),
                    l2l.data.transforms.LoadData(meta_dataset),
                    l2l.data.transforms.RemapLabels(meta_dataset),
                    l2l.data.transforms.ConsecutiveLabels(meta_dataset),
                ]

        meta_loader = l2l.data.TaskDataset(l2l.data.MetaDataset(meta_dataset),
                                           task_transforms=meta_transforms)

        return meta_loader