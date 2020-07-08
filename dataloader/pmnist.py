# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys, os
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from utils import *
import copy


class PermutedMNIST(datasets.MNIST):

    def __init__(self, root, task_num, train=True, permute_idx=None, transform=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        self.data = torch.stack([img.float().view(-1)[permute_idx].view(1,28,28) for img in self.data])
        self.tl = (task_num) * torch.ones(len(self.data),dtype=torch.long)
        self.td = (task_num+1) * torch.ones(len(self.data),dtype=torch.long)


    def __getitem__(self, index):

        img, target, tl, td = self.data[index], self.targets[index], self.tl[index], self.td[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            print ("We are transforming")
            target = self.target_transform(target)

        return img, target#, tl, td

    def __len__(self):
        return self.data.size(0)

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        temp = []
        for img in self.data[sample_idx]:
            if self.transform is not None:
                img = self.transform(img)

            temp.append(img)
        return temp


class DatasetGen(object):

    def __init__(self, args):
        super(DatasetGen, self).__init__()

        self.seed = args.seed
        self.batch_size=args.batch_size
        self.pc_valid=0.15
        # self.num_samples = args.samples
        self.num_task = 10
        self.root = './data'
        # self.use_memory = args.use_memory

        self.meta_learn = args.meta_learn
        self.ways=args.ways
        self.shots=args.shots
        self.meta_label=args.meta_label

        self.inputsize = [1, 28, 28]
        mean = (0.1307,)
        std = (0.3081,)
        self.transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])

        self.taskcla = [ 10 for t in range(self.num_task)]

        self.train_set, self.test_set = {}, {}
        self.indices = {}
        self.dataloaders = {}
        self.idx={}
        self.get_idx()

        self.pin_memory = True
        self.num_workers = 4

        self.task_memory = []



    def get(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        self.train_set[task_id] = PermutedMNIST(root=self.root, task_num=task_id, train=True,
                                                permute_idx=self.idx[task_id], transform=self.transformation)

        self.test_set[task_id] = PermutedMNIST(root=self.root, task_num=task_id, train=False,
                                               permute_idx=self.idx[task_id], transform=self.transformation)

        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id],
                                                                 [len(self.train_set[task_id]) - split, split])

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size,
                                                num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=self.batch_size,
                                                num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size,
                                                num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)

        if self.meta_learn:
            meta_dataset = copy.deepcopy(self.train_set[task_id])
            meta_loader = self.get_meta_loader(meta_dataset)
            self.dataloaders[task_id]['meta'] = meta_loader
        else:
            self.dataloaders[task_id]['meta'] = None

        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = 'pmnist-{}'.format(task_id+1)

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

    def get_idx(self):
        for i in range(len(self.taskcla)):
            idx = list(range(self.inputsize[1] * self.inputsize[2]))
            self.idx[i] = shuffle(idx, random_state=self.seed * 100 + i)


def create_bookkeeping(dataset, ways, meta_label='supervised'):
    """
    Iterates over the entire dataset and creates a map of target to indices.
    Returns: A dict with key as the label and value as list of indices.
    """
    assert hasattr(dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

    labels_to_indices = defaultdict(list)
    indices_to_labels = defaultdict(int)

    if meta_label == 'supervised':
        #------- Original (Supervised) ---#
        for i in range(len(dataset)):
            try:
                label = dataset[i][1]
                # if label is a Tensor, then take get the scalar value
                if hasattr(label, 'item'):
                    label = dataset[i][1].item()
            except ValueError as e:
                raise ValueError(
                    'Requires scalar labels. \n' + str(e))

            labels_to_indices[label].append(i)
            indices_to_labels[i] = label
    else:
        #-------Random Unsupervised-------#
        labels = list(range(ways))
        targets = []
        for i in range(len(dataset)):
            l = random.choice(labels)

            labels_to_indices[l].append(i)
            indices_to_labels[i] = l

            targets.append(l)

        dataset.targets = targets

    dataset.labels_to_indices = labels_to_indices
    dataset.indices_to_labels = indices_to_labels
    dataset.labels = list(dataset.labels_to_indices.keys())

    # print(dataset.labels_to_indices)