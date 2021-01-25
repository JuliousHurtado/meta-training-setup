# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from PIL import Image
import os
import os.path
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets import ImageFolder
import numpy as np

import torch
from torchvision import transforms

from utils import *



class ImageNet(torch.utils.data.Dataset):

    def __init__(self, root, classes, task_num, train, transform=None, 
                                        transform_feats=None):
        super(ImageNet, self).__init__()
        if train:
            self.name='train'
        else:
            self.name='val'
        self.transform = transform
        root = os.path.join(root, 'ILSVRC2012') # TODO: alterar con el nombre de la carpeta de Imagenet
        
        root_split = os.path.join(root, self.name)
        self.classes = []
        for folder in os.listdir(root_split):
            self.classes.append(folder)
        
        self.classes.sort()
        self.classes = np.array(self.classes) # For easier indexing
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.imgs = []
        self.labels = []
        
        classes_to_include = self.classes[classes]
        
        for c in classes_to_include:
            path = os.path.join(root_split, c)
            label = self.class_to_idx[c]
            for img in os.listdir(path):
                img = os.path.join(path,img)
                self.imgs.append(img)
                self.labels.append(label)
        
        # Create dictionary for labels
        self.labels = np.array(self.labels)
        self.class_labels = [self.class_to_idx[c] for c in classes_to_include]
        self.old_label_to_new_label = {label: i for i, label in enumerate(self.class_labels)}
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(self.imgs[index])

        imgnpy= np.array(img)
        if self.transform:

            if len(imgnpy.shape) == 2:    
                imgnpy = np.expand_dims(imgnpy, axis=-1)
                imgnpy = np.repeat(imgnpy, 3, axis=2)
                img = Image.fromarray(imgnpy)
            img_org = self.transform(img)
        else:
            img_org = img
        label = self.labels[index]
        label = self.old_label_to_new_label[label]
        
        return img_org, label, img_org

class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args):
        super(DatasetGen, self).__init__()

        self.seed = args.seed
        self.batch_size=args.batch_size
        self.pc_valid=args.pc_valid
        self.root = args.data_dir
        self.latent_dim = args.latent_dim
        self.use_memory = args.use_memory

        self.num_tasks = args.ntasks

        self.num_classes = 1000

        self.num_samples = args.samples

        self.inputsize = [3,224,224]           # Changed to suit ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transformation = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.transformation_feats = None


        # Obtener self.num_classes y self.num_tasks
        self.taskcla = [[t, int(self.num_classes/self.num_tasks)] for t in range(self.num_tasks)]

        self.indices = {}
        self.dataloaders = {}
        self.idx={}

        self.num_workers = args.workers
        self.pin_memory = True

        np.random.seed(self.seed)
        task_ids = np.split(np.random.permutation(self.num_classes),self.num_tasks)
        self.task_ids = [list(arr) for arr in task_ids]

        self.train_set = {}
        self.train_split = {}
        self.test_set = {}


    def get(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        self.train_set[task_id] = ImageNet(root=self.root, classes=self.task_ids[task_id],
                                                task_num=task_id, train=True, transform=self.transformation,
                                                transform_feats=self.transformation_feats)

        self.test_set[task_id] = ImageNet(root=self.root, classes=self.task_ids[task_id], task_num=task_id, 
                                        train=False, transform=self.transformation,
                                        transform_feats=self.transformation_feats)


        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id], [len(self.train_set[task_id]) - split, split])
        self.train_split[task_id] = train_split

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory,shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=int(self.batch_size * self.pc_valid),
                                                   num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory, shuffle=True)


        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = 'iImageNet-{}-{}'.format(task_id,self.task_ids[task_id])

        print ("Task ID: ", task_id)
        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Validation set size: {} images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Train+Val  set size: {} images of {}x{}".format(len(valid_loader.dataset)+len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        return self.dataloaders
