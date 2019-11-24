#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.omniglot import Omniglot

import numpy as np

class RandomSet(Dataset):

    def __init__(self, root = '', transform=None, target_transform=None, download=False, to_color = False):
        self.transform = transform

        self.x = torch.rand(500, 3, 224, 224)
        self.y = torch.randint(0, 20, (500,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]

        return x, y
