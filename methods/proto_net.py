#!/usr/bin/env python
# coding: utf-8

# The code is adapted from Oscar Knagg
# https://github.com/oscarknagg/few-shot
# and he has a great set of medium articles around it.

import argparse

import numpy as np
import torch
from PIL.Image import LANCZOS
from torch.nn import Module
from torch.optim import Adam
from torch.optim import Optimizer
from torchvision import transforms
from typing import Callable

import learn2learn as l2l
from learn2learn.vision.datasets.full_omniglot import FullOmniglot
from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.vision.models import OmniglotCNN


class ProtoNet(BaseLearner):
    def __init__(self, model, device, k_way, n_shot, matching_fn = 'l2'):
        super(ProtoNet, self).__init__()
        self.model = model
        self.device = device

        self.k_way = k_way
        self.n_shot = n_shot
        self.matching_fn = matching_fn

    def meta_train(self, adaptation_data, evaluation_data, loss):
        x_support = torch.stack(adaptation_data.data).float().to(self.device)
        y_support = torch.LongTensor(adaptation_data.label).to(self.device)

        x_query = torch.stack(evaluation_data.data).float().to(self.device)
        x_support_query = torch.cat([x_support, x_query], dim=0)

        return self.episode(loss, x_support_query, y_support)

    def episode(self, loss, x, y):

        embeddings = self.model(x)
        support = embeddings[:self.n_shot * self.k_way]
        queries = embeddings[self.n_shot * self.k_way:]
        prototypes = self.compute_prototypes(support)
        distances = self.pairwise_distances(queries, prototypes)

        log_p_y = (-distances).log_softmax(dim=1)
        loss = loss(log_p_y, y)
        y_pred = (-distances).softmax(dim=1)

        return loss, y_pred

    def compute_prototypes(self, support):
        """Compute class prototypes from support samples.
        # Arguments
            support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
                dimension.
            k: int. "k-way" i.e. number of classes in the classification task
            n: int. "n-shot" of the classification task
        # Returns
            class_prototypes: Prototypes aka mean embeddings for each class
        """
        class_prototypes = support.reshape(self.k_way, self.n_shot, -1).mean(dim=1)

        return class_prototypes


    def pairwise_distances(self, x, y):
        """Efficiently calculate pairwise distances (or other similarity scores) between
        two sets of samples.
        # Arguments
            x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
            y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
            matching_fn: Distance metric/similarity score to compute between samples
        """
        n_x = x.shape[0]
        n_y = y.shape[0]

        if self.matching_fn == 'l2':
            distances = (
                    x.unsqueeze(1).expand(n_x, n_y, -1) -
                    y.unsqueeze(0).expand(n_x, n_y, -1)
            ).pow(2).sum(dim=2)
            return distances
        elif self.matching_fn == 'cosine':
            normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
            normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

            expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
            expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

            cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
            return 1 - cosine_similarities
        elif self.matching_fn == 'dot':
            expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
            expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

            return -(expanded_x * expanded_y).sum(dim=2)
        else:
            raise (ValueError('Unsupported similarity function'))

    def categorical_accuracy(self, y, y_pred):
        return torch.eq(y_pred.argmax(dim=-1), y).sum() / y_pred.shape[0]