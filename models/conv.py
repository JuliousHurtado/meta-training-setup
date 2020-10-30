# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from .regularization import MaskRegularization, DiffRegularization

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Shared(nn.Module):

    def __init__(self,args, hiddens):
        super(Shared, self).__init__()

        self.ncha,size,_=args.inputsize
        self.taskcla=args.taskcla
        self.latent_dim = args.latent_dim

        k_size = [size//8, size//10, 2]

        self.conv1=nn.Conv2d(self.ncha,hiddens[0],kernel_size=k_size[0])
        s=compute_conv_output_size(size,k_size[0])
        s=s//2
        #self.bn1 = nn.BatchNorm2d(hiddens[0], affine=True, track_running_stats=False)
        self.conv2=nn.Conv2d(hiddens[0],hiddens[1],kernel_size=k_size[1])
        s=compute_conv_output_size(s,k_size[1])
        s=s//2
        #self.bn2 = nn.BatchNorm2d(hiddens[1], affine=True, track_running_stats=False)
        self.conv3=nn.Conv2d(hiddens[1],hiddens[2],kernel_size=k_size[2])
        s=compute_conv_output_size(s,k_size[2])
        s=s//2
        #self.bn3 = nn.BatchNorm2d(hiddens[2], affine=True, track_running_stats=False)
        self.maxpool=nn.MaxPool2d(2)
        self.relu=nn.ReLU()

        # self.drop1=nn.Dropout(0.2)
        self.drop2=nn.Dropout(0.5)
        self.fc1=nn.Linear(hiddens[2]*s*s,self.latent_dim)
        #self.bn4=nn.BatchNorm1d(hiddens[3], affine=True, track_running_stats=False)
        #self.fc2=nn.Linear(hiddens[3],self.latent_dim)
        #self.bn5=nn.BatchNorm1d(self.latent_dim, affine=True, track_running_stats=False)
        #self.fc3=nn.Linear(hiddens[4],self.latent_dim)
        # self.bn6=nn.BatchNorm1d(hiddens[5], affine=True, track_running_stats=False)
        # self.fc4=nn.Linear(hiddens[5], self.latent_dim)
        # self.bn7=nn.BatchNorm1d(self.latent_dim, affine=True, track_running_stats=False)

    def forward(self, x_s, mask):
        # print(x_s.size())
        if len(x_s.size()) == 2:
            x_s = x_s.view(x_s.size(0), 1, 28, 28)

        #h = self.maxpool(self.relu(self.bn1(self.conv1(x_s))))
        h = self.maxpool(self.relu(self.conv1(x_s)))
        if mask:
            h = h * mask[0][0]# + mask[0][1]
        #h = self.maxpool(self.relu(self.bn2(self.conv2(h))))
        h = self.maxpool(self.relu(self.conv2(h)))
        if mask:
            h = h * mask[1][0]# + mask[1][1]
        #h = self.maxpool(self.relu(self.bn3(self.conv3(h))))
        h = self.maxpool(self.relu(self.conv3(h)))
        if mask:
            h = h * mask[2][0]# + mask[2][1]
        h = h.view(x_s.size(0), -1)
        #h = self.drop2(self.relu(self.bn4(self.fc1(h))))
        #h = self.drop2(self.relu(self.bn4(self.fc2(h))))
        h = self.drop2(self.relu(self.fc1(h)))
        #h = self.drop2(self.relu(self.fc2(h)))
        #h = self.drop2(self.relu(self.fc3(h)))
        # h = self.drop2(self.bn7(self.relu(self.fc4(h))))
        return h

class Private(nn.Module):
    def __init__(self, args, hiddens, layers):
        super(Private, self).__init__()

        self.ncha,self.size,_=args.inputsize
        self.taskcla=args.taskcla
        self.latent_dim = args.latent_dim
        self.num_tasks = args.ntasks
        self.layers = layers
        self.use_resnet = args.resnet18

        k_size = [self.size//8, self.size//10, 2]
        hiddens = [ int(h/2) for h in hiddens ]
        hiddens.insert(0, self.ncha)

        self.conv = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.last_em = nn.ModuleList()
        for i in range(self.num_tasks):
            conv = nn.ModuleList()
            linear = nn.ModuleList()

            s = self.size
            for j in range(self.layers):
                layer = nn.Sequential()
                layer.add_module('conv{}'.format(j+1), nn.Conv2d(hiddens[j], hiddens[j+1], kernel_size=k_size[j]))
                layer.add_module('bn{}'.format(j+1), nn.BatchNorm2d(hiddens[j+1]))
                #layer.add_module('drop{}'.format(j+1), nn.Dropout(0.2))
                layer.add_module('relu{}'.format(j+1), nn.ReLU(inplace=True))
                layer.add_module('maxpool{}'.format(j+1), nn.MaxPool2d(2))
                conv.append(layer)

                mask_lin = nn.Sequential()
                mask_lin.add_module('linear{}'.format(j+1), nn.Linear(hiddens[j+1],hiddens[j+1]*2))
                #mask_lin.add_module('relu{}'.format(j+1), nn.ReLU(inplace=True))
                #mask_lin.add_module('sigmoid{}'.format(j+1), nn.Sigmoid())
                linear.append(mask_lin)

                s=compute_conv_output_size(s,k_size[j])
                s=s//2

            last_em = nn.Sequential(
                                nn.Linear(hiddens[j+1]*s*s, self.latent_dim),
                                #nn.BatchNorm1d(self.latent_dim),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(0.5),
                    )

            self.conv.append(conv)
            self.linear.append(linear)
            self.last_em.append(last_em)

    def forward(self, x, task_id):
        m = []

        if len(x.size()) == 2:
            x = x.view(x.size(0), 1, 28, 28)

        for i in range(self.layers):
            x = self.conv[task_id][i](x)
            x_m = x.clone().view(x.size(0),x.size(1),-1).mean(dim=2)
            #x_m = x.clone().view(x.size(0),x.size(1),-1).max(dim=2)
            m.append(self.linear[task_id][i](x_m).unsqueeze(2).unsqueeze(3))

        x = x.view(x.size(0), -1)
        x = self.last_em[task_id](x)

        return m, x

class PrivateResnet(nn.Module):
    def __init__(self, args, hiddens, layers, ):
        super(PrivateResnet, self).__init__()

        self.use_resnet = args.resnet18
        self.layers = layers
        self.hiddens = hiddens

        if self.use_resnet:
            resnet18 = models.resnet18(pretrained=args.resnet_pre_trained)
            modules = list(resnet18.children())[:-1]
            self.feat_extraction = nn.Sequential(*modules)
            for p in self.feat_extraction.parameters():
                p.requires_grad = False

            self.num_ftrs = resnet18.fc.in_features
            
        else:
            self.ncha,self.size,_=args.inputsize
            hiddens = [ int(h/2) for h in hiddens ]
            hiddens.insert(0, self.ncha)
            k_size = [self.size//8, self.size//10, 2]

            self.conv = nn.ModuleList()
            for i in range(args.ntasks):
                s = self.size
                layer = nn.Sequential()
                for j in range(self.layers):
                    layer.add_module('conv{}'.format(j+1), nn.Conv2d(hiddens[j], hiddens[j+1], kernel_size=k_size[j]))
                    layer.add_module('bn{}'.format(j+1), nn.BatchNorm2d(hiddens[j+1]))
                    layer.add_module('relu{}'.format(j+1), nn.ReLU(inplace=True))
                    layer.add_module('maxpool{}'.format(j+1), nn.MaxPool2d(2))
                    #conv.append(layer)

                    s=compute_conv_output_size(s,k_size[j])
                    s=s//2

                self.conv.append(layer)
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.num_ftrs = hiddens[j+1]#*s*s

        self.linear = nn.ModuleList()
        self.last_em = nn.ModuleList()
        for i in range(args.ntasks):
            linear = nn.ModuleList()
            for j in range(self.layers):
                mask_lin = nn.Sequential(
                                nn.Linear(self.num_ftrs, int(self.hiddens[j])),
                                # nn.Linear(self.num_ftrs, int(self.hiddens[j]/2)),
                                nn.ReLU(),
                                # nn.Linear(int(self.hiddens[j]/2), self.hiddens[j]),
                                #nn.Dropout(0.25),
                            )
                linear.append(mask_lin)
            # mask_lin = nn.Linear(self.num_ftrs, 2*self.hiddens[-1])
            self.last_em.append(nn.Sequential(
                        nn.Linear(self.num_ftrs, args.latent_dim),
                        nn.ReLU(),
                        #nn.Dropout(0.5),
                        #nn.Linear(args.latent_dim, args.latent_dim),
                        #nn.ReLU(),
                        #nn.Dropout(0.5)
                        ))
            self.linear.append(linear)

    def forward(self, x, task_id):
        m = []

        if len(x.size()) == 2:
            x = x.view(x.size(0), 1, 224, 224)

        if self.use_resnet:
            x = self.feat_extraction(x).squeeze()
        else:
            x = self.conv[task_id](x)
            x = self.avgpool(x).squeeze()
            #x = self.conv[task_id](x).squeeze().view(x.size(0),self.num_ftrs)

        for i in range(self.layers):
            film_vector = self.linear[task_id][i](x.clone()).view(x.size(0), 1, self.hiddens[i])
            m.append([
                film_vector[:,0,:].unsqueeze(2).unsqueeze(3),
                ])
        # film_vector = self.linear[task_id][0](x.clone()).view(
        #     x.size(0), -1, self.hiddens[-1])
        # m.append([
        #         film_vector[:,0,:self.hiddens[0]].unsqueeze(2).unsqueeze(3),
        #         film_vector[:,1,:self.hiddens[0]].unsqueeze(2).unsqueeze(3),
        #         ])
        # for i in range(1,self.layers):
        #     m.append([
        #         film_vector[:,0,self.hiddens[i-1]:self.hiddens[i]].unsqueeze(2).unsqueeze(3),
        #         film_vector[:,1,self.hiddens[i-1]:self.hiddens[i]].unsqueeze(2).unsqueeze(3),
        #         ])

        x = self.last_em[task_id](x)

        return m, x

class Net(nn.Module):

    def __init__(self, args, device):
        super(Net, self).__init__()
        self.ncha,size,_=args.inputsize
        self.taskcla=args.taskcla
        self.num_tasks = args.ntasks
        self.image_size = self.ncha*size*size
        self.args=args

        self.hidden1 = args.head_units
        self.hidden2 = args.head_units//2


        if args.experiment == 'cifar100':
            hiddens = [64, 128, 256, 512, 512]

        elif args.experiment == 'miniimagenet':
            hiddens = [64, 128, 256, 512, 512]

        elif args.experiment == 'multidatasets':
            hiddens = [64, 128, 256, 512, 512]

        elif args.experiment == 'mnist5' or args.experiment == 'pmnist':
            hiddens = [32, 64, 128, 256, 256]

        else:
            raise NotImplementedError

        self.regs = []
        if args.mask_reg:
            self.regs.append(MaskRegularization(args.mask_theta))

        # if args.diff_reg:
        #     self.regs.append(DiffRegularization(args.diff_theta))

        self.shared = None
        self.private = None
        if args.use_share:
            self.shared = Shared(args, hiddens)
        if args.use_private:
            #self.private = Private(args, hiddens, 3)
            self.private = PrivateResnet(args, hiddens, 3)

        self.con_pri_shd = args.con_pri_shd
        self.use_share = args.use_share
        self.use_private = args.use_private
        self.use_mask = args.use_mask
        # self.diff_pri_shar = args.diff_pri_shar

        self.latent_dim = args.latent_dim
        if self.con_pri_shd:
            self.latent_dim *= 2

        self.drop = nn.Dropout(0.5)
        self.head = nn.ModuleList()
        for i in range(self.num_tasks):
            self.head.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden1),
                    #nn.BatchNorm1d(self.hidden1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(self.hidden1, self.hidden2),
                    #nn.BatchNorm1d(self.hidden2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden2, self.taskcla[i][1])
                ))


    def forward(self, x_s, x_p, task_id, use_only_share=False):
        if self.use_private:
            m_p, x_p = self.private(x_p, task_id)
        else:
            m_p = None

        #m_p = [ [torch.ones_like(m[0]),torch.ones_like(m[1])] for m in m_p ]

        if not self.use_mask:
            m_p = None

        if self.use_share:
            #if self.diff_pri_shar:
            #    x_s = torch.cat((x_s[:int(x_p.size(0)/2)], x_s[int(x_p.size(0)/2):]))
            x_s = self.shared(x_s, m_p)
        else:
            x_s = torch.zeros_like(x_p)
        
        if use_only_share:
            x_s = torch.nn.functional.normalize(x_s, dim=1)
            x = torch.cat([torch.zeros_like(x_p), x_s], dim=1)
        elif self.use_share and self.use_private and self.con_pri_shd:
            x_s = torch.nn.functional.normalize(x_s, dim=1)
            x_p = torch.nn.functional.normalize(x_p, dim=1)
            x = torch.cat([x_p, x_s], dim=1)
        elif self.use_share:
            x = x_s
        else:
            x = x_p

        loss = 0
        # FIX
        # for reg in self.regs:
        #     loss += reg(m_p)

        return self.head[task_id](x), loss

    def forward2(self, x, task_id):
        m_p, x_p = self.private(x.clone(), task_id)
        #m_p = [ [torch.ones_like(m[0])] for m in m_p ]
        reg_loss = 0.0
        #for m in m_p:
        #    reg_loss += m[0].abs().sum()/m[0].size(0)
        x_s = self.shared(x.clone(), m_p)
        #x_s = torch.zeros_like(x_p)
        #x_p = torch.zeros_like(x_p)
        # x_s = torch.nn.functional.normalize(x_s, dim=1)
        # x_p = torch.nn.functional.normalize(x_p, dim=1)
        # x_s /= x_s.sum(dim=1).unsqueeze(1)
        # x_p /= x_p.sum(dim=1).unsqueeze(1)
        x = self.drop(torch.cat([x_p, x_s], dim=1))
        return self.head[task_id](x), reg_loss

    def forward3(self, x, task_id):
        m_p, x_p = self.private(x.clone(), task_id)
        x_s = self.shared(x.clone(), m_p)
        return self.shared_clf(x_s)

    def forward4(self, x, task_id):
        m_p, x_p = self.private(x.clone(), task_id)
        m_p = [ [torch.ones_like(m[0])] for m in m_p ]
        x_s = self.shared(x.clone(), m_p)
        return self.shared_clf(x_s)

    def print_model_size(self):
        if self.use_private:
            count_P = sum(p.numel() for p in self.private.parameters() if p.requires_grad)
            if self.private.use_resnet:
                count_pre_train = sum(p.numel() for p in self.private.feat_extraction.parameters())
            else:
                count_pre_train = 0
        else:
            count_P = 0
            count_pre_train = 0
        if self.use_share:
            count_S = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
        else:
            count_S = 0
        count_H = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        print('Num parameters in S \t = %s ' % (self.pretty_print(count_S)))
        print('Num parameters in P \t = %s,  per task = %s ' % (self.pretty_print(count_P),self.pretty_print(count_P/self.num_tasks)))
        print('Num parameters in p \t = %s,  per task = %s ' % (self.pretty_print(count_H),self.pretty_print(count_H/self.num_tasks)))
        print('Num parameters in P+p \t = %s ' % self.pretty_print(count_P+count_H))
        print('-------------------------->   Architecture size: %s parameters (%sB)' % (self.pretty_print(count_S + count_P + count_H),
                                                                    self.pretty_print(4*(count_S + count_P + count_H))))
        print("------------------------------------------------------------------------------")
        print("                               TOTAL:  %sB" % self.pretty_print(4*(count_S + count_P + count_H)))
        print("------------------------------------------------------------------------------")
        print('Num parameters in Pre-train \t = %s ' % self.pretty_print(count_pre_train))
        print("                               Parameters: \t %s" % self.pretty_print((count_S + count_P + count_H + count_pre_train)))
        print("                               New TOTAL: \t %sB" % self.pretty_print(4*(count_S + count_P + count_H + count_pre_train)))
        print("------------------------------------------------------------------------------")

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

