# Base on the code: https://github.com/facebookresearch/Adversarial-Continual-Learning

import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

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

        self.conv2=nn.Conv2d(hiddens[0],hiddens[1],kernel_size=k_size[1])
        s=compute_conv_output_size(s,k_size[1])
        s=s//2

        self.conv3=nn.Conv2d(hiddens[1],hiddens[2],kernel_size=k_size[2])
        s=compute_conv_output_size(s,k_size[2])
        s=s//2

        self.maxpool=nn.MaxPool2d(2)
        self.relu=nn.ReLU()

        self.drop2=nn.Dropout(0.5)
        self.fc1=nn.Linear(hiddens[2]*s*s,self.latent_dim)

    def forward(self, x_s, mask):
        if len(x_s.size()) == 2:
            x_s = x_s.view(x_s.size(0), 1, 32, 32)

        h = self.maxpool(self.relu(self.conv1(x_s)))
        if mask:
            h = h * mask[0][0]

        h = self.maxpool(self.relu(self.conv2(h)))
        if mask:
            h = h * mask[1][0]

        h = self.maxpool(self.relu(self.conv3(h)))
        if mask:
            h = h * mask[2][0]

        h = h.view(x_s.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))

        return h

class Private(nn.Module):
    def __init__(self, args, hiddens, layers, ):
        super(Private, self).__init__()

        self.use_resnet = args.resnet18
        self.layers = layers
        self.hiddens = hiddens
        self.dim_embedding = args.latent_dim
        self.num_tasks = args.ntasks
        self.one_representation = args.use_one_representation
        self.use_pca = False

        if args.use_one_representation:
            self.num_tasks = 1

        if self.use_resnet:
            resnet18 = models.resnet18(pretrained=args.resnet_pre_trained)
            modules = list(resnet18.children())[:-1]
            self.feat_extraction = nn.Sequential(*modules)
            for p in self.feat_extraction.parameters():
                p.requires_grad = False

            self.num_ftrs = resnet18.fc.in_features
            # assert False, self.feat_extraction

            if args.use_pca:
                self.use_pca = True
                self.rank_matrix = {}

        else:
            if args.experiment == 'cifar100':
                hiddens=[32,32]
                flatten=1152

            elif args.experiment == 'miniimagenet':
                hiddens=[8,8]
                flatten=1800

            elif args.experiment == 'multidatasets':
                hiddens=[32,32]
                flatten=1152
            
            self.ncha,self.size,_=args.inputsize
            self.conv = torch.nn.ModuleList()

            for _ in range(self.num_tasks):
                layer = torch.nn.Sequential()
                layer.add_module('conv1',nn.Conv2d(self.ncha, hiddens[0], kernel_size=self.size // 8))
                layer.add_module('bn1', nn.BatchNorm2d(hiddens[0]))
                layer.add_module('relu1', nn.ReLU(inplace=True))
                # layer.add_module('drop1', nn.Dropout(0.2))
                layer.add_module('maxpool1', nn.MaxPool2d(2))
                layer.add_module('conv2', nn.Conv2d(hiddens[0], hiddens[1], kernel_size=self.size // 10))
                layer.add_module('bn2', nn.BatchNorm2d(hiddens[1]))
                layer.add_module('relu2', nn.ReLU(inplace=True))
                # layer.add_module('dropout2', nn.Dropout(0.5))
                layer.add_module('maxpool2', nn.MaxPool2d(2))
                layer.add_module('flatten', nn.Flatten())
                layer.add_module('linear1', nn.Linear(flatten,self.dim_embedding))
                layer.add_module('relu3', nn.ReLU(inplace=True))
                #layer.add_module('drop', nn.Dropout(0.5))
                self.conv.append(layer)
            self.num_ftrs = self.dim_embedding

        if args.use_relu:
            ac_funt = nn.ReLU()
        else:
            ac_funt = nn.Sigmoid()
            
        self.linear = nn.ModuleList()
        for i in range(args.ntasks):
            linear = nn.ModuleList()
            for j in range(self.layers):
                mask_lin = nn.Sequential(
                                nn.Linear(self.num_ftrs, int(self.hiddens[j])),
                                ac_funt, 
                            )
                linear.append(mask_lin)
            self.linear.append(linear)
                
    def forward(self, x, task_id):
        m = []

        if len(x.size()) == 2:
            x = x.view(x.size(0), 1, 224, 224)

        if self.use_resnet:
            x = self.feat_extraction(x).squeeze()
        else:
            if self.one_representation:
                x = self.conv[0](x)
            else:
                x = self.conv[task_id](x)

        if self.use_pca:
            for i in range(self.layers):
                m.append([torch.matmul(x, self.rank_matrix[task_id][:, :self.hiddens[i]]).unsqueeze(2).unsqueeze(3)])
        else:
            for i in range(self.layers):
                film_vector = self.linear[task_id][i](x.clone()).view(x.size(0), 1, self.hiddens[i])
                m.append([
                    film_vector[:,0,:].unsqueeze(2).unsqueeze(3),
                    ])

        return m, x

    def train_pca(self, dataloader, task_id, device):
        x = []
        for batch in dataloader:
            input = batch[0].to(device)
            x.append(self.feat_extraction(input).squeeze().cpu())
        
        x = torch.cat(x)
        (_, _, V) = torch.pca_lowrank(x.clone(), q=self.hiddens[2], niter=5)
        self.rank_matrix[task_id] = V.to(device).clone()

class Net(nn.Module):

    def __init__(self, args, device):
        super(Net, self).__init__()
        self.ncha,size,_=args.inputsize
        self.taskcla=args.taskcla
        self.num_tasks = args.ntasks
        self.image_size = self.ncha*size*size
        self.args=args

        if args.experiment == 'cifar100':
            hiddens = [64, 128, 256, 512, 512]

        elif args.experiment == 'miniimagenet':
            hiddens = [64, 128, 256, 512, 512]

        elif args.experiment == 'multidatasets':
            hiddens = [64, 128, 256, 512, 512]

        elif args.experiment == 'mnist5' or args.experiment == 'pmnist':
            hiddens = [32, 64, 128, 256, 256]
            
        elif args.experiment == 'imagenet':
            hiddens = [64, 128, 256, 512, 512]
        else:
            raise NotImplementedError

        self.use_share = args.use_share
        self.use_mask = args.use_mask
        self.only_shared = args.only_shared

        self.shared = None
        self.private = None
        if args.use_share:
            self.shared = Shared(args, hiddens)
        if not self.only_shared:
            self.private = Private(args, hiddens, 3)

        self.latent_dim = args.latent_dim

        self.head = nn.ModuleList()
        for i in range(self.num_tasks):
            self.head.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.taskcla[i][1])
                ))


    def forward(self, x, task_id, inputs_feats, shared_clf = False, task_pri = None):
        if self.only_shared:
            m_p = None
        else:
            x_p = x.clone()
            m_p, x_p = self.private(x_p, task_id)

            if not self.use_mask:
                m_p = [ [torch.ones_like(m[0])] for m in m_p ]

        x_s = self.shared(x.clone(), m_p)
        
        if shared_clf:
            pred = self.shared_clf(x_s)
        else:
            pred = self.head[task_id](x_s)

        return pred, 0

    def print_model_size(self):
        if self.private:
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

    def get_masks(self, x, task_id, inputs_feats):
        if self.only_shared:
            return None
        if self.private.use_resnet:
            x_p = inputs_feats
        else:
            x_p = x.clone()
        m_p, _ = self.private(x_p, task_id)
        return m_p

    def get_features(self, x, task_id, inputs_feats):
        if self.private.use_resnet:
            x_p = inputs_feats
        else:
            x_p = x.clone()

        if self.private.use_resnet:
            x = self.private.feat_extraction(x_p).squeeze()
        else:
            if self.private.one_representation:
                x = self.private.conv[0](x)
            else:
                x = self.private.conv[task_id](x_p)

        return x

    def get_mask_from_features(self, x, task_id):
        m = []
        for i in range(self.private.layers):
            film_vector = self.private.linear[task_id][i](x.clone()).view(x.size(0), 1, self.private.hiddens[i])
            m.append([
                film_vector[:,0,:].unsqueeze(2).unsqueeze(3),
                ])
        return m