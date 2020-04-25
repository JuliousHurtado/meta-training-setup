import copy

import torch
from torch import nn
from torch.nn import functional as F

import torchvision.models as models
from learn2learn.vision.models import MiniImagenetCNN

class TaskEspecific(nn.Module):
    def __init__(self, filters, device):
        super(TaskEspecific, self).__init__()

        self.features = self.getExtractor()
        self.filters = filters
        self.grads = {}

        mlp = []
        for elem in filters:
            #Partamos simple con la función que crea estos filtros
            s = filters[elem]['sizes']
            if filters[elem]['n_filters'] > 0:
                mlp.append(nn.Linear(512, filters[elem]['n_filters']*s[1]*s[2]*s[3]))
        self.mlp = nn.Sequential(*mlp)

    def getExtractor(self):
        extractor = models.resnet18(pretrained=True)
        modules = list(extractor.children())[:-1]
        extractor = nn.Sequential(*modules)
        for p in extractor.parameters():
            p.requires_grad = False

        return extractor

    # def saveGrad(self, name):
    #     def hook(grad):
    #         #print(grad)
    #         self.grads[name] = grad
    #     return hook

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        f = {}
        for i, elem in enumerate(self.filters):
            s = self.filters[elem]['sizes']
            if self.filters[elem]['n_filters'] > 0:
                f[elem] =  self.mlp[i](x).view(-1,self.filters[elem]['n_filters'],s[1],s[2],s[3]).mean(dim=0)
        return f

class TaskModel(nn.Module):
    """docstring for TaskModel"""
    def __init__(self, path_pre_tranied_model, percentage_filter, split_batch, device, load_meta_model=True):
        super(TaskModel, self).__init__()

        self.linear_clfs = {}
        self.device = device
        self.p_filter = percentage_filter
        self.split_batch = split_batch

        self.meta_model = MiniImagenetCNN(5)
        if load_meta_model:
            self.loadMetaModel(path_pre_tranied_model)

        filters = self.getFilterZero()

        self.task_model = TaskEspecific(filters, device)

    def setLinear(self, task, num_classes=0):
        if task not in self.linear_clfs:
            self.linear_clfs[task] = nn.Linear(self.meta_model.linear.weight.size(1), num_classes).to(self.device)
        self.meta_model.linear = self.linear_clfs[task]

    def loadMetaModel(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_model.load_state_dict(checkpoint['checkpoint'])

    def getFilterZero(self):
        selected_filters = {}
        count = 0
        for elem in self.meta_model.base:
            for p in elem.children() :
                if isinstance(p, torch.nn.Conv2d):
                    sizes = p.weight.size()
                    temp = p.weight.view(sizes[0],-1)       
                    n = temp.norm(2,1)#.sum(1)

                    index = list(range(sizes[0]))
                    _,index = zip(*sorted(zip(n, index)))

                    if self.p_filter == 0.0:
                        f_replace = None
                        no_index = index
                        n_filters = 0
                    elif self.p_filter == 1.0:
                        f_replace = index
                        no_index = None
                        n_filters = sizes[0]
                    else:
                        f_replace = torch.tensor(index[:int(sizes[0]*self.p_filter)]).to(self.device)
                        no_index = torch.tensor(index[int(sizes[0]*self.p_filter):]).to(self.device)
                        n_filters = int(sizes[0]*self.p_filter)
                    
                    selected_filters[count] = { 
                                'index': f_replace,
                                'no_index': no_index,
                                'n_filters': n_filters,
                                'sizes': sizes}
                    #p.weight[selected_filters[count]['index']].mul(0)

                    count += 1

        return selected_filters

    # def setFilters(self, f):
    #     count = 0
    #     for elem in self.meta_model.base:
    #         for p in elem.children():
    #             if isinstance(p, torch.nn.Conv2d):
    #                 p.weight[self.task_model.filters[count]['index']].add(f[count])
    #                 count += 1

    # def printGradConv(self):
    #     count = 0
    #     for elem in self.meta_model.base:
    #         for p in elem.children():
    #             if isinstance(p, torch.nn.Conv2d):
    #                 print(p.weight[self.task_model.filters[count]['index']].grad)
    #                 count += 1

    def getTaskParameters(self, linear=True):
        params = []
        for p in self.task_model.mlp.parameters():
            params.append(p)

        if linear:
            for p in self.meta_model.linear.parameters():
                params.append(p)

        return params

    # def saveGradTask(self):
    #     for i, elem in enumerate(self.task_model.filters):
    #         self.task_model.mlp[i].weight.register_hook(print)

    def forward(self, x, y):
        if self.training and self.split_batch:
            p = int(x.size(0)/2)
            x1 = x[:p]
            y1 = y[:p]
            x2 = x[p:]
            y2 = y[p:]
        else:
            x1 = x[:]
            x2 = x[:]
            y2 = y

        f = self.task_model(x1)
        for i, elem in enumerate(self.meta_model.base):
            if self.p_filter == 0.0:
                x2 = F.conv2d(x2, elem.conv.weight, None, elem.conv.stride, elem.conv.padding, elem.conv.dilation, elem.conv.groups)
            elif self.p_filter == 1.0:
                x2 = F.conv2d(x2, f[i], None, elem.conv.stride, elem.conv.padding, elem.conv.dilation, elem.conv.groups)
            else:
                x_1 = F.conv2d(x2, elem.conv.weight[self.task_model.filters[i]['no_index']], None, elem.conv.stride, elem.conv.padding, elem.conv.dilation, elem.conv.groups)
                x_2 = F.conv2d(x2, f[i], None, elem.conv.stride, elem.conv.padding, elem.conv.dilation, elem.conv.groups)
                x2 = torch.cat([x_1,x_2], dim=1)
            
            x2 = elem.normalize(x2)
            x2 = elem.relu(x2)
            x2 = elem.max_pool(x2)

        out = self.meta_model.linear(x2.view(-1, 25 * 32))
        return out, y2

    def forwardNoHead(self, x):
        x1 = x[:]
        x2 = x[:]
        f = self.task_model(x1)
        for i, elem in enumerate(self.meta_model.base):
            if self.p_filter == 0.0:
                x2 = F.conv2d(x2, elem.conv.weight, None, elem.conv.stride, elem.conv.padding, elem.conv.dilation, elem.conv.groups)
            elif self.p_filter == 1.0:
                x2 = F.conv2d(x2, f[i], None, elem.conv.stride, elem.conv.padding, elem.conv.dilation, elem.conv.groups)
            else:
                x_1 = F.conv2d(x2, elem.conv.weight[self.task_model.filters[i]['no_index']], None, elem.conv.stride, elem.conv.padding, elem.conv.dilation, elem.conv.groups)
                x_2 = F.conv2d(x2, f[i], None, elem.conv.stride, elem.conv.padding, elem.conv.dilation, elem.conv.groups)
                x2 = torch.cat([x_1,x_2], dim=1)
            
            x2 = elem.normalize(x2)
            x2 = elem.relu(x2)
            x2 = elem.max_pool(x2)

        return x2

    def forwardOnlyHead(self, x):
        if self.training and self.split_batch:
            p = int(x.size(0)/2)
            x1 = x[:p]
            x2 = x[p:]
        else:
            x1 = x[:]
            x2 = x[:]

        out = self.meta_model.linear(x.view(-1, 25*32))
        return out