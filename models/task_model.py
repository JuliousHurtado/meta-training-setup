import copy

import torch
from torch import nn

import torchvision.models as models
from learn2learn.vision.models import MiniImagenetCNN

class TaskEspecific(nn.Module):
    def __init__(self, filters, device):
        super(TaskEspecific, self).__init__()

        self.features = self.getExtractor()
        self.filters = filters

        mlp = []
        for elem in filters:
            #Partamos simple con la función que crea estos filtros
            s = filters[elem]['sizes']
            mlp.append(nn.Linear(512, filters[elem]['n_filters']*s[1]*s[2]*s[3]))
        self.mlp = nn.Sequential(*mlp)

    def getExtractor(self):
        extractor = models.resnet18(pretrained=True)
        modules = list(extractor.children())[:-1]
        extractor = nn.Sequential(*modules)
        for p in extractor.parameters():
            p.requires_grad = False

        return extractor

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        f = {}
        for i, elem in enumerate(self.filters):
            s = self.filters[elem]['sizes']
            f[elem] =  self.mlp[i](x).view(-1,self.filters[elem]['n_filters'],s[1],s[2],s[3]).mean(dim=0)
        return f

class TaskModel(nn.Module):
    """docstring for TaskModel"""
    def __init__(self, path_pre_tranied_model, percentage_filter, device):
        super(TaskModel, self).__init__()

        self.linear_clfs = {}
        self.device = device
        self.p_filter = percentage_filter

        self.meta_model = MiniImagenetCNN(5)
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

                    selected_filters[count] = { 
                                'index': torch.tensor(index[:int(sizes[0]*self.p_filter)]).to(self.device),
                                'n_filters': int(sizes[0]*self.p_filter),
                                'sizes': sizes}
                    p.weight[selected_filters[count]['index']].mul(0)

                    count += 1

        return selected_filters

    def setFilters(self, f):
        count = 0
        for elem in self.meta_model.base:
            for p in elem.children():
                if isinstance(p, torch.nn.Conv2d):
                    p.weight[self.task_model.filters[count]['index']].add(f[count]) 
                    count += 1

    def forward(self, x, y):
        if self.training:
            p = int(x.size(0)/2)
            x1 = x[:p]
            y1 = y[:p]
            x2 = x[p:]
            y2 = y[p:]
        else:
            x1 = copy.deepcopy(x)
            x2 = copy.deepcopy(x)
            y2 = y

        f = self.task_model(x1)
        self.setFilters(f)
        out = self.meta_model(x2)

        return out, y2