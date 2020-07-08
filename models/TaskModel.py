import torch
from torch import nn
import copy

def maml_init_(module):
    nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    nn.init.constant_(module.bias.data, 0.0)
    return module

class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 max_pool=True,
                 max_pool_factor=1.0):
        super(ConvBlock, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=stride,
                                         stride=stride,
                                         ceil_mode=False,
                                         )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = nn.BatchNorm2d(out_channels,
                                        affine=True,
                                        # eps=1e-3,
                                        # momentum=0.999,
                                        track_running_stats=False,
                                        )
        #nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=1,
                              bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x

class ConvBase(nn.Sequential):
    def __init__(self,
                 output_size,
                 hidden=64,
                 channels=1,
                 max_pool=False,
                 layers=4,
                 max_pool_factor=1.0):
        core = [ConvBlock(channels,
                          hidden,
                          (3, 3),
                          max_pool=max_pool,
                          max_pool_factor=max_pool_factor),
                ]
        for l in range(layers - 1):
            core.append(ConvBlock(hidden,
                                  hidden,
                                  kernel_size=(3, 3),
                                  max_pool=max_pool,
                                  max_pool_factor=max_pool_factor))
        super(ConvBase, self).__init__(*core)

class MiniImagenetCNN(nn.Module):
    def __init__(self, hidden_size=32, layers=4, channels=3):
        super(MiniImagenetCNN, self).__init__()
        self.base = ConvBase(output_size=hidden_size,
                             hidden=hidden_size,
                             channels=channels,
                             max_pool=True,
                             layers=layers,
                             max_pool_factor=4 // layers)

        self.hidden_size = hidden_size
        self.layers = layers
        self.linear = None

    def forward(self, x):
        x = self.base(x)
        x = self.linear(x.view(x.size(0), 4 * self.hidden_size))
        return x

    def setLinearLayer(self, layer):
        self.linear = layer


class TaskManager(nn.Module):
    def __init__(self, output_size, channels=3, hidden_size=32, layers=4, device = 'cpu'):
        super(TaskManager, self).__init__()

        self.task = []
        linear_layer = nn.Linear(4 * hidden_size, output_size).to(device)
        maml_init_(linear_layer)
        self.task.append(linear_layer)

        self.model = MiniImagenetCNN(hidden_size=hidden_size, channels=channels, layers=layers)

        self.setLinearLayer(0, output_size)

    def setLinearLayer(self, task, num_cls):
        if len(self.task) > task:
            self.model.setLinearLayer(self.task[task])
        else:
            if num_cls == self.task[-1].weight.size(0):
                self.task.append(copy.deepcopy(self.task[-1]))
            else:
                linear_layer = nn.Linear(self.task[-1].weight.size(1), num_cls).to(device)
                self.task.append(linear_layer)

            self.model.setLinearLayer(self.task[task])

    def getLinearParameters(self):
        return self.model.linear.parameters()

    def forward(self, x):
        return self.model(x)