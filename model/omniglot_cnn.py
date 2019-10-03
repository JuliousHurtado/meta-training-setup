import torch
from scipy.stats import truncnorm
from torch import nn

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
                                        eps=1e-3,
                                        momentum=0.999,
                                        track_running_stats=False,
                                        )
        # TODO: Add BN bias.
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

    # NOTE:
    #     Omniglot: hidden=64, channels=1, no max_pool
    #     MiniImagenet: hidden=32, channels=3, max_pool

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

class OmniglotCNN(nn.Module):
    """
    [Source]()
    **Description**
    The convolutional network commonly used for Omniglot, as described by Finn et al, 2017.
    This network assumes inputs of shapes (1, 28, 28).
    **References**
    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.” ICML.
    **Arguments**
    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.
    **Example**
    ~~~python
    model = OmniglotCNN(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(self, output_size=5, hidden_size=64, layers=4):
        super(OmniglotCNN, self).__init__()
        self.hidden_size = hidden_size
        self.base = ConvBase(output_size=hidden_size,
                             hidden=hidden_size,
                             channels=3,
                             max_pool=True,
                             layers=layers,
                             max_pool_factor=4 // layers)
        self.linear = nn.Linear(25 * hidden_size, output_size, bias=True)
        maml_init_(self.linear)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.base(x)
        x = self.linear(x.view(-1, 25 * self.hidden_size))
        return x
