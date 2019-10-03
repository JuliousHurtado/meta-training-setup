import torch
from scipy.stats import truncnorm
from torch import nn

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
                             channels=1,
                             max_pool=False,
                             layers=layers)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.linear.weight.data.normal_()
        self.linear.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.base(x)
        x = x.mean(dim=[2, 3])
        x = self.linear(x)
        return x