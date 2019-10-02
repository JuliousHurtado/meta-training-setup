  
#!/usr/bin/env python3

from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.omniglot import Omniglot

import numpy as np

class FullOmniglot(Dataset):
    """
    [[Source]]()
    **Description**
    This class provides an interface to the Omniglot dataset.
    The Omniglot dataset was introduced by Lake et al., 2015.
    Omniglot consists of 1623 character classes from 50 different alphabets, each containing 20 samples.
    While the original dataset is separated in background and evaluation sets,
    this class concatenates both sets and leaves to the user the choice of classes splitting
    as was done in Ravi and Larochelle, 2017.
    The background and evaluation splits are available in the `torchvision` package.
    **References**
    1. Lake et al. 2015. “Human-Level Concept Learning through Probabilistic Program Induction.” Science.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.
    **Arguments**
    * **root** (str) - Path to download the data.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.
    **Example**
    ~~~python
    omniglot = l2l.vision.datasets.FullOmniglot(root='./data',
                                                transform=transforms.Compose([
                                                    l2l.vision.transforms.RandomDiscreteRotation(
                                                        [0.0, 90.0, 180.0, 270.0]),
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=True)
    omniglot = l2l.data.MetaDataset(omniglot)
    ~~~
    """

    def __init__(self, root, transform=None, target_transform=None, download=False, to_color = False):
        self.transform = transform
        self.target_transform = target_transform
        self.to_color = to_color

        # Set up both the background and eval dataset
        omni_background = Omniglot(root, background=True, download=download)
        # Eval labels also start from 0.
        # It's important to add 964 to label values in eval so they don't overwrite background dataset.
        omni_evaluation = Omniglot(root,
                                   background=False,
                                   download=download,
                                   target_transform=lambda x: x + len(omni_background._characters))

        self.dataset = ConcatDataset((omni_background, omni_evaluation))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, character_class = self.dataset[item]
        if self.to_color:
            image = self.toColor(image)
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def toColor(self, img):
        # We make the assumption that the images are square.
        #side = int(np.sqrt(img.shape[0]))
        # To load an array as a PIL.Image we must first reshape it to 2D.
        # img = Image.fromarray(img.reshape((side, side)))
        img = img.convert('RGB')
        return img
