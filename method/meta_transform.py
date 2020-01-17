import random

class NWays(object):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)
    **Description**
    Keeps samples from N random labels present in the task description.
    **Arguments**
    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **n** (int, *optional*, default=2) - Number of labels to sample from the task
        description's labels.
    """

    def __init__(self, dataset, n=2, sort = True):
        self.n = n
        self.dataset = dataset
        self.sort = sort

    def __call__(self, task_description):
        classes = list(set([self.dataset.indices_to_labels[dd.index] for dd in task_description]))
        classes = random.sample(classes, k=self.n)

        if self.sort:
        	classes.sort()

        return [dd for dd in task_description if self.dataset.indices_to_labels[dd.index] in classes]