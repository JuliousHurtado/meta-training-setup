import random
import torch as th
from torchvision import transforms
import learn2learn as l2l
from PIL.Image import LANCZOS
import argparse

from methods.maml import MAML
from methods.meta_sgd import MetaSGD
from methods.proto_net import ProtoNet
from methods.transferMeta import TMAML
from methods.meta_restnet import MetaRestNet

from data.datasets.full_omniglot import FullOmniglot
from data.datasets.mini_imagenet import MiniImagenet
from data.datasets.randomSet import RandomSet

def getRandomDataset(ways):
    tasks_list = [20, 10, 10]
    generators = {'train': None, 'validation': None, 'test': None}
    for mode, tasks in zip(['train','validation','test'], tasks_list):
        dataset = RandomSet()
        dataset = l2l.data.MetaDataset(dataset)
        generators[mode] = l2l.data.TaskGenerator(dataset=dataset, ways=ways, tasks=tasks)

    return generators['train'], generators['validation'], generators['test']

def getDatasets(dataset, ways):
    tasks_list = [20000, 1024, 1024]
    generators = {'train': None, 'validation': None, 'test': None}
    if dataset == 'mini-imagenet':
        for mode, tasks in zip(['train','validation','test'], tasks_list):
            dataset = MiniImagenet(root='./data/data', mode=mode, 
                                transform = transforms.Compose([
                                    transforms.Resize(224, interpolation=LANCZOS),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]))

            dataset = l2l.data.MetaDataset(dataset)
            generators[mode] = l2l.data.TaskGenerator(dataset=dataset, ways=ways, tasks=tasks)
    else:
        omniglot = FullOmniglot(root='./data/data',
                                                transform=transforms.Compose([
                                                    l2l.vision.transforms.RandomDiscreteRotation(
                                                        [0.0, 90.0, 180.0, 270.0]),
                                                    transforms.Resize(224, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=False, to_color = True)

        omniglot = l2l.data.MetaDataset(omniglot)
        classes = list(range(1623))
        random.shuffle(classes)
        generators['train'] = l2l.data.TaskGenerator(dataset=omniglot,
                                                 ways=ways,
                                                 classes=classes[:1100],
                                                 tasks=20000)
        generators['validation'] = l2l.data.TaskGenerator(dataset=omniglot,
                                                 ways=ways,
                                                 classes=classes[1100:1200],
                                                 tasks=1024)
        generators['test'] = l2l.data.TaskGenerator(dataset=omniglot,
                                                ways=ways,
                                                classes=classes[1200:],
                                                tasks=1024)

    return generators['train'], generators['validation'], generators['test']

def saveValues(name_file, results, args):
    th.save({
            'results': results,
            'args': args
            }, name_file)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def getMetaAlgorithm(args, model, device):
    if args['algorithm'] == 'maml':
        meta_model = MAML(model, lr=args['fast_lr'], adaptation_steps = args['adaptation_steps'], 
                                device = device,
                                first_order=args['first_order'])
    elif args['algorithm'] == 'meta-sgd':
        meta_model = MetaSGD(model, adaptation_steps = args['adaptation_steps'], 
                                device = device,
                                lr=args['fast_lr'], 
                                first_order=args['first_order'])
    elif args['algorithm'] == 'protonet':
        meta_model = ProtoNet(model, device = device,
                                k_way = args['ways'],
                                n_shot = args['shots'])
    elif args['algorithm'] == 'tmaml':
        if args['min_used'] > 1:
            args['min_used'] = 1
        meta_model = TMAML(model, lr=args['fast_lr'], adaptation_steps = args['adaptation_steps'], 
                                min_used = args['meta_batch_size']*args['min_used'],
                                device = device,
                                first_order=args['first_order'])
    elif args['algorithm'] == 'meta-resnet':
        meta_model = MetaRestNet(model, lr=args['fast_lr'], adaptation_steps = args['adaptation_steps'], 
                                device = device, first_order=args['first_order'],
                                num_freeze_layers = args['freeze_block'])
    else:
        meta_model = model

    return meta_model