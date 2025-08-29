from .generators import *
from .classifier import *

import torch
import torch.nn as nn


def load_generator(model2load, dataset, **kwargs):
    """
    Load a pretrained network

    Args:
        model2load (str):
        dataset: the dataset
        use_gpu: boolean, whether or not to use GPU
        kwargs: the keyword arguments of the constructor
    """
    constructor = {
        'mnist': MNIST_Generator,
        'fmnist': G_FMNIST,
        'old_mnist': OLD_MNIST_Generator,
        'cifar10': CIFAR10_Generator,
        'lsun': LSUN_Generator,
        'celeba': Celeba_Generator,
        'celeba2': G_CelebA,
        'old_celeba': OLD_Celeba_Generator,
    }

    if dataset.lower() not in constructor.keys():
        raise ValueError('Unrecognized dataset: %s' % dataset)

    if dataset == 'fmnist' or dataset == 'celeba2':
        kwargs['activation'] = nn.ELU
    if dataset == 'celeba2':
        kwargs['batch_norm'] = True

    netG = constructor[dataset.lower()](**kwargs)
    netG.eval()

    if model2load is not None:
        ckpt = torch.load(model2load, map_location='cpu')
        if dataset == 'celeba2' or dataset == 'fmnist':
            netG = nn.DataParallel(netG)
        netG.load_state_dict(ckpt)

    return netG


def load_classifier(model2load, dataset, **kwargs):
    """
    Load a classifier network

    Args:
        model2load (str)
        dataset: the dataset
        kwargs: the keyword argument of the constructor
    """
    constructor = {
        'mnist': MNIST_Classifier,
        'fmnist': Classifier_FMNIST,
        'celeba2': Classifier_CelebA,
    }

    if dataset == 'fmnist' or dataset == 'celeba2':
        kwargs['activation'] = nn.ELU

    if dataset.lower() not in constructor.keys():
        raise ValueError('Unrecognized dataset: %s' % dataset)

    classifier = constructor[dataset.lower()](**kwargs)
    classifier.eval()

    if model2load is not None:
        ckpt = torch.load(model2load, map_location='cpu')
        if dataset == 'celeba2':
            classifier = nn.DataParallel(classifier)
        classifier.load_state_dict(ckpt)

    return classifier


