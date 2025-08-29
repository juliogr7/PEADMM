import os
import sys
sys.path.insert(0, './')
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.classifier import MNIST_Classifier as MLP
from util.attack import PGM
from dataset import load_mnist
from util.evaluation import accuracy, AverageCalculator
from util.device_parser import config_visible_gpu, parse_device_alloc
from util.param_parser import IntListParser, DictParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 100,
        help = 'the batch size, default = 100')
    parser.add_argument('--input_dim', type = int, default = 784,
        help = 'the number of input dimensions, default = 784')
    parser.add_argument('--hidden_dims', action = IntListParser, default = [],
        help = 'the number of neurons in hidden layers, default = []')
    parser.add_argument('--output_class', type = int, default = 10,
        help = 'the number of classes, default = 10')

    parser.add_argument('--dataset', type = str, default = 'mnist',
        help = 'the dataset, default = "mnist", default = ["mnist", "svhn", "fmnist"]')
    parser.add_argument('--attacker', action = DictParser, default = None,
        help = 'the information of the attacker, format: step_size=XX,threshold=XX,iter_num=XX, default = None')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'the model to be loaded')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'choose which gpu to use, default = None')

    args = parser.parse_args()
    config_visible_gpu(args.gpu)
    if not args.gpu in ['cpu',] and torch.cuda.is_available():
        device = torch.device('cuda:0')
        device_ids = 'cuda'
    else:
        device = torch.device('cpu')
        device_ids = 'cpu'

    model = MLP(input_dim = args.input_dim, hidden_dims = args.hidden_dims, output_class = args.output_class)
    ckpt = torch.load(args.model2load)
    model.load_state_dict(ckpt)

    _, model = parse_device_alloc(device_config = None, model = model)
    criterion = nn.CrossEntropyLoss()
    if not device_ids in ['cpu',]:
        criterion = criterion.cuda(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, betas = (0.9, 0.99), weight_decay = 1e-4)

    if args.attacker is not None:
        attacker = PGM(step_size = args.attacker['step_size'], threshold = args.attacker['threshold'], iter_num = int(args.attacker['iter_num']))
    else:
        attacker = None

    if args.dataset.lower() in ['mnist',]:
        test_loader = load_mnist(batch_size = args.batch_size, dset = 'test')
        test_num_per_class = 1000
    else:
        raise ValueError('Unrecognized dataset %s'%args.dataset)

    test_batch_num = test_num_per_class * 10 // args.batch_size

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()

    model.eval()
    for idx in range(test_batch_num):

        data_batch, label_batch = test_loader.next()
        if not device_ids in ['cpu',]:
            data_batch = Variable(data_batch).cuda(device)
            label_batch = Variable(label_batch.cuda(device, async = True))
        else:
            data_batch = Variable(data_batch)
            label_batch = Variable(label_batch)

        if attacker is not None:
            data_batch = attacker.attack(model, optimizer, data_batch, label_batch)

        logits = model(data_batch)
        loss = criterion(logits, label_batch)
        acc = accuracy(logits.data, label_batch)

        acc_calculator.update(acc.item(), args.batch_size)
        loss_calculator.update(loss.item(), args.batch_size)
    print('accuracy = %.2f%%'%(acc_calculator.average * 100.,))
    print('loss = %.4f'%(loss_calculator.average,))
