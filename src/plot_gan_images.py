import os
import sys
sys.path.insert(0, './')
import torch
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util.device_parser import parse_device_alloc, config_visible_gpu
from util.plot_image import plot_synthetic_data, plot_mnist_data, plot_cifar10_data, plot_lsun_data
from models.generators import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 100,
        help = 'batch size, default = 100')
    parser.add_argument('--batch_num', type = int, default = 100,
        help = 'batch num, default = 100')
    parser.add_argument('--rows', type = int, default = 6,
        help = 'number of rows, default = 6')
    parser.add_argument('--cols', type = int, default = 6,
        help = 'number of columns, default = 6')
    parser.add_argument('--input_dim', type = int, default = 128,
        help = 'dimension of the input, default = 128')
    parser.add_argument('--gpu', type = str, default = None,
        help = 'GPU, default = None')
    parser.add_argument('--output_file', type = str, default = None,
        help = 'output files, default = None')
    parser.add_argument('--scale', type = int, default = 64,
        help = 'the scale of the images, default = 64')

    parser.add_argument('--use_elu', type = int, default = 0,
        help = 'whether or not to use ELU, default = 0, meaning no')

    parser.add_argument('--dataset', type = str, default = 'lsun',
        help = 'the dataset, default = "lsun"')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'model to be loaded')
    parser.add_argument('--model_width', type = int, default = 32,
        help = 'the width of the model, default = 32')
    parser.add_argument('--data_folder', type = str, default = './data',
        help = 'the data folder of LSUN, default = "./data"')

    args = parser.parse_args()
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu'

    if args.output_file == None:
        raise ValueError('need to specify the output file')
    if args.model2load == None:
        raise ValueError('model2load cannot be None')

    args.use_elu = True if args.use_elu != 0 else False

    if args.dataset.lower() in ['mnist',]:
        model = MNIST_Generator(elu=args.use_elu)
        model.load_state_dict(torch.load(args.model2load, map_location='cpu'))
        _, model = parse_device_alloc(device_config = None, model = model)
        model.eval()
        plot_mnist_data(true_rows = 0, fake_rows = args.rows, cols = args.cols, data = None,
            netG = model, input_dim = args.input_dim, use_gpu = use_gpu, output_file = args.output_file)
    elif args.dataset.lower() in ['cifar10',]:
        model = CIFAR10_Generator(elu=args.use_elu)
        model.load_state_dict(torch.load(args.model2load, map_location='cpu'))
        _, model = parse_device_alloc(device_config = None, model = model)
        model.eval()
        plot_cifar10_data(true_rows = 0, fake_rows = args.rows, cols = args.cols, data = None,
            netG = model, input_dim = args.input_dim, use_gpu = use_gpu, output_file = args.output_file)
    elif args.dataset.lower() in ['lsun', 'celeba']:
        model = LSUN_Generator(num_channel = args.model_width, elu = args.use_elu)
        model.load_state_dict(torch.load(args.model2load, map_location='cpu'))
        _, model = parse_device_alloc(device_config = None, model = model)
        model.eval()
        plot_lsun_data(true_rows = 0, fake_rows = args.rows, cols = args.cols, data = None, scale = args.scale,
            netG = model, input_dim = args.input_dim, use_gpu = use_gpu, output_file = args.output_file)
    else:
        raise ValueError('Unrecognized dataset: %s'%args.dataset)

