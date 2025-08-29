import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.autograd import Variable

def imshow(image, dimshuffle = None):
    '''
    >>> image of shape [height, width, channels]
    '''
    if dimshuffle != None:
        image = np.transpose(image, dimshuffle)

    image = np.clip(image, a_min = 0.0, a_max = 1.0)
    channel_num = image.shape[-1]
    if channel_num == 3:
        plt.imshow(image)
    elif channel_num == 1:
        stacked_image = np.concatenate([image, image, image], axis = 2)
        plt.imshow(stacked_image)
    else:
        raise ValueError('image format is wrong, channel_num = %d'%channel_num)

def plot_synthetic_data(batch_size, batch_num, data, netG, input_dim, use_gpu, output_file):

    true_sample_list = np.zeros([batch_size * batch_num, 2], dtype = np.float32)
    fake_sample_list = np.zeros([batch_size * batch_num, 2], dtype = np.float32)

    for batch_idx in range(batch_num):
        true_sample_list[batch_size * batch_idx : batch_size * (batch_idx + 1)] = data.next()
        noise = torch.randn(batch_size, input_dim)
        if use_gpu:
            noise = noise.cuda()
        noise = Variable(noise, volatile = True)
        fake_batch = netG(noise)
        fake_sample_list[batch_size * batch_idx : batch_size * (batch_idx + 1)] = np.array(fake_batch.detach())

    plt.scatter(x = true_sample_list[:, 0], y = true_sample_list[:, 1], c = 'b', marker = '+', s = 15, alpha = 0.3)
    plt.scatter(x = fake_sample_list[:, 0], y = fake_sample_list[:, 1], c = 'r', marker = '+', s = 15, alpha = 0.3)
    plt.savefig(output_file, ddi = 500, bbox_inches = 'tight')
    plt.clf()


def plot_mnist_data(true_rows, fake_rows, cols, data, netG, input_dim, use_gpu, output_file):

    rows = true_rows + fake_rows
    cols = cols

    plt.figure(figsize = (rows, cols))
    gs1 = gridspec.GridSpec(rows, cols)
    gs1.update(wspace = 0., hspace = 0.)

    for row_idx in range(rows):

        for col_idx in range(cols):

            if row_idx < true_rows:
                image = np.array(data.next()[0])
            else:
                noise = torch.randn(1, input_dim)
                if use_gpu:
                    noise = noise.cuda()
                noise = Variable(noise, volatile = True)
                image = netG(noise).detach()
                image = np.array(image[0])
            image = image.reshape(28, 28, 1)

            subplot_idx = row_idx * cols + col_idx 
            plt.subplot(gs1[subplot_idx])
            imshow(image)
            plt.xticks([])
            plt.yticks([])

    # plt.suptitle('First %d Rows are Real Data, Last %d Rows are Fake Data'%(true_rows, fake_rows))
    plt.savefig(output_file, ddi = 500, bbox_inches = 'tight')
    plt.clf()

def plot_cifar10_data(true_rows, fake_rows, cols, data, netG, input_dim, use_gpu, output_file):

    rows = true_rows + fake_rows
    cols = cols

    plt.figure(figsize = (rows, cols))
    gs1 = gridspec.GridSpec(rows, cols)
    gs1.update(wspace = 0., hspace = 0.)

    for row_idx in range(rows):

        for col_idx in range(cols):

            if row_idx < true_rows:
                image = np.array(data.next()[0])
            else:
                noise = torch.randn(1, input_dim)
                if use_gpu:
                    noise = noise.cuda()
                noise = Variable(noise, volatile = True)
                image = netG(noise).detach()
                image = np.array(image[0])
            image = image.reshape(3, 32, 32)
            image = image.transpose(1, 2, 0)
            image = (image + 1.) / 2.

            subplot_idx = row_idx * cols + col_idx
            plt.subplot(gs1[subplot_idx])
            imshow(image)
            plt.xticks([])
            plt.yticks([])

    plt.suptitle('First %d Rows are Real Data, Last %d Rows are Fake Data'%(true_rows, fake_rows))
    plt.savefig(output_file, ddi = 500, bbox_inches = 'tight')
    plt.clf()

def plot_lsun_data(true_rows, fake_rows, cols, data, scale, netG, input_dim, use_gpu, output_file):

    rows = true_rows + fake_rows
    cols = cols

    plt.figure(figsize = (rows, cols))
    gs1 = gridspec.GridSpec(rows, cols)
    gs1.update(wspace = 0., hspace = 0.)

    for row_idx in range(rows):

        for col_idx in range(cols):

            if row_idx < true_rows:
                image = np.array(data.next()[0])
            else:
                noise = torch.randn(1, input_dim)
                if use_gpu:
                    noise = noise.cuda()
                noise = Variable(noise, volatile = True)
                image = netG(noise).detach()
                image = np.array(image[0])
            image = image.reshape(3, scale, scale)
            image = image.transpose(1, 2, 0)
            image = (image + 1.) / 2.

            subplot_idx = row_idx * cols + col_idx
            plt.subplot(gs1[subplot_idx])
            imshow(image)
            plt.xticks([])
            plt.yticks([])

    # plt.suptitle('First %d Rows are Real Data, Last %d Rows are Fake Data'%(true_rows, fake_rows))
    plt.savefig(output_file, ddi = 500, bbox_inches = 'tight')
    plt.clf()

# Plot Celeba data in the same way as LSUN data
plot_celeba_data = plot_lsun_data
