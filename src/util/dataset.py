import os
import random
import sklearn
import sklearn.datasets
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from PIL import Image

class SPIDataset(Dataset):
    def __init__(self, transformed_images, original_images, labels, H=None):
        self.transformed_images = transformed_images
        self.original_images = original_images
        self.labels = labels
        self.H = H

    def __len__(self):
        return self.transformed_images.shape[0]

    def __getitem__(self, idx):
        return self.transformed_images[idx], self.original_images[idx], self.labels[idx]

    def get_mask(self):
        return self.H

class ImageFolder(torch.utils.data.Dataset):

    def __init__(self, root, data_transform = None, target_transform = None):
        '''
        >>> Process images under a folder
        '''
        self.root = root
        self.image_list = []
        self.data_transform = data_transform
        self.target_transform = target_transform

        for folder, _, fnames in os.walk(root):
            for fname in fnames:
                if fname.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:          # fname is an image
                    path = os.path.join(folder, fname)
                    self.image_list.append([path, 0])

        if len(self.image_list) == 0:
            raise ValueError('No images found under the folder %s'%root)
        print('There are %d images found under the folder %s'%(len(self.image_list), root))

    def __getitem__(self, index):
        '''
        >>> obtain the image and target by []
        '''
        path, target = self.image_list[index]
        image = Image.open(path).convert('RGB')
        if self.data_transform is not None:
            image = self.data_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self,):
        '''
        >>> length
        '''
        return len(self.image_list)

def gaussian_25(batch_size, datapoints = 100000):
    '''
    >>> Samples based on 25 Gassians
    '''
    dataset = []
    for i in xrange(datapoints / 25):
        for x in xrange(-2, 3):
            for y in xrange(-2, 3):
                point = np.random.randn(2) * 0.05
                point[0] += 2 * x
                point[1] += 2 * y
                dataset.append(point)

    dataset = np.array(dataset, dtype = 'float32')
    np.random.shuffle(dataset)
    dataset /= 2 * np.sqrt(2)

    while True:
        for i in xrange(datapoints / batch_size):
            yield dataset[i * batch_size: (i + 1) * batch_size]

def gaussian_8(batch_size, scale = 2.):
    '''
    >>> Samples based on 8 Gassians
    '''
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2)),]
    centers = [(x * scale, y * scale) for x, y in centers]
    while True:
        data_batch = []
        for i in xrange(batch_size):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            data_batch.append(point)
        data_batch = np.array(data_batch, dtype = 'float32')
        data_batch /= np.sqrt(2)
        yield data_batch

def gaussian_8_multivar(batch_size, scale = 2.):
    '''
    >>> Samples based on 8 Gaussians with different variance
    '''
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2)),]
    centers = [(x * scale, y * scale) for x, y in centers]
    variances = [0.02, 0.05, 0.1, 0.2, 0.02, 0.05, 0.1, 0.2]
    while True:
        data_batch = []
        for i in xrange(batch_size):
            center_idx = np.random.randint(8)
            center = centers[center_idx]
            variance = variances[center_idx]
            point = np.random.randn(2) * variance
            point[0] += center[0]
            point[1] += center[1]
            data_batch.append(point)
        data_batch = np.array(data_batch, dtype = 'float32')
        data_batch /= np.sqrt(2)
        yield data_batch

def swiss_roll(batch_size):
    '''
    >>> Swiss roll
    '''

    while True:
        data = sklearn.datasets.make_swiss_roll(
            n_samples = batch_size,
            noise = 1. / 4.
            )[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 7.5
        yield data

def mnist_loader(batch_size, batch_size_test = None, horizontal_flip = False, random_clip = False, normalization = None):

    batch_size_test = batch_size if batch_size_test == None else batch_size_test

    if normalization == None:
        normalization = transforms.Normalize(mean = [0., 0., 0.], std = [1., 1., 1.])

    basic_transform = [transforms.ToTensor(), normalization]
    data_augumentation = []
    if horizontal_flip == True:
        data_augumentation.append(transforms.RandomHorizontalFlip())
    if random_clip == True:
        data_augumentation.append(transforms.RandomCrop(28, 4))

    train_set = datasets.MNIST(root = './data', train = True, download = True,
        transform = transforms.Compose(data_augumentation + basic_transform))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    test_set = datasets.MNIST(root = './data', train = False, download = True,
        transform = transforms.Compose(basic_transform))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size_test, shuffle = False, num_workers = 4, pin_memory = True)

    return train_loader, test_loader

def mnist(batch_size):
    '''
    >>> generate images from MNIST
    '''

    train_loader, test_loader = mnist_loader(batch_size)

    while True:
        for idx, (data_batch, label_batch) in enumerate(train_loader, 0):
            if data_batch.shape[0] != batch_size:
                continue
            yield data_batch.view(data_batch.shape[0], -1)

def cifar10_loader(batch_size, batch_size_test = None, horizontal_flip = False, random_clip = False, normalization = None):

    batch_size_test = batch_size if batch_size_test == None else batch_size_test

    if normalization == None:
        normalization = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])

    basic_transform = [transforms.ToTensor(), normalization]
    data_augumentation = []
    if horizontal_flip == True:
        data_augumentation.append(transforms.RandomHorizontalFlip())
    if random_clip == True:
        data_augumentation.append(transforms.RandomCrop(32, 4))

    train_set = datasets.CIFAR10(root = './data', train = True, download = True,
        transform = transforms.Compose(data_augumentation + basic_transform))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    test_set = datasets.CIFAR10(root = './data', train = False, download = True,
        transform = transforms.Compose(basic_transform))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size_test, shuffle = False, num_workers = 4, pin_memory = True)

    return train_loader, test_loader

def cifar10(batch_size):
    '''
    >>> generate image from CIFAR10
    '''
    train_loader, test_loader = cifar10_loader(batch_size)

    while True:
        for idx, (data_batch, label_batch) in enumerate(train_loader, 0):
            if data_batch.shape[0] != batch_size:
                continue
            yield data_batch.view(data_batch.shape[0], -1)

def lsun_loader(batch_size, batch_size_test = None, scale = 64, horizontal_flip = False, normalization = None, data_folder = './data'):

    batch_size_test = batch_size if batch_size_test == None else batch_size_test

    if normalization == None:
        normalization = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    basic_transform = [transforms.ToTensor(), normalization]
    data_augumentation = []
    if horizontal_flip == True:
        data_augumentation.append(transforms.RandomHorizontalFlip())
    if scale != None:
        data_augumentation.append(transforms.Scale(scale))
        data_augumentation.append(transforms.CenterCrop(scale))

    train_set = datasets.LSUN(root = data_folder, classes = ['bedroom_train',],
        transform = transforms.Compose(data_augumentation + basic_transform))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    test_set = datasets.LSUN(root = data_folder, classes = ['bedroom_val',],
        transform = transforms.Compose(data_augumentation + basic_transform))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size_test, shuffle = True, num_workers = 4, pin_memory = True)

    return train_loader, test_loader

def lsun(batch_size, scale = 64, data_folder = './data'):
    '''
    >>> generate image from LSUN
    '''
    train_loader, test_loader = lsun_loader(batch_size, scale = scale, data_folder = data_folder)

    while True:
        for idx, (data_batch, label_batch) in enumerate(train_loader, 0):
            if data_batch.shape[0] != batch_size:
                continue
            yield data_batch.view(data_batch.shape[0], -1)

def celeba_loader(batch_size, scale = 64, data_folder = './data'):
    '''
    >>> generate image from Celeba
    '''

    dataset = ImageFolder(root = data_folder, data_transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Scale(scale),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

    return data_loader

def celeba(batch_size, scale = 64, data_folder = './data'):

    data_loader = celeba_loader(batch_size, scale = scale, data_folder = data_folder)

    while True:
        for idx, (data_batch, label_batch) in enumerate(data_loader, 0):
            if data_batch.shape[0] != batch_size:
                continue
            yield data_batch.view(data_batch.shape[0], -1)


