import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
from .dataset import SPIDataset
import numpy as np

def transform_images_loader(data_loader, physics, args, H = None): # By us, not in submit_code originally
    import torch.utils.data as data
    # import matplotlib.pyplot as plt

    original_images_train = []
    transformed_images = []
    labels = []
    
    for batch in data_loader:
        images, batch_labels = batch
        if images.min() < 0:
            images = images * 0.5 + 0.5

        if H is not None:
            transformed = images @ H.T
        else:
            transformed = physics(images)
        original_images_train.append(images)
        transformed_images.append(transformed.detach())
        # plt.imshow(transformed[0].detach().squeeze(0), cmap='gray') # Debug
        # plt.title('Single pixel')
        # plt.show()
        labels.append(batch_labels)

    transformed_images_tensor = torch.cat(transformed_images, dim=0)
    original_images_train_tensor = torch.cat(original_images_train, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    if H is None:
        H = physics.mask.squeeze(0).squeeze(0)

    dataset = SPIDataset(transformed_images_tensor,
                                 original_images_train_tensor,
                                 labels_tensor,
                                 H=H)

    transformed_data_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # transformed_dataset = data.TensorDataset(transformed_images_tensor,
    #                                       original_images_train_tensor, 
    #                                       labels_tensor)
    # transformed_data_loader = data.DataLoader(transformed_dataset, args.batch_size, shuffle=False,
    #                         num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    return transformed_data_loader

def attack_images_loader(data_loader, args): # By us, not original
    import torch.utils.data as data
    from src.models.loader import load_classifier
    from torch import optim

    std = args.std

    ckpt_mnist = 'pretrained_classifier/mnist_0_1/300-300-300-0.0.ckpt'
    classifier = load_classifier(model2load=ckpt_mnist, dataset='mnist')
    pgm = PGM(step_size=0.01, threshold=std, iter_num=30)
    optimizer = optim.SGD(classifier.parameters(), lr=1)

    attacked_images = []
    labels = []
    original_images_train = []
    
    for batch in data_loader:
        images, batch_labels = batch
        if images.min() < 0:
            images = images * 0.5 + 0.5
        attacked = pgm.attack(
            classifier,
            optimizer=optimizer,
            data_batch=images,
            label_batch=batch_labels
        )
        original_images_train.append(images)
        attacked_images.append(attacked.detach())
        labels.append(batch_labels)

    attacked_images_tensor = torch.cat(attacked_images, dim=0)
    original_images_train_tensor = torch.cat(original_images_train, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    attacked_dataset = data.TensorDataset(attacked_images_tensor,
                                          original_images_train_tensor, 
                                          labels_tensor)
    attacked_data_loader = data.DataLoader(attacked_dataset, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    return attacked_data_loader

def projection(ori_pt, max_norm, order = np.inf):
    '''
    Projection of the original point into the max norm constraint

    >>> ori_pt: original point
    >>> max_norm: maximum norms allowed
    >>> order: order of the norm used
    '''

    if order in [np.inf,]:
        prj_pt = torch.clamp(ori_pt, min = -max_norm, max = max_norm)
    elif order in [2,]:
        ori_shape = ori_pt.size()
        flat_pt = ori_pt.view(ori_shape[0], -1)
        pt_norm = torch.norm(flat_pt, dim = 1, p = 2)
        pt_norm_clip = torch.clamp(pt_norm, min = -max_norm, max = max_norm)
        resize_ratio = (pt_norm + 1e-8) / (pt_norm_clip + 1e-8)
        flat_pt = flat_pt / resize_ratio.view(-1, 1)
        prj_pt = flat_pt.view(ori_shape)
    else:
        raise ValueError('Invalid order of norms: %s'%str(order))

    return prj_pt

def query_grad(model, criterion, optimizer, data_batch, label_batch, weight = None):
    '''
    >>> we calculate the gradient of one or more models
    '''

    if isinstance(model, (tuple, list)):
        weight = np.ones([len(model),]) if weight == None else weight
        accumulated_grad = 0
        for idx, (m, opt) in enumerate(zip(model, optimizer)):
            logits = m(data_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            accumulated_grad += (data_batch.grad * weight[idx]).detach()

            opt.zero_grad()
            data_batch.grad.zero_()
        return (accumulated_grad / np.sum(weight)).detach()
    else:
        logits = model(data_batch)
        loss = criterion(logits, label_batch)
        loss.backward()

        return data_batch.grad.detach()

class Attacker(object):

    def __init__(self,):

        pass

    def attack(self, model, optimizer, data_batch, label_batch, criterion = nn.CrossEntropyLoss()):

        raise NotImplementedError('An abstract function should not be called')

class PGM(Attacker):

    def __init__(self, step_size, threshold, iter_num, order = np.inf, image_range = [0., 1.]):
        '''
        >>> threshold: maximum perturbation
        >>> iter_num: the maximum number of iterations
        >>> order: norm of the attack
        '''
        self.step_size = torch.FloatTensor(np.array(step_size))
        self.threshold = torch.FloatTensor(np.array(threshold))
        self.iter_num = iter_num
        self.order = order
        self.image_min = image_range[0]
        self.image_max = image_range[1]

    def attack(self, model, optimizer, data_batch, label_batch, criterion = nn.CrossEntropyLoss()):
        '''
        >>> model: model to be fooled
        >>> optimizer: the optimizer binded to the model
        >>> data_batch, label_batch: clean data batch and corresponding label
        >>> step_size: step size for a single iteration
        '''

        data_batch = data_batch.detach()                # detach data_batch from the original graph, making it a leaf node
        label_batch = label_batch.detach()              # detach label_batch from the original graph, making it a leaf node
        data_batch.requires_grad_()

        # Reserve the original point
        data_batch_0 = data_batch.detach()

        # Move to the same device
        step_size = self.step_size.to(data_batch.device)
        threshold = self.threshold.to(data_batch.device)

        for iter_idx in range(self.iter_num):

            data_grad = query_grad(model, criterion, optimizer, data_batch, label_batch)

            if self.order == np.inf:
                next_point = data_batch + step_size * torch.sign(data_grad)
            elif self.order == 2:
                ori_shape = data_batch.size()
                flat_grad = data_grad.view(ori_shape[0], -1)
                norm_grad = torch.norm(flat_grad, dim = 1, p = 2)
                perturbation = step_size * (flat_grad + 1e-8) / (torch.norm(flat_grad, dim = 1, p = 2).view(-1, 1) + 1e-8)
                next_point = data_batch + perturbation.view(ori_shape)
            else:
                raise ValueError('Invalid order of norms: %s'%str(self.order))

            next_point = data_batch_0 + projection(next_point - data_batch_0, max_norm = threshold, order = self.order)
            next_point = torch.clamp(next_point, min = self.image_min, max = self.image_max)
            data_batch = next_point.detach().requires_grad_()

        return data_batch

