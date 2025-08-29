import torch
import torch.nn as nn

import numpy as np

class MNIST_Classifier(nn.Module):
    '''
    >>> General class for multilayer perceptron
    >>> Suitable for MNIST
    '''

    def __init__(self, input_dim = 784, hidden_dims = [300, 300, 300], output_class = 10, nonlinearity = 'relu'):
        '''
        >>> input_dim, hidden_dims, output_class: the dim of input neurons, hidden neurons and output neurons
        >>> dropout: the dropout rate i.e. the probability to deactivate the neuron, None means no dropout
        '''

        super(MNIST_Classifier, self).__init__()

        self.neurons = [input_dim,] + hidden_dims + [output_class,]
        self.nonlinearity = nonlinearity
        
        self.layers = []
        self.main_block = nn.Sequential()

        self.str2func = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'sigd': nn.Sigmoid()}

        for idx, (in_dim, out_dim) in enumerate(zip(self.neurons[:-2], self.neurons[1:-1])):
            self.main_block.add_module('layer_%d'%idx, nn.Linear(in_dim, out_dim))
            self.main_block.add_module('nonlinear_%d'%idx, self.str2func[self.nonlinearity])

        self.output = nn.Linear(self.neurons[-2], output_class)

    def forward(self, x):
        '''
        >>> x: 2-d tensor of shape [batch_size, in_dim]
        '''

        out = x.view(x.size(0), -1)
        out = self.main_block(out)
        out = self.output(out)

        return out


class Classifier_FMNIST(nn.Module):
    def __init__(self, activation=None, elu=True):
        if activation is None:
            def activation(): return nn.ReLU()

        super().__init__()
        self.activation_name = type(activation()).__name__
        self.convolution = nn.Sequential(*[
            nn.Conv2d(1, 32, 5, 1, 2),
            activation(),
            nn.Conv2d(32, 32, 5, 2, 2),
            activation(),
            nn.Conv2d(32, 64, 3, 1, 1),
            activation(),
            nn.Conv2d(64, 64, 3, 2, 1),
            activation(),
        ])
        self.fc = _fully_connected(
                [7 * 7 * 64, 1024, 10], activation=activation,
                final_activation=None)

    def forward(self, x):
        x = self.convolution(x)
        return self.fc(x.view(x.shape[0], -1))

    def __str__(self):
        return '-'.join([
            'convnet-fc-10class', self.activation_name])


class Classifier_CelebA(nn.Module):
    def __init__(self, activation=None):
        if activation is None:
            def activation(): return nn.ReLU()

        super().__init__()
        self.activation_name = type(activation()).__name__
        self.convolution = nn.Sequential(*[
            nn.Conv2d(3, 64, 5, 2, 2),
            activation(),
            nn.Conv2d(64, 64, 5, 2, 2),
            activation(),
            nn.Conv2d(64, 64, 3, 2, 1),
            activation()
        ])
        self.fc = _fully_connected(
                [8 * 8 * 64, 384, 192, 2], activation=activation,
                final_activation=None)

    def forward(self, x):
        x = self.convolution(x)
        return self.fc(x.view(x.shape[0], -1))

    def __str__(self):
        return '-'.join([
            'convnet-fc-2class', self.activation_name])


def _fully_connected(layer_sizes, activation, final_activation):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(*layer_sizes[i:i+2]))
        if i < len(layer_sizes) - 2:
            layers.append(activation())
    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)
