import torch.nn as nn
from .blocks import *


class MNIST_Generator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=784, elu=False):
        super(MNIST_Generator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, 4 * 4 * 4 * self.hidden_dim),
            nn.ReLU(True) if elu == False else nn.ELU(inplace=True),
            )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.hidden_dim, 2 * self.hidden_dim, 5),
            nn.ReLU(True) if elu == False else nn.ELU(inplace=True),
            )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.hidden_dim, self.hidden_dim, 5),
            nn.ReLU(True) if elu == False else nn.ELU(inplace=True),
            )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 1, 8, stride=2)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise):
        output = self.preprocess(noise).view(-1, 4 * self.hidden_dim, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.block3(output)
        output = self.sigmoid(output)

        return output.view(-1, 1, 28, 28)

    def weight_init(self, init_type='normal', init_var=0.02):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
            elif 'normlayer' in name and 'weight' in name:
                if init_type.lower() in ['normal', 'gauss', 'gaussian']:
                    param.data.normal_(1.0, init_var)
                elif init_type.lower() in ['uniform',]:
                    param.data.uniform_(1.0 - init_var, 1.0 + init_var)
                else:
                    raise ValueError('wrong init_type %s' % init_type)
            else:
                if init_type.lower() in ['normal', 'gauss', 'gaussian']:
                    param.data.normal_(0.0, init_var)
                elif init_type.lower() in ['uniform',]:
                    param.data.uniform_(-init_var, init_var)
                else:
                    raise ValueError('wrong init_type %s'%init_type)

class OLD_MNIST_Generator(nn.Module):
    def __init__(self, input_dim=20, ngf=200, nc=1, elu=False):
        """
        Fully connected neural network for MNIST Generator

        Args:
            input_dim: int, the number of input neurons
            ngf: int, the number of hidden neurons
            nc: int, the output channels
        """
        super().__init__()

        self.input_dim = input_dim
        self.ngf = ngf
        self.nc = nc

        self.main = nn.Sequential(
            nn.Linear(self.input_dim, self.ngf),
            nn.ReLU(True) if elu == False else nn.ELU(inplace=True),
            nn.Linear(self.ngf, self.nc * 28 * 28)
            )

        self.sigmoid = nn.Tanh()

    def forward(self, x):
        """
        Foward function of MNIST_FC_G, output range (-1, 1)
        """
        x = x.view(-1, self.input_dim)
        x = self.main(x)
        x = self.sigmoid(x)

        return x.view(-1, 1, 28, 28)


class OLD_Celeba_Generator(nn.Module):
    """
    DCGAN generator architecture with ELU activations
    source: https://github.com/SubarnaTripathi/ReverseGAN

    Args:
        input_dim: dimension of input noise vector
        ngf: dimension of filters
        nc: number of channels in the generated image
    """
    def __init__(self, input_dim=100, ngf=64, nc=3, elu=False):
        super().__init__()
        self.input_dim = input_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution: [64, 100, 1, 1]
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True) if elu == False else nn.ELU(inplace=True),
            # state size. (ngf * 8) x 4 x 4: [64, 512, 4, 4]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True) if elu == False else nn.ELU(inplace=True),
            # state size. (ngf * 4) x 8 x 8: [64, 256, 8, 8]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True) if elu == False else nn.ELU(inplace=True),
            # state size. (ngf * 2) x 16 x 16: [64, 128, 16, 16]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True) if elu == False else nn.ELU(inplace=True),
            # state size. (ngf) x 32 x 32: [64, 64, 32, 32]
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64: [64, 3, 64, 64]
        )

    def forward(self, x):
        x = x.view(1, self.input_dim, 1, 1)
        return self.main(x).view(-1, 3, 64, 64)


class CIFAR10_Generator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=3*32*32, elu=False):
        super(CIFAR10_Generator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.preprocess = nn.Sequential(
            nn.Linear(self.input_dim, 4 * 4 * 4 * self.hidden_dim),
            BatchNorm1dLayer(4 * 4 * 4 * self.hidden_dim),
            nn.ReLU(True) if elu == False else nn.ELU(inplace=True),
            )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.hidden_dim, 2 * self.hidden_dim, 2, stride = 2),
            BatchNorm2dLayer(2 * self.hidden_dim),
            nn.ReLU(True) if elu == False else nn.ELU(inplace=True),
            )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.hidden_dim, self.hidden_dim, 2, stride = 2),
            BatchNorm2dLayer(self.hidden_dim),
            nn.ReLU(True) if elu == False else nn.ELU(inplace = True),
            )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 3, 2, stride = 2)
            )
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.hidden_dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)

    def weight_init(self, init_type='normal', init_var=0.02):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
            elif 'normlayer' in name and 'weight' in name:
                if init_type.lower() in ['normal', 'gauss', 'gaussian']:
                    param.data.normal_(1.0, init_var)
                elif init_type.lower() in ['uniform',]:
                    param.data.uniform_(1.0 - init_var, 1.0 + init_var)
                else:
                    raise ValueError('wrong init_type %s' % init_type)
            else:
                if init_type.lower() in ['normal', 'gauss', 'gaussian']:
                    param.data.normal_(0.0, init_var)
                elif init_type.lower() in ['uniform',]:
                    param.data.uniform_(-init_var, init_var)
                else:
                    raise ValueError('wrong init_type %s' % init_type)


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, bias=True):
        super(ConvLayer, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                stride=1, padding=self.padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, in_channel, out_channel, resample, kernel_size=3, elu=False):
        super(ResidualBlock, self).__init__()

        if resample == 'down':
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size = 2),
                ConvLayer(in_channel = in_channel, out_channel = out_channel, kernel_size = 1),
                )
            self.block = nn.Sequential(
                LayerNormLayer([in_channel, input_dim, input_dim]),
                nn.ReLU() if elu == False else nn.ELU(),
                ConvLayer(in_channel=in_channel, out_channel=in_channel, kernel_size=kernel_size),
                LayerNormLayer([in_channel, input_dim, input_dim]),
                nn.ReLU() if elu == False else nn.ELU(),
                ConvLayer(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size),
                nn.AvgPool2d(kernel_size=2),
                )
        elif resample == 'up':
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvLayer(in_channel=in_channel, out_channel=out_channel, kernel_size=1)
                )
            self.block = nn.Sequential(
                BatchNorm2dLayer(in_channel),
                nn.ReLU() if elu == False else nn.ELU(),
                nn.Upsample(scale_factor=2),
                ConvLayer(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size),
                BatchNorm2dLayer(out_channel),
                nn.ReLU() if elu == False else nn.ELU(),
                ConvLayer(in_channel = out_channel, out_channel = out_channel, kernel_size = kernel_size),
                )
        else:
            raise ValueError('invalid resample value')

    def forward(self, x):
        shortcut = self.shortcut(x)
        forward = self.block(x)
        return shortcut + forward


class LSUN_Generator(nn.Module):
    def __init__(self, input_dim=128, num_channel=32, elu=False):
        super(LSUN_Generator, self).__init__()

        self.num_channel = num_channel
        self.preprocess = nn.Linear(input_dim, 4 * 4 * 8 * num_channel)

        self.resBlock = nn.Sequential(
            ResidualBlock(input_dim=4, in_channel=8*num_channel, out_channel=8*num_channel, resample='up', elu=elu),
            ResidualBlock(input_dim=8, in_channel=8*num_channel, out_channel=4*num_channel, resample='up', elu=elu),
            ResidualBlock(input_dim=16, in_channel=4*num_channel, out_channel=2*num_channel, resample='up', elu=elu),
            ResidualBlock(input_dim=32, in_channel=2*num_channel, out_channel=num_channel, resample='up', elu=elu),
            )

        self.outBlock = nn.Sequential(
            BatchNorm2dLayer(num_channel),
            nn.ReLU() if elu == False else nn.ELU(),
            ConvLayer(in_channel=num_channel, out_channel=3, kernel_size=3),
            nn.Tanh(),
            )

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 8 * self.num_channel, 4, 4)
        output = self.resBlock(output)
        output = self.outBlock(output)

        return output

    def weight_init(self, init_type='normal', init_var=0.02):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
            elif 'normlayer' in name and 'weight' in name:
                if init_type.lower() in ['normal', 'gauss', 'gaussian']:
                    param.data.normal_(1.0, init_var)
                elif init_type.lower() in ['uniform',]:
                    param.data.uniform_(1.0 - init_var, 1.0 + init_var)
                else:
                    raise ValueError('wrong init_type %s' % init_type)
            else:
                if init_type.lower() in ['normal', 'gauss', 'gaussian']:
                    param.data.normal_(0.0, init_var)
                elif init_type.lower() in ['uniform',]:
                    param.data.uniform_(-init_var, init_var)
                else:
                    raise ValueError('wrong init_type %s' % init_type)


class G_LSUN(nn.Module):
    """Generator for the LSUN dataset"""
    def __init__(
            self, input_dim=128, channels=64, output_shape=(3, 64, 64),
            batch_norm=False, activation=None, elu=True):
        super().__init__()
        if activation is None:
            def activation(): return nn.ReLU()

        self.activation_name = type(activation()).__name__
        self.output_shape = output_shape
        layers = [
            nn.ConvTranspose2d(input_dim, 16*channels, 4, 1, 0),
            nn.ConvTranspose2d(16*channels, 8*channels, 4, 2, 1),
            nn.BatchNorm2d(8*channels) if batch_norm else None,
            activation(),
            nn.ConvTranspose2d(8*channels, 4*channels, 4, 2, 1),
            nn.BatchNorm2d(4*channels) if batch_norm else None,
            activation(),
            nn.ConvTranspose2d(4*channels, 2*channels, 4, 2, 1),
            nn.BatchNorm2d(2*channels) if batch_norm else None,
            activation(),
            nn.ConvTranspose2d(2 * channels, 3, 4, 2, 1),
            nn.Tanh(),
            ]

        layers = filter(None, layers)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        n = x.shape[0]
        x = x.view(n, -1, 1, 1)
        x = self.main(x)
        return x.view(n, *self.output_shape)

    def __str__(self):
        return '-'.join([
            'dc-gen-batchnorm', self.activation_name, 'Tanh'])

G_CelebA = G_LSUN

class G_FMNIST(nn.Module):
    """Generator for the MNIST dataset"""
    def __init__(
            self, input_dim=128, hidden_dim=64, output_shape=(1, 28, 28),
            activation=None, final_activation=None, elu=True):
        super().__init__()
        if activation is None:
            def activation(): return nn.ReLU()
        if final_activation is None:
            def final_activation(): return nn.Sigmoid()

        self.activation_name = type(activation()).__name__
        self.f_activation_name = type(final_activation()).__name__
        self.hidden_dim = hidden_dim
        self.output_shape = output_shape
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, 4 * 4 * 4 * self.hidden_dim),
            activation(),
            )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.hidden_dim, 2 * self.hidden_dim, 5),
            activation(),
            )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.hidden_dim, self.hidden_dim, 5),
            activation(),
            nn.ConvTranspose2d(self.hidden_dim, 1, 8, stride=2),
            final_activation(),
            )

    def forward(self, x):
        x = self.preprocess(x).view(-1, 4 * self.hidden_dim, 4, 4)
        x = self.block1(x)
        x = x[:, :, :7, :7]
        x = self.block2(x)
        return x.view(-1, *self.output_shape)

    def __str__(self):
        return '-'.join([
            'dc-gen', self.activation_name, self.f_activation_name])


# For Celeba, we use the same architecture as the one in LSUN
Celeba_Generator = LSUN_Generator

