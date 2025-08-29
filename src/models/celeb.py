import pdb
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

class DCGAN_G(nn.Module):
    """
    DCGAN generator architecture
    source: https://github.com/SubarnaTripathi/ReverseGAN

    Args:
        ngpu: number of gpus to use
        nz: dimension of input noise vector
        ngf: dimension of filters
        nc: number of channels in the generated image
    """
    def __init__(self, ngpu, nz, ngf, nc, **kwargs):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution: [64, 100, 1, 1]
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf * 8) x 4 x 4: [64, 512, 4, 4]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf * 4) x 8 x 8: [64, 256, 8, 8]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf * 2) x 16 x 16: [64, 128, 16, 16]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32: [64, 64, 32, 32]
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64: [64, 3, 64, 64]
        )

    def forward(self, x):
        return self.main(x)


class DCGAN_G_ELU(nn.Module):
    """
    DCGAN generator architecture with ELU activations
    source: https://github.com/SubarnaTripathi/ReverseGAN

    Args:
        ngpu: number of gpus to use
        nz: dimension of input noise vector
        ngf: dimension of filters
        nc: number of channels in the generated image
    """
    def __init__(self, ngpu, nz, ngf, nc, **kwargs):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution: [64, 100, 1, 1]
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ELU(inplace=True),
            # state size. (ngf * 8) x 4 x 4: [64, 512, 4, 4]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(inplace=True),
            # state size. (ngf * 4) x 8 x 8: [64, 256, 8, 8]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(inplace=True),
            # state size. (ngf * 2) x 16 x 16: [64, 128, 16, 16]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ELU(inplace=True),
            # state size. (ngf) x 32 x 32: [64, 64, 32, 32]
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64: [64, 3, 64, 64]
        )

    def forward(self, x):
        return self.main(x)


class FC_G(nn.Module):
    """
    Fully connected generator architecture

    Args:
        input_dim (int): dimension of input noise vector
        output_dim (np.array): array with dimensions of output
    """
    def __init__(self, input_dim=100, output_dim=np.array([3, 64, 64]), **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, np.prod(output_dim))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))
        x = x.view(x.shape[0], *self.output_dim)
        return x
