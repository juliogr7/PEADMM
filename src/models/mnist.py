import torch
import torch.nn as nn

import numpy as np

class MNIST_Conv_G(nn.Module):

    def __init__(self, nz = 128, ngf = 64, nc = 1):
        """
        ConvNet generator architecture

        Args:
            nz: dimension of input noise vector
            ngf: dimension of filters
            nc: number of channels in the generated image
        """
        super(MNIST_Conv_G, self).__init__()

        self.nz = nz
        self.ngf = ngf

        self.preprocess = nn.Sequential(
            nn.Linear(nz, 4 * 4 * 4 * self.ngf),
            nn.ReLU(True),
            )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.ngf, 2 * self.ngf, 5),
            nn.ReLU(True),
            )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.ngf, self.ngf, 5),
            nn.ReLU(True),
            )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf, nc, 8, stride = 2)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, noise):
        """
        Forward function of MNIST_Conv_G, output range (-1, 1)
        """

        noise = noise.view(-1, self.nz)
        output = self.preprocess(noise).view(-1, 4 * self.ngf, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.block3(output)
        output = self.sigmoid(output)

        output = output * 2. - 1.

        return output

class MNIST_FC_G(nn.Module):
    def __init__(self, nz=20, ngf=200, nc=1):
        """
        Fully connected neural network for MNIST Generator

        Args:
            nz: int, the number of input neurons
            ngf: int, the number of hidden neurons
            nc: int, the output channels
        """
        super(MNIST_FC_G, self).__init__()

        self.nz = 20
        self.ngf = 200
        self.nc = 1

        self.main = nn.Sequential(
            nn.Linear(self.nz, self.ngf),
            nn.ReLU(True),
            nn.Linear(self.ngf, self.nc * 28 * 28)
            )

        self.sigmoid = nn.Tanh()

    def forward(self, noise):
        """
        Foward function of MNIST_FC_G, output range (-1, 1)
        """
        noise = noise.view(-1, self.nz)
        output = self.main(noise)
        output = self.sigmoid(output)

        return output

class MNIST_FC_D(nn.Module):

    def __init__(self, nc = 1, ngf = 128):
        """
        Fully connected neural network for MNIST Discriminator

        Args:
            nc: int, the output channels
            ngf: int, the number of hidden neurons
        """
        super(MNIST_FC_D, self).__init__()

        self.nc = 1
        self.ngf = 128

        self.main = nn.Sequential(
            nn.Linear(self.nc * 28 * 28, self.ngf),
            nn.ReLU(True),
            nn.Linear(self.ngf, 1)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, noise):
        """
        Forward function of MNIST_FC_D, output range (0, 1)
        """

        noise = noise.view(-1, self.nc * 28 * 28)
        output = self.main(noise)
        output = self.sigmoid(output)

        return output


class MNIST_FC_ELU(nn.Module):
    def __init__(self, nz=20, ngf=200, nc=1):
        """
        Fully connected neural network for MNIST Generator

        Args:
            nz: int, the number of input neurons
            ngf: int, the number of hidden neurons
            nc: int, the output channels
        """
        super().__init__()

        self.nz = 20
        self.ngf = 200
        self.nc = 1

        self.main = nn.Sequential(
            nn.Linear(self.nz, self.ngf),
            nn.ELU(True),
            nn.Linear(self.ngf, self.nc * 28 * 28)
            )

        self.sigmoid = nn.Tanh()

    def forward(self, x):
        """
        Foward function of MNIST_FC_G, output range (-1, 1)
        """
        x = x.view(-1, self.nz)
        x = self.main(x)
        x = self.sigmoid(x)

        return x.view(-1, 28, 28)

