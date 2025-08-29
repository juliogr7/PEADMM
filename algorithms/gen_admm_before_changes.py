import copy

from src import augmented_lagrangian as auglagr

import torch
import torchmetrics
from torch import nn, autograd

from src.proj_l1 import proj_l1
from src.functions import MeasurementError
import torch.nn.functional as F

def inf_norm(x):
    return torch.max(torch.abs(x))

import torch.nn.functional as F

class ModifyOutputGenerator(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, z):
        x = self.generator(z)
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False).to(z.device)
        return x

class GenADMMAlgorithm(nn.Module):
    def __init__(self, generator, mode, z_dim, gamma=3e-5, beta=1.0, sigma=1.0, max_iter=100, encoder=None, H = None):
        super(GenADMMAlgorithm, self).__init__()
        
        self.mode = mode
        self.encoder = encoder
        self.H = H
        if self.mode == 'exact':
            self.generator = ModifyOutputGenerator(generator)
        else:
            self.generator = generator

        for param in self.generator.parameters():
            param.requires_grad = False

        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ADMM parameters

        self.z_dim = z_dim
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma
        self.max_iter = max_iter

        # metrics

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.psnr_loss = torchmetrics.functional.image.peak_signal_noise_ratio
        self.ssim_loss = torchmetrics.functional.structural_similarity_index_measure

        self.admm_metrics = {f'Iter': [],
                             f'x/admm/mse': [],
                             f'x/admm/mae': [],
                             f'x/admm/psnr': [],
                             f'x/admm/ssim': [],
                             f'z/admm/mean': [],
                             f'z/admm/var': [],
                             f'z/admm/std': []}

    def forward(self, y, ref_x):

        # Initialize variables
        beta = self.beta
        gamma = self.gamma
        sigma = self.sigma
        device = y.device

        # if self.mode == 'exact':
            # H = torch.tensor(self.H).to(device)
            # H = H.repeat(y.shape[0], 1, 1, 1)

        if self.encoder is not None and self.mode == 'prop':
            z = torch.nn.Parameter(self.encoder(y).detach().clone(), requires_grad=True)
        else:
            z = torch.nn.Parameter(torch.randn((y.size(0), self.z_dim), device=y.device), requires_grad=True)  # Solution matrix for all samples

        z_best = z.detach().clone()

        x = self.generator(z)

        fn_best = inf_norm(self.generator(z) - y)

        lambda_ = torch.zeros_like(x, requires_grad=False)
        # v = torch.zeros_like(x, requires_grad=False).to(device)
        
        if self.mode == 'exact':
            fn = MeasurementError(A = self.H, b = y) # A @ x - b
        else:
            fn = lambda x: inf_norm(y - x) / x.numel()

        aug_lagr = auglagr.AugmentedLagrangian(fn, self.generator)
        next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)

        admm_metrics = copy.deepcopy(self.admm_metrics)
        for i in range(self.max_iter):
            al, infs = aug_lagr(x, z, beta, lambda_)  # line 2
            gradz = autograd.grad(al, z)[0]

            
            if self.mode == 'exact':
                x = aug_lagr.x_exact(x, z, beta, lambda_).to(device)
                # I = torch.eye(x.shape[3]).to(device)
                # torch.matmul((torch.linalg.inv(beta * I) + torch.matmul(torch.transpose(H,2,3), H)), (torch.matmul(torch.transpose(H,2,3), y) + beta * (v - lambda_)))
            else:
                with torch.no_grad():
                    x = - lambda_ / beta + self.generator(z)
                h = x - y
                x = h - proj_l1(beta * h) / beta + y

            z = z - gamma / beta * gradz
            
            # v = self.generator(z)

            al, infs = aug_lagr(x, z, beta, lambda_)  # line 3
            lambda_ = lambda_ + sigma * aug_lagr.lambda_grad(x, z, beta, lambda_)  # line 5

            fn_value = fn(self.generator(z))

            if fn_value < fn_best:
                z_best = z.clone()
                fn_best = fn_value.item()

            beta, sigma = next_bs(i, infs)  # line 4

            # Compute metrics

            x_hat = self.generator(z)
            x_mse = self.mse_loss(x_hat, ref_x)
            x_mae = self.mae_loss(x_hat, ref_x)
            x_psnr = self.psnr_loss(x_hat, ref_x, data_range=1.0)
            x_ssim = self.ssim_loss(x_hat, ref_x, data_range=1.0)

            z_mean = z.mean()
            z_var = z.var()
            z_std = z.std()

            admm_metrics['Iter'].append(i)
            admm_metrics['x/admm/mse'].append(x_mse.item())
            admm_metrics['x/admm/mae'].append(x_mae.item())
            admm_metrics['x/admm/psnr'].append(x_psnr.item())
            admm_metrics['x/admm/ssim'].append(x_ssim.item())
            admm_metrics['z/admm/mean'].append(z_mean.item())
            admm_metrics['z/admm/var'].append(z_var.item())
            admm_metrics['z/admm/std'].append(z_std.item())

            if i % 100 == 0:
                print(f'Iter {i}, x_mse: {x_mse.item()}, x_mae: {x_mae.item()}, '
                      f'x_psnr: {x_psnr.item()}, x_ssim: {x_ssim.item()}, '
                      f'z_mean: {z_mean.item()}, z_var: {z_var.item()}, z_std: {z_std.item()}')

        return self.generator(z_best), z, admm_metrics
