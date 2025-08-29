import copy
import sys
from src import augmented_lagrangian as auglagr

import torch
import torchmetrics
from torch import nn, autograd

from src.proj_l1 import proj_l1
from src.functions import MeasurementError
import matplotlib.pyplot as plt
import os
save_dir = os.path.dirname(os.path.abspath(__file__))

def l2_norm(x):
    return torch.norm(x.reshape(x.shape[0], -1), dim=1)


def inf_norm(x):
    return torch.amax(torch.abs(x), dim=(1, 2, 3))


class GenADMMAlgorithm(nn.Module):
    def __init__(self, generator, mode, z_dim, gamma=3e-5, beta=1.0, sigma=1.0, max_iter=100,
                 encoder=None, exact=False, optical_encoder = None):
        super(GenADMMAlgorithm, self).__init__()

        self.mode = mode
        self.encoder = encoder
        self.use_exact = exact

        self.optical_encoder = optical_encoder
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
        beta = self.beta * torch.ones(y.shape[0], device=y.device)
        gamma = self.gamma * torch.ones(y.shape[0], device=y.device)
        sigma = self.sigma * torch.ones(y.shape[0], device=y.device)

        if self.encoder is not None and self.mode == 'prop':
            if self.optical_encoder is not None:
                x0 = self.optical_encoder.vec2image(self.optical_encoder.get_inverse(y))
                z = torch.nn.Parameter(self.encoder(x0).detach().clone(), requires_grad=True)
            else:
                z = torch.nn.Parameter(self.encoder(y).detach().clone(), requires_grad=True)
        else:
            z = torch.nn.Parameter(torch.randn((y.size(0), self.z_dim), device=y.device),
                                   requires_grad=True)  # Solution matrix for all samples

        z_best = z.detach().clone()

        x = self.generator(z)

        if self.mode == 'prop':
            # Guardar GT, measurement (como imagen) y reconstrucción solo en 300 y 6000
            # if i in [0]:
            idx = 15
            x_best = x
            x_best_img = x_best[idx].detach().cpu().squeeze().numpy()
            ref_img = ref_x[idx].detach().cpu().squeeze().numpy()
            measurement_img = y[idx].detach().cpu().squeeze().numpy()
            psnr_val = self.psnr_loss(x_best[idx:idx+1], ref_x[idx:idx+1], data_range=1.0).item()
            ssim_val = self.ssim_loss(x_best[idx:idx+1], ref_x[idx:idx+1], data_range=1.0).item()
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(ref_img, cmap='gray')
            axes[0].set_title('GT')
            axes[0].axis('off')
            axes[1].imshow(measurement_img, cmap='gray')
            axes[1].set_title('Measurement')
            axes[1].axis('off')
            axes[2].imshow(x_best_img, cmap='gray')
            axes[2].set_title(f'Reconstr.\nPSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.2f}')
            axes[2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'gani_recon_idx{idx}_gt_meas_recon.svg'))
            plt.close()

        lambda_ = torch.zeros_like(x, requires_grad=False)

        if self.use_exact:
            fn = lambda x: l2_norm(y - self.optical_encoder.get_measurement(self.optical_encoder.image2vec(x)))
            fn_best = l2_norm(
                self.optical_encoder.get_measurement(self.optical_encoder.image2vec(self.generator(z))) - y)
        else:
            fn = lambda x: inf_norm(y - x) / x[0].numel()
            fn_best = inf_norm(self.generator(z) - y)

        aug_lagr = auglagr.AugmentedLagrangian(fn, self.generator)
        next_bs = aug_lagr.adaptive_bs(beta_0=beta, sigma_0=sigma)

        admm_metrics = copy.deepcopy(self.admm_metrics)

        # v = torch.zeros_like(x, requires_grad=False)


        for i in range(self.max_iter):
            if i % 10 == 0:
                print(f"Iteración {i}")
            al, infs = aug_lagr(x, z, beta, lambda_)  # line 2
            gradz = autograd.grad(al.sum(), z, retain_graph=False)[0]

            beta_aux = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            if self.use_exact or self.optical_encoder is not None:
                x = self.optical_encoder.close_solution(x, y, beta.unsqueeze(-1).unsqueeze(-1), lambda_)
            else:
                with torch.no_grad():
                    x = - lambda_ / beta_aux + self.generator(z)
                h = x - y
                x = h - proj_l1(beta_aux * h) / beta_aux + y

            z = z - (gamma / beta).unsqueeze(-1) * gradz

            al, infs = aug_lagr(x, z, beta, lambda_)  # line 3
            lambda_ = (lambda_ + sigma.unsqueeze(-1).unsqueeze(-1)
                       .unsqueeze(-1) * aug_lagr.lambda_grad(x, z, beta, lambda_))  # line 5

            with torch.no_grad():
                fn_value = fn(self.generator(z))
                perf_mask = fn_value < fn_best

                if perf_mask.sum() > 0:
                    z_best[perf_mask] = z[perf_mask]
                    fn_best[perf_mask] = fn_value[perf_mask]

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

            # Guardar GT, measurement (como imagen) y reconstrucción solo en 300 y 6000
            if i in [300, 6000]:
                # Generar reconstricciones actuales
                x_best = self.generator(z).detach()
                num_to_save = min(15, x_best.size(0))
                fig, axes = plt.subplots(num_to_save, 3, figsize=(9, 2.0 * num_to_save))
                if num_to_save == 1:
                    axes = axes.reshape(1, 3)
                for idx in range(num_to_save):
                    x_best_img = x_best[idx].detach().cpu().squeeze().numpy()
                    ref_img = ref_x[idx].detach().cpu().squeeze().numpy()
                    measurement_img = y[idx].detach().cpu().squeeze().numpy()
                    psnr_val = self.psnr_loss(x_best[idx:idx+1], ref_x[idx:idx+1], data_range=1.0).item()
                    ssim_val = self.ssim_loss(x_best[idx:idx+1], ref_x[idx:idx+1], data_range=1.0).item()
                    axes[idx, 0].imshow(ref_img, cmap='gray')
                    axes[idx, 0].set_title('GT')
                    axes[idx, 0].axis('off')
                    axes[idx, 1].imshow(measurement_img, cmap='gray')
                    axes[idx, 1].set_title('Measurement')
                    axes[idx, 1].axis('off')
                    axes[idx, 2].imshow(x_best_img, cmap='gray')
                    axes[idx, 2].set_title(f'Reconstr.\nPSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.2f}')
                    axes[idx, 2].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{self.mode}_recon_first15_iter_{i}_gt_meas_recon.svg'))
                plt.close()

            # Guardar histograma solo en 5000 y 25000
            if i in [5000, 25000]:
                plt.figure()
                z_cpu = z.detach().cpu().numpy().flatten()
                z_mean = z_cpu.mean()
                z_std = z_cpu.std()
                plt.hist(z_cpu, bins=50, color='blue', alpha=0.7)
                plt.title(f'Histograma de z en iteración {i}\nMedia: {z_mean:.4f}  Std: {z_std:.4f}')
                plt.xlabel('Valor de z')
                plt.ylabel('Frecuencia')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{self.mode}_hist_z_iter_{i}.svg'))
                plt.close()

        
        import sys
        sys.exit()
        return self.generator(z_best), z, admm_metrics
