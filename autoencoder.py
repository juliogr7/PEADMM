import copy
import time

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
import wandb
from einops import rearrange
from matplotlib import pyplot as plt
from torchmetrics.functional.image import image_gradients
from torchvision.utils import make_grid

from algorithms.gen_admm import GenADMMAlgorithm
from src.models import loader
from src import params


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# GAN Inversion Adversarial / SPI MNIST = Convolutional Encoder + GAN Decoder
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

class ConvolutionalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, elu, single_pixel=False):

        super(ConvolutionalEncoder, self).__init__()
        self.elu = elu
        self.single_pixel = single_pixel

        if self.single_pixel:
            self.resize_28 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0)  # For Single-Pixel

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=8, stride=2),
            nn.ELU(True) if self.elu else nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=5),
            nn.ELU(True) if self.elu else nn.ReLU(True)
        )

        # The GAN crops 1d (from 8x8 to 7x7) so I used padding to do the inverse process (7x7 -> 8x8)
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),  # Add a dimension (right and bottom) to the image, so now its 8x8
            nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, kernel_size=5),
            nn.ELU(True) if self.elu else nn.ReLU(True)
        )

        self.linear = nn.Linear(4 * 4 * 4 * hidden_dim, input_dim)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        if self.single_pixel:
            # x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
            x = self.resize_28(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.elu:
            out = (out - out.min()) / (out.max() - out.min())  # Normalize for ELU
        return out


class Decoder_GAN(nn.Module):
    def __init__(self, elu, cuda=True, single_pixel=False):
        super().__init__()

        self.single_pixel = single_pixel

        dataset = 'mnist'
        ckpt_name = dataset + '_elu' if elu else dataset + '_relu'

        generator = loader.load_generator(
            params.ckpts[ckpt_name], dataset,
            input_dim=params.input_dims[dataset], elu=elu)

        self.generator = generator

    def forward(self, x):
        output = self.generator(x).to(x.device)

        if self.single_pixel:
            output = F.interpolate(output, size=(32, 32), mode='bilinear', align_corners=False).to(output.device)

        return output


def tv_loss(x_hat, x):
    dy, dx = image_gradients(x)
    dy_hat, dx_hat = image_gradients(x_hat)

    x_tv = torch.abs(dy) + torch.abs(dx)
    x_hat_tv = torch.abs(dy_hat) + torch.abs(dx_hat)

    return F.mse_loss(x_tv, x_hat_tv)


from skimage.metrics import structural_similarity as ssim
import numpy as np

# Asume que img1 y img2 están en [0, 1] y son de shape [H, W]
def compute_skimage_ssim(pred, gt):
    pred_np = pred.squeeze().cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()
    return ssim(gt_np, pred_np, data_range=1.0)


class Autoencoder_GAN_MNIST(L.LightningModule):
    def __init__(self, lr=1e-3, patient=15 * 5, elu_gan=True, elu_encoder=True, adversarial=False, single_pixel=False):
        super().__init__()
        self.encoder = ConvolutionalEncoder(input_dim=128, hidden_dim=64, elu=elu_encoder, single_pixel=single_pixel)
        self.decoder = Decoder_GAN(elu=elu_gan, single_pixel=single_pixel).eval()
        self.init_lr = lr
        self.patient = patient
        self.adversarial = adversarial
        self.single_pixel = single_pixel

        for param in self.decoder.parameters():  # Frozen GAN
            param.requires_grad = False

        # criterion and metrics

        self.train_total_time = 0.0
        self.val_total_time = 0.0

        self.mse = nn.MSELoss()

        self.mae = torchmetrics.functional.mean_absolute_error
        self.psnr = torchmetrics.functional.image.peak_signal_noise_ratio
        self.ssim = compute_skimage_ssim

    def forward(self, x, return_latent=False):

        self.decoder.eval()  # Freeze GAN

        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        if return_latent:
            return x_hat, latent
        else:
            return x_hat

    def general_step(self, mode, batch, batch_idx):

        if self.adversarial or self.single_pixel:
            x, x_original, _ = batch
            if x.min() < 0:
                x = x * 0.5 + 0.5
            if x_original.min() < 0:
                x_original = x_original * 0.5 + 0.5
        else:
            x, _ = batch
            if x.min() < 0:
                x = x * 0.5 + 0.5
            x_original = x.clone()

        self.decoder.eval()  # Freeze GAN

        latent = self.encoder(x)
        x_hat = self.decoder(latent)

        loss = self.mse(x_hat, x_original)

        mae_loss = self.mae(x_hat, x_original)
        psnr_loss = self.psnr(x_hat, x_original, data_range=1.0)
        ssim_loss = self.ssim(x_hat, x_original)

        latent_mean = latent.mean()
        latent_std = latent.std()

        self.log_dict({f'recon/{mode}_loss': loss,
                       f'recon/{mode}_mae': mae_loss,
                       f'recon/{mode}_psnr': psnr_loss,
                       f'recon/{mode}_ssim': ssim_loss,
                       f'latent/{mode}/latent_mean': latent_mean,
                       f'latent/{mode}/latent_std': latent_std}, on_step=False, on_epoch=True)

        if mode in ['val', 'test'] and batch_idx == 0:
            # Guardar solo las imágenes con idx 11 y 15 (si existen en el batch)
            indices = [11, 15]
            imgs, recons, psnrs, ssims = [], [], [], []
            for idx in indices:
                if idx < x.shape[0]:
                    img = x[idx].cpu().numpy().squeeze()
                    recon = x_hat[idx].cpu().numpy().squeeze()
                    psnr_val = self.psnr(torch.tensor(recon).unsqueeze(0), torch.tensor(img).unsqueeze(0), data_range=1.0)
                    ssim_val = self.ssim(torch.tensor(recon), torch.tensor(img))
                    imgs.append(img)
                    recons.append(recon)
                    psnrs.append(psnr_val)
                    ssims.append(ssim_val)

            if imgs:
                import os
                os.makedirs('inference_results', exist_ok=True)
                fig, axes = plt.subplots(2, len(imgs), figsize=(len(imgs)*3, 6))
                for i in range(len(imgs)):
                    axes[0, i].imshow(imgs[i])
                    axes[0, i].axis('off')
                    axes[0, i].set_title(f'Original (idx {indices[i]})')
                    axes[1, i].imshow(recons[i])
                    axes[1, i].axis('off')
                    axes[1, i].set_title(f'PSNR: {psnrs[i]:.2f}\nSSIM: {ssims[i]:.4f}')

                fig.text(0.04, 0.75, 'Original', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
                fig.text(0.04, 0.26, 'Reconstruida', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
                plt.tight_layout(rect=[0.06, 0, 1, 1])
                save_path = os.path.join('inference_results', f'{mode}_indices_11_15_std025_.svg')
                plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
                print(f'Figura guardada en: {save_path}')
                wandb.log({f'figure/{mode}/indices_11_15': wandb.Image(plt)})
                plt.close()

        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.general_step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.general_step('test', batch, batch_idx)

    def on_train_epoch_start(self):
        self.train_epoch_start_time = time.time()

    def on_validation_epoch_start(self):
        self.val_epoch_start_time = time.time()

    def on_general_epoch_end(self, mode):
        if mode in ['train', 'val']:
            epoch_time = time.time() - (self.train_epoch_start_time if mode == 'train' else self.val_epoch_start_time)
            total_time_attr = f"{mode}_total_time"
            setattr(self, total_time_attr, getattr(self, total_time_attr) + epoch_time)

            self.log_dict({"Epoch": self.current_epoch,
                           f"Time/{mode}/epoch": epoch_time,
                           f"Time/{mode}/total": getattr(self, total_time_attr)})

    def on_train_epoch_end(self):
        self.on_general_epoch_end(mode='train')

    def on_validation_epoch_end(self):
        self.on_general_epoch_end(mode='val')

    def on_test_epoch_end(self):
        self.on_general_epoch_end(mode='test')

    def load_checkpoint(self, checkpoint_path, trainable=False):
        checkpoint = torch.load(checkpoint_path)['state_dict']

        encoder_checkpoint = {}
        decoder_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith('encoder'):
                encoder_checkpoint[key.replace('encoder.', '')] = value
            elif key.startswith('decoder'):
                decoder_checkpoint[key.replace('decoder.', '')] = value

        self.encoder.load_state_dict(encoder_checkpoint)
        self.decoder.load_state_dict(decoder_checkpoint)

        self.encoder.requires_grad_(trainable)
        self.decoder.requires_grad_(trainable)

    def get_model(self):
        return nn.Sequential(self.encoder, self.decoder)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.init_lr)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=self.patient,
                                                               verbose=True, mode='max')
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=reduce_lr, monitor='recon/val_psnr'))


class Autoencoder_GAN_MNIST_SPI(L.LightningModule):
    def __init__(self, optical_encoder, lr=1e-3, patient=15 * 5, elu_gan=True, elu_encoder=True):
        super().__init__()
        self.optical_encoder = optical_encoder
        self.encoder = ConvolutionalEncoder(input_dim=128, hidden_dim=64, elu=elu_encoder, single_pixel=True)
        self.decoder = Decoder_GAN(elu=elu_gan, single_pixel=True).eval()
        self.init_lr = lr
        self.patient = patient

        for param in self.optical_encoder.parameters():  # Frozen optical encoder
            param.requires_grad = False

        for param in self.decoder.parameters():  # Frozen GAN
            param.requires_grad = False

        # criterion and metrics

        self.train_total_time = 0.0
        self.val_total_time = 0.0

        self.mse = nn.MSELoss()

        self.mae = torchmetrics.functional.mean_absolute_error
        self.psnr = torchmetrics.functional.image.peak_signal_noise_ratio
        self.ssim = torchmetrics.functional.structural_similarity_index_measure

    def forward(self, x0, return_latent=False):
        self.decoder.eval()  # Freeze GAN

        latent = self.encoder(x0)
        x_hat = self.decoder(latent)

        if return_latent:
            return x_hat, latent
        else:
            return x_hat

    def general_step(self, mode, batch, batch_idx):
        self.decoder.eval()  # Freeze GAN
        x, _ = batch

        y = self.optical_encoder.get_measurement(self.optical_encoder.image2vec(x))
        x0 = self.optical_encoder.vec2image(self.optical_encoder.get_inverse(y))

        latent = self.encoder(x0)
        x_hat = self.decoder(latent)

        loss = self.mse(x_hat, x)
        mae_loss = self.mae(x_hat, x)
        psnr_loss = self.psnr(x_hat, x, data_range=1.0)
        ssim_loss = self.ssim(x_hat, x, data_range=1.0)

        latent_mean = latent.mean()
        latent_std = latent.std()

        self.log_dict({f'recon/{mode}_loss': loss,
                       f'recon/{mode}_mae': mae_loss,
                       f'recon/{mode}_psnr': psnr_loss,
                       f'recon/{mode}_ssim': ssim_loss,
                       f'latent/{mode}/latent_mean': latent_mean,
                       f'latent/{mode}/latent_std': latent_std}, on_step=False, on_epoch=True)

        if mode in ['val', 'test'] and batch_idx == 0:
            x_grid = make_grid(x[:16], nrow=4, normalize=True)
            x_hat_grid = make_grid(x_hat[:16], nrow=4, normalize=True)
            x_gt_grid = make_grid(x[:16], nrow=4, normalize=True)

            plt.figure(figsize=(14, 5))
            plt.suptitle(f'Epoch: {self.current_epoch}. MSE Loss: {loss:.4f} - MAE Loss: {mae_loss:.4f} - '
                         f'PSNR: {psnr_loss:.2f} - SSIM: {ssim_loss:.4f}')

            plt.subplot(1, 3, 1)
            plt.imshow(x_grid[0].cpu())
            plt.axis('off')
            plt.title('Adversarial attacked')

            plt.subplot(1, 3, 2)
            plt.imshow(x_hat_grid[0].cpu().to(torch.float32).numpy())
            plt.axis('off')
            plt.title('Reconstructed')

            plt.subplot(1, 3, 3)
            plt.imshow(x_gt_grid[0].cpu())
            plt.axis('off')
            plt.title('Ground Truth')

            plt.tight_layout()

            # save figure in wandb

            wandb.log({f'figure/{mode}/recon': wandb.Image(plt)})
            plt.close()

        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.general_step('val', batch, batch_idx)

    def test_step(self, batch):
        self.decoder.eval()  # Freeze GAN
        x, _ = batch

        y = self.optical_encoder.get_measurement(self.optical_encoder.image2vec(x))
        x0 = self.optical_encoder.vec2image(self.optical_encoder.get_inverse(y))
        x_hat = self(x0)  # Forward

        psnr_list = []
        ssim_list = []

        for i in range(x.shape[0]):
            psnr_i = self.psnr(x_hat[i].unsqueeze(0), x[i].unsqueeze(0), data_range=1.0)
            ssim_i = self.ssim(x_hat[i].unsqueeze(0), x[i].unsqueeze(0), data_range=1.0)
            psnr_list.append(psnr_i.cpu())
            ssim_list.append(ssim_i.cpu())

        return {
            "original": x.cpu(),
            "transformed": y.cpu(),
            "reconstructed": x_hat.cpu(),
            "psnr": torch.stack(psnr_list),
            "ssim": torch.stack(ssim_list),
        }

        # return self.general_step('test', batch, batch_idx)

    def on_train_epoch_start(self):
        self.train_epoch_start_time = time.time()

    def on_validation_epoch_start(self):
        self.val_epoch_start_time = time.time()

    def on_general_epoch_end(self, mode):
        if mode in ['train', 'val']:
            epoch_time = time.time() - (self.train_epoch_start_time if mode == 'train' else self.val_epoch_start_time)
            total_time_attr = f"{mode}_total_time"
            setattr(self, total_time_attr, getattr(self, total_time_attr) + epoch_time)

            self.log_dict({"Epoch": self.current_epoch,
                           f"Time/{mode}/epoch": epoch_time,
                           f"Time/{mode}/total": getattr(self, total_time_attr)})

    def on_train_epoch_end(self):
        self.on_general_epoch_end(mode='train')

    def on_validation_epoch_end(self):
        self.on_general_epoch_end(mode='val')

    def on_test_epoch_end(self):
        self.on_general_epoch_end(mode='test')

    def load_checkpoint(self, checkpoint_path, trainable=False):
        checkpoint = torch.load(checkpoint_path)['state_dict']

        encoder_checkpoint = {}
        decoder_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith('encoder'):
                encoder_checkpoint[key.replace('encoder.', '')] = value
            elif key.startswith('decoder'):
                decoder_checkpoint[key.replace('decoder.', '')] = value

        self.encoder.load_state_dict(encoder_checkpoint)
        self.decoder.load_state_dict(decoder_checkpoint)

        self.encoder.requires_grad_(trainable)
        self.decoder.requires_grad_(trainable)

    def get_model(self):
        return nn.Sequential(self.encoder, self.decoder)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.init_lr)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=self.patient,
                                                               verbose=True, mode='max')
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=reduce_lr, monitor='recon/val_psnr'))


class GenADMM(L.LightningModule):
    def __init__(self, generator, mode, z_dim, gamma=3e-5, beta=1.0, sigma=1.0, max_iter=100, encoder=None,
                 exact=False):
        super().__init__()
        self.admm = GenADMMAlgorithm(generator, mode, z_dim, gamma=gamma, beta=beta, sigma=sigma, max_iter=max_iter,
                                     encoder=encoder, exact=exact)
        self.generator = generator
        self.exact = exact

        for param in self.generator.parameters():
            param.requires_grad = False

        for param in self.admm.parameters():
            param.requires_grad = False

        # metrics

        self.total_time = 0.0

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

        self.dummy_model = torch.nn.Linear(1, 1)

    def compute_metrics(self):
        # stack each value horizontally

        admm_metrics = copy.deepcopy(self.admm_metrics)
        for key, value in self.admm_metrics.items():
            admm_metrics[key] = np.mean(value, axis=0)

        # iterate over each iteration
        for i in range(len(admm_metrics['Iter'])):
            wandb.log({f'Iter': admm_metrics['Iter'][i],
                       f'x/admm/mse': admm_metrics['x/admm/mse'][i],
                       f'x/admm/mae': admm_metrics['x/admm/mae'][i],
                       f'x/admm/psnr': admm_metrics['x/admm/psnr'][i],
                       f'x/admm/ssim': admm_metrics['x/admm/ssim'][i],
                       f'z/admm/mean': admm_metrics['z/admm/mean'][i],
                       f'z/admm/var': admm_metrics['z/admm/var'][i],
                       f'z/admm/std': admm_metrics['z/admm/std'][i]},
                      )

    def training_step(self, batch, batch_idx):

        total_batches = self.trainer.num_training_batches
        print(f"Batch: {batch_idx + 1} / {total_batches}")

        y, x, _ = batch

        if x.min() < 0:
            print("x normalized to [0, 1]")
            x = x * 0.5 + 0.5

        # if y.min() < 0:
        #     print("YESSS AAA")
        #     y = y * 0.5 + 0.5

        x_hat, z_hat, admm_metrics = self.admm(y, ref_x=x)
        # x_hat = self.optical_encoder.vec2image(x_hat)

        # metrics per iteration

        for key, value in admm_metrics.items():
            self.admm_metrics[key].append(value)

        # metrics

        x_mse = self.mse_loss(x_hat, x)
        x_mae = self.mae_loss(x_hat, x)
        x_psnr = self.psnr_loss(x_hat, x, data_range=1.0)
        x_ssim = self.ssim_loss(x_hat, x, data_range=1.0)

        z_mean = z_hat.mean()
        z_var = z_hat.var()
        z_std = z_hat.std()

        self.log_dict({f'x/test/mse': x_mse,
                       f'x/test/mae': x_mae,
                       f'x/test/psnr': x_psnr,
                       f'x/test/ssim': x_ssim,
                       f'z/test/mean': z_mean,
                       f'z/test/var': z_var,
                       f'z/test/std': z_std}, on_step=True, on_epoch=True)

        # plot figure

        if batch_idx == 0:
            x_grid = make_grid(x[:16], nrow=4, normalize=True)
            x_hat_grid = make_grid(x_hat[:16], nrow=4, normalize=True)

            wandb.log({"figure/x": wandb.Image(x_grid)})
            wandb.log({"figure/x_hat": wandb.Image(x_hat_grid)})

            plt.figure(figsize=(10, 10))

            num_samples = 16 if len(x) > 16 else len(x)
            for i in range(num_samples):
                plt.subplot(4, 4, i + 1)
                plt.imshow(x_hat[i].permute(1, 2, 0).cpu().detach().numpy())
                plt.title(f'psnr: {self.psnr_loss(x_hat[i][None, ...], x[i][None, ...], data_range=1.0):.2f}, '
                          f'ssim: {self.ssim_loss(x_hat[i][None, ...], x[i][None, ...], data_range=1.0):.2f}')
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"results/{self.loggers[1].name.split('/csv')[0]}/x_hat.svg")
            plt.close()

            if self.current_epoch == 0:
                plt.figure(figsize=(10, 10))

                for i in range(num_samples):
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(x[i].permute(1, 2, 0).cpu().detach().numpy())
                    plt.title('gt')
                    plt.axis('off')

                plt.tight_layout()
                plt.savefig(f"results/{self.loggers[1].name.split('/csv')[0]}/x.svg")
                plt.close()

            return self.dummy_model(torch.ones(1, 1).to(x.device))

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.total_time += epoch_time
        self.log_dict({"Epoch": self.current_epoch, "Time": epoch_time, "Total Time": self.total_time})

    def configure_optimizers(self):
        return torch.optim.Adam(self.generator.parameters(), lr=1e-3)


class GenSPCADMM(L.LightningModule):
    def __init__(self, optical_encoder, generator, mode, z_dim, gamma=3e-5, beta=1.0, sigma=1.0, max_iter=100,
                 encoder=None):
        super().__init__()
        self.optical_encoder = optical_encoder
        self.admm = GenADMMAlgorithm(generator, mode, z_dim, gamma=gamma, beta=beta, sigma=sigma,
                                     max_iter=max_iter, encoder=encoder, exact=True, optical_encoder=optical_encoder)
        self.generator = generator

        for param in self.generator.parameters():
            param.requires_grad = False

        for param in self.admm.parameters():
            param.requires_grad = False

        # metrics

        self.total_time = 0.0

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

        self.dummy_model = torch.nn.Linear(1, 1)

    def compute_metrics(self):
        # stack each value horizontally

        admm_metrics = copy.deepcopy(self.admm_metrics)
        for key, value in self.admm_metrics.items():
            admm_metrics[key] = np.mean(value, axis=0)

        # iterate over each iteration
        for i in range(len(admm_metrics['Iter'])):
            wandb.log({f'Iter': admm_metrics['Iter'][i],
                       f'x/admm/mse': admm_metrics['x/admm/mse'][i],
                       f'x/admm/mae': admm_metrics['x/admm/mae'][i],
                       f'x/admm/psnr': admm_metrics['x/admm/psnr'][i],
                       f'x/admm/ssim': admm_metrics['x/admm/ssim'][i],
                       f'z/admm/mean': admm_metrics['z/admm/mean'][i],
                       f'z/admm/var': admm_metrics['z/admm/var'][i],
                       f'z/admm/std': admm_metrics['z/admm/std'][i]},
                      )

    def training_step(self, batch, batch_idx):
        
        total_batches = self.trainer.num_training_batches
        print(f"Batch: {batch_idx + 1} / {total_batches}")
        
        x, _ = batch
        x_flat = rearrange(x, 'b c m n -> b (c n m)')
        y = self.optical_encoder.get_measurement(x_flat)

        x_hat, z_hat, admm_metrics = self.admm(y, ref_x=x)

        # metrics per iteration

        for key, value in admm_metrics.items():
            self.admm_metrics[key].append(value)

        # metrics

        x_mse = self.mse_loss(x_hat, x)
        x_mae = self.mae_loss(x_hat, x)
        x_psnr = self.psnr_loss(x_hat, x, data_range=1.0)
        x_ssim = self.ssim_loss(x_hat, x, data_range=1.0)

        z_mean = z_hat.mean()
        z_var = z_hat.var()
        z_std = z_hat.std()

        self.log_dict({f'x/test/mse': x_mse,
                       f'x/test/mae': x_mae,
                       f'x/test/psnr': x_psnr,
                       f'x/test/ssim': x_ssim,
                       f'z/test/mean': z_mean,
                       f'z/test/var': z_var,
                       f'z/test/std': z_std}, on_step=True, on_epoch=True)

        # plot figure

        if batch_idx == 0:
            x_grid = make_grid(x[:16], nrow=4, normalize=True)
            x_hat_grid = make_grid(x_hat[:16], nrow=4, normalize=True)

            wandb.log({"figure/x": wandb.Image(x_grid)})
            wandb.log({"figure/x_hat": wandb.Image(x_hat_grid)})

            plt.figure(figsize=(10, 10))

            num_samples = 16 if len(x) > 16 else len(x)
            for i in range(num_samples):
                plt.subplot(4, 4, i + 1)
                plt.imshow(x_hat[i].permute(1, 2, 0).cpu().detach().numpy())
                plt.title(f'psnr: {self.psnr_loss(x_hat[i][None, ...], x[i][None, ...], data_range=1.0):.2f}, '
                          f'ssim: {self.ssim_loss(x_hat[i][None, ...], x[i][None, ...], data_range=1.0):.2f}')
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"results/{self.loggers[1].name.split('/csv')[0]}/x_hat.svg")
            plt.close()

            if self.current_epoch == 0:
                plt.figure(figsize=(10, 10))

                for i in range(num_samples):
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(x[i].permute(1, 2, 0).cpu().detach().numpy())
                    plt.title('gt')
                    plt.axis('off')

                plt.tight_layout()
                plt.savefig(f"results/{self.loggers[1].name.split('/csv')[0]}/x.svg")
                plt.close()

        return self.dummy_model(torch.ones(1, 1).to(x.device))

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.total_time += epoch_time
        self.log_dict({"Epoch": self.current_epoch, "Time": epoch_time, "Total Time": self.total_time})

    def configure_optimizers(self):
        return torch.optim.Adam(self.generator.parameters(), lr=1e-3)


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# Convolutional Autoencoder
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

class Encoder(nn.Module):
    def __init__(self, num_channels, num_bands):
        super().__init__()

        self.conv1 = nn.Conv2d(num_bands, int(8 * num_channels), 3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(int(8 * num_channels), int(8 * num_channels), 3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(int(8 * num_channels), int(4 * num_channels), 3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(int(4 * num_channels), int(4 * num_channels), 3, padding=1)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(int(4 * num_channels), int(2 * num_channels), 3, padding=1)
        self.act5 = nn.ReLU()
        self.conv6 = nn.Conv2d(int(2 * num_channels), num_channels, 3, padding=1)
        self.act6 = nn.Tanh()

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def apply_l2_regularization(self, reg_param=None):
        loss = 0
        if reg_param is not None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    loss += reg_param * torch.norm(module.weight, p=2)
                    loss += reg_param * torch.norm(module.bias, p=2)

        return loss

    def forward(self, x):
        x1 = self.act1(self.conv1(x))
        x2 = self.act2(self.conv2(x1))
        x3 = self.act3(self.conv3(x2))
        x4 = self.act4(self.conv4(x3))
        x5 = self.act5(self.conv5(x4))
        x6 = self.act6(self.conv6(x5))

        return x6


class Decoder(nn.Module):
    def __init__(self, num_channels, num_bands):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, int(2 * num_channels), 3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(int(2 * num_channels), int(4 * num_channels), 3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(int(4 * num_channels), int(4 * num_channels), 3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(int(4 * num_channels), int(8 * num_channels), 3, padding=1)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(int(8 * num_channels), int(8 * num_channels), 3, padding=1)
        self.act5 = nn.ReLU()
        self.conv6 = nn.Conv2d(int(8 * num_channels), num_bands, 3, padding=1)
        self.act6 = nn.ReLU()

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def apply_l2_regularization(self, reg_param=None):
        loss = 0
        if reg_param is not None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    loss += reg_param * torch.norm(module.weight, p=2)
                    loss += reg_param * torch.norm(module.bias, p=2)

        return loss

    def forward(self, x):
        x1 = self.act1(self.conv1(x))
        x2 = self.act2(self.conv2(x1))
        x3 = self.act3(self.conv3(x2))
        x4 = self.act4(self.conv4(x3))
        x5 = self.act5(self.conv5(x4))
        x6 = self.act6(self.conv6(x5))

        return x6


def tv_loss(x_hat, x):
    dy, dx = image_gradients(x)
    dy_hat, dx_hat = image_gradients(x_hat)

    x_tv = torch.abs(dy) + torch.abs(dx)
    x_hat_tv = torch.abs(dy_hat) + torch.abs(dx_hat)

    return F.mse_loss(x_tv, x_hat_tv)


class Autoencoder(L.LightningModule):
    def __init__(self, num_channels, num_bands, lr=1e-3, patient=15 * 5):
        super().__init__()
        self.encoder = Encoder(num_channels, num_bands)
        self.decoder = Decoder(num_channels, num_bands)
        self.init_lr = lr
        self.patient = patient

        # criterion and metrics

        self.mse = nn.MSELoss()
        self.tv_reg = torchmetrics.functional.total_variation

        self.mae = torchmetrics.functional.mean_absolute_error
        self.psnr = torchmetrics.functional.image.peak_signal_noise_ratio
        self.ssim = torchmetrics.functional.structural_similarity_index_measure
        self.sam = torchmetrics.functional.image.spectral_angle_mapper

    def forward(self, x, return_latent=False):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        if return_latent:
            return x_hat, latent
        else:
            return x_hat

    def test_step(self, batch):

        x, _ = batch
        if x.min() < 0:
            x = x * 0.5 + 0.5

        x_hat = self(x)  # Forward

        psnr_list = []
        ssim_list = []

        for i in range(x.shape[0]):
            psnr_i = self.psnr(x_hat[i].unsqueeze(0), x[i].unsqueeze(0), data_range=1.0)
            ssim_i = self.ssim(x_hat[i].unsqueeze(0), x[i].unsqueeze(0), data_range=1.0)
            psnr_list.append(psnr_i.cpu())
            ssim_list.append(ssim_i.cpu())

        return {
            "original": x.cpu(),
            "reconstructed": x_hat.cpu(),
            "psnr": torch.stack(psnr_list),
            "ssim": torch.stack(ssim_list),
        }

    def general_step(self, mode, batch):
        x, _ = batch
        if x.min() < 0:
            x = x * 0.5 + 0.5

        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        loss = self.mse(x_hat, x)

        encoder_l2_reg_loss = self.encoder.apply_l2_regularization(reg_param=1e-8)
        decoder_l2_reg_loss = self.decoder.apply_l2_regularization(reg_param=1e-8)
        tv_reg_loss = tv_loss(x_hat, x)

        loss += encoder_l2_reg_loss + decoder_l2_reg_loss + 0.1 * tv_reg_loss

        mae_loss = self.mae(x_hat, x)
        psnr_loss = self.psnr(x_hat, x, data_range=1.0)
        ssim_loss = self.ssim(x_hat, x, data_range=1.0)

        self.log_dict({f'{mode}_loss': loss,
                       f'{mode}_tv': tv_reg_loss,
                       f'{mode}_mae': mae_loss,
                       f'{mode}_psnr': psnr_loss,
                       f'{mode}_ssim': ssim_loss}, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch):
        return self.general_step('train', batch)

    def validation_step(self, batch):
        return self.general_step('val', batch)

    def load_checkpoint(self, checkpoint_path, trainable=False):
        checkpoint = torch.load(checkpoint_path)['state_dict']

        encoder_checkpoint = {}
        decoder_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith('encoder'):
                encoder_checkpoint[key.replace('encoder.', '')] = value
            elif key.startswith('decoder'):
                decoder_checkpoint[key.replace('decoder.', '')] = value

        self.encoder.load_state_dict(encoder_checkpoint)
        self.decoder.load_state_dict(decoder_checkpoint)

        self.encoder.requires_grad_(trainable)
        self.decoder.requires_grad_(trainable)

    def get_model(self):
        return nn.Sequential(self.encoder, self.decoder)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=self.patient,
                                                               verbose=True, mode='max')
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=reduce_lr, monitor='train_psnr'))


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# Linear Autoencoder
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


class LinearEncoder(nn.Module):
    def __init__(self, num_channels, input_size):
        super().__init__()

        # Define fully connected layers instead of convolutions
        self.fc1 = nn.Linear(input_size, 8 * num_channels)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(8 * num_channels, 8 * num_channels)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(8 * num_channels, 4 * num_channels)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(4 * num_channels, 4 * num_channels)
        self.act4 = nn.ReLU()
        self.fc5 = nn.Linear(4 * num_channels, 2 * num_channels)
        self.act5 = nn.ReLU()
        self.fc6 = nn.Linear(2 * num_channels, num_channels)
        self.act6 = nn.Tanh()

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def apply_l2_regularization(self, reg_param=None):
        loss = 0
        if reg_param is not None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    loss += reg_param * torch.norm(module.weight, p=2)
                    loss += reg_param * torch.norm(module.bias, p=2)

        return loss

    def forward(self, x):
        # Forward pass through the fully connected layers
        x1 = self.act1(self.fc1(x))
        x2 = self.act2(self.fc2(x1))
        x3 = self.act3(self.fc3(x2))
        x4 = self.act4(self.fc4(x3))
        x5 = self.act5(self.fc5(x4))
        x6 = self.act6(self.fc6(x5))

        return x6


class LinearDecoder(nn.Module):
    def __init__(self, num_channels, num_bands):
        super().__init__()

        # Define fully connected layers instead of convolutions
        self.fc1 = nn.Linear(num_channels, 8 * num_channels)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(8 * num_channels, 8 * num_channels)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(8 * num_channels, 4 * num_channels)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(4 * num_channels, 4 * num_channels)
        self.act4 = nn.ReLU()
        self.fc5 = nn.Linear(4 * num_channels, 2 * num_channels)
        self.act5 = nn.ReLU()
        self.fc6 = nn.Linear(2 * num_channels, num_bands)
        self.act6 = nn.ReLU()

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def apply_l2_regularization(self, reg_param=None):
        loss = 0
        if reg_param is not None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    loss += reg_param * torch.norm(module.weight, p=2)
                    loss += reg_param * torch.norm(module.bias, p=2)

        return loss

    def forward(self, x):
        # Forward pass through the fully connected layers
        x1 = self.act1(self.fc1(x))
        x2 = self.act2(self.fc2(x1))
        x3 = self.act3(self.fc3(x2))
        x4 = self.act4(self.fc4(x3))
        x5 = self.act5(self.fc5(x4))
        x6 = self.act6(self.fc6(x5))

        return x6


class LinearAutoencoder(L.LightningModule):
    def __init__(self, image_size, num_channels, lr=1e-3, patient=15 * 5):
        super().__init__()
        self.encoder = LinearEncoder(num_channels, image_size ** 2)
        self.decoder = LinearDecoder(num_channels, image_size ** 2)
        self.init_lr = lr
        self.patient = patient

        # criterion and metrics

        self.mse = nn.MSELoss()
        self.tv_reg = torchmetrics.functional.total_variation

        self.mae = torchmetrics.functional.mean_absolute_error
        self.psnr = torchmetrics.functional.image.peak_signal_noise_ratio
        self.ssim = torchmetrics.functional.structural_similarity_index_measure
        self.sam = torchmetrics.functional.image.spectral_angle_mapper

    def general_step(self, mode, batch, batch_idx):
        x, _ = batch
        if x.min() < 0:
            x = x * 0.5 + 0.5

        x_flat = rearrange(x, 'b c m n -> b (c m n)')
        latent = self.encoder(x_flat)
        x_hat_flat = self.decoder(latent)
        x_hat = rearrange(x_hat_flat, 'b (c m n) -> b c m n', c=1, m=x.shape[-2], n=x.shape[-1])
        loss = self.mse(x_hat, x)

        encoder_l2_reg_loss = self.encoder.apply_l2_regularization(reg_param=1e-8)
        decoder_l2_reg_loss = self.decoder.apply_l2_regularization(reg_param=1e-8)
        tv_reg_loss = tv_loss(x_hat, x)

        loss += encoder_l2_reg_loss + decoder_l2_reg_loss + 0.1 * tv_reg_loss

        mae_loss = self.mae(x_hat, x)
        psnr_loss = self.psnr(x_hat, x, data_range=1.0)
        ssim_loss = self.ssim(x_hat, x, data_range=1.0)

        self.log_dict({f'{mode}_loss': loss,
                       f'{mode}_tv': tv_reg_loss,
                       f'{mode}_mae': mae_loss,
                       f'{mode}_psnr': psnr_loss,
                       f'{mode}_ssim': ssim_loss}, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.general_step('val', batch, batch_idx)

    def load_checkpoint(self, checkpoint_path, trainable=False):
        checkpoint = torch.load(checkpoint_path)['state_dict']

        encoder_checkpoint = {}
        decoder_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith('encoder'):
                encoder_checkpoint[key.replace('encoder.', '')] = value
            elif key.startswith('decoder'):
                decoder_checkpoint[key.replace('decoder.', '')] = value

        self.encoder.load_state_dict(encoder_checkpoint)
        self.decoder.load_state_dict(decoder_checkpoint)

        self.encoder.requires_grad_(trainable)
        self.decoder.requires_grad_(trainable)

    def get_model(self):
        return nn.Sequential(self.encoder, self.decoder)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=self.patient,
                                                               verbose=True, mode='max')
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=reduce_lr, monitor='train_psnr'))
