import argparse
import os
from itertools import product
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets

from algorithms.optical_encoders import LinearSPC
from autoencoder import GenADMM, Decoder_GAN, GenSPCADMM

from utils.callbacks import MyPrintingCallback, OverrideEpochStepCallback
from utils.exp_setting import experiment_setting

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb
from datetime import datetime

from deepinv.physics import SinglePixelCamera

# torch.set_float32_matmul_precision('high')


def init_parser():
    parser = argparse.ArgumentParser(description='Training script')

    # data args

    parser.add_argument('--save-name', default='admm_spi_digit_mnist',
                        help='path to save specific experiment (This will be stored in result_path folder)')
    parser.add_argument('--generator-path',
                        default='pretrained_generators/elu/mnist_netG.ckpt',
                        help='path to the pre-trained generator')

    # datasets and model

    parser.add_argument('--data-dir', default='data/single_pixel_mnist',
                        help='path to the data directory')
    parser.add_argument('--dataset-name', default='mnist', type=str,
                        help='dataset name')
    parser.add_argument('--image-size', default=32, type=int,
                        help='image size for training')
    parser.add_argument('--num-bands', default=1, type=int,
                        help='number of bands in the dataset')
    parser.add_argument('--num-channels', default=1, type=int,
                        help='number of channels in the dataset')

    # hyper parameters

    parser.add_argument('--z-dim', default=128, type=int,
                        help='latent space dimension')
    parser.add_argument('--cr', default=0.1, type=float,
                        help='Compression ratio')
    parser.add_argument('--sensing-matrix', default='zig_zag',
                        choices=['zig_zag', 'cake_cutting', 'random_binary'],
                        help='sensing matrix type')
    parser.add_argument('--gamma', default=0.001, type=float, #0.001
                        help='ADMM gamma')
    parser.add_argument('--beta', default=10, type=float,
                        help='ADMM beta')
    parser.add_argument('--sigma', default=1, type=float,
                        help='ADMM sigma')
    parser.add_argument('--max-iter', default=10000, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--n-images', default=32, type=int,
                        help='NUmber of test images')
    parser.add_argument('--elu-gan', default=True, type=bool,
                        help='Use ELU activation in GAN')

    # gpu config

    parser.add_argument('-j', '--num-workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')

    return parser


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = init_parser()
    args = parser.parse_args()

    name = 'base_spi_admm'
    experiment_setting(__file__, name, args)

    entity_name = 'generative_stsiva'
    date = datetime.today().strftime('%Y_%m_%d_%H_%M')
    config = (f"{name}_sm{args.sensing_matrix}_samples_{args.n_images}_cr{args.cr}_iter_{args.max_iter}_"
              f"gamma_{args.gamma}_beta_{args.beta}_sigma_{args.sigma}_{date}")
    wandb.login(key="890516cdb328d76a5ba65e9fd699f5f70696edf5")
    wandb.Api()
    wandb.init(project='gen_stsiva_spi', entity=entity_name, name=config, config=vars(args))

    print(f"\n {args} \n")

    # dataset

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.MNIST(root="data", train=False, transform=transform, download=True)
    test_dataset = Subset(test_dataset, range(0, args.n_images))  # testing with first 100 images
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Datasets loaded!')

    optical_encoder = LinearSPC((args.image_size, args.image_size, args.num_channels),
                                compression_ratio=args.cr, sensing_matrix=args.sensing_matrix, snr=-1)

    # model
    # generator = load_generator(args.generator_path, args.dataset_name, input_dim=args.z_dim, elu=args.elu_gan)
    generator = Decoder_GAN(elu = args.elu_gan, single_pixel = True).eval()
    model = GenSPCADMM(optical_encoder, generator, 'base', args.z_dim, gamma=args.gamma, beta=args.beta,
                       sigma=args.sigma, max_iter=args.max_iter)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')

    # train

    wdb_logger = WandbLogger(project='generative_stsiva', name=args.save_name, save_code=False, log_model=False)
    wdb_logger.experiment.config.update(vars(args))
    csv_logger = CSVLogger('results', name=f'{name}/{args.save_name}/csv')

    trainer = L.Trainer(max_epochs=1, logger=[wdb_logger, csv_logger], precision='32',
                        callbacks=[MyPrintingCallback(), OverrideEpochStepCallback(), lr_monitor],
                        log_every_n_steps=1, enable_progress_bar=False)

    trainer.fit(model, test_loader)
    model.compute_metrics()

if __name__ == '__main__':
    main()
