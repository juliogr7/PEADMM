import argparse
import glob
import os

import torch
import torch.utils.data as data
from torch.utils.data import Subset

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from autoencoder import GenADMM, ConvolutionalEncoder, Decoder_GAN
from src.models.loader import load_generator

from utils.callbacks import MyPrintingCallback, OverrideEpochStepCallback
from utils.exp_setting import experiment_setting

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb
from datetime import datetime


# torch.set_float32_matmul_precision('high')

def init_parser():
    parser = argparse.ArgumentParser(description='Training script')

    # data args

    parser.add_argument('--generator-path',
                        default='pretrained_generators/elu/mnist_netG.ckpt',
                        help='path to the pre-trained generator')

    # datasets and model

    parser.add_argument('--dataset-name', default='mnist', type=str,
                        help='dataset name')
    parser.add_argument('--data-dir', default='data/attacked_mnist',
                        help='path to the data directory')
    parser.add_argument('--image-size', default=28, type=int,
                        help='image size for training')
    parser.add_argument('--num-bands', default=1, type=int,
                        help='number of bands in the dataset')
    parser.add_argument('--num-channels', default=1, type=int,
                        help='number of channels in the dataset')

    # hyper parameters

    parser.add_argument('--z-dim', default=128, type=int,
                        help='latent space dimension')
    parser.add_argument('--std', default=0.25, type=float,
                        help='Standard deviation of the noise')
    parser.add_argument('--gamma', default=1e-5, type=float,
                        help='ADMM gamma')
    parser.add_argument('--beta', default=1e-6, type=float,
                        help='ADMM beta')
    parser.add_argument('--sigma', default=1e-12, type=float,
                        help='ADMM sigma')
    parser.add_argument('--max-iter', default=25100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--n-images', default=10000, type=int,
                        help='NUmber of test images')
    
    # ELU or ReLU Encoder
    parser.add_argument('--elu_encoder', dest='elu_encoder', default=False,
                        help='Use ELU activation in Encoder')
    parser.add_argument('--elu_gan', dest='elu_gan', default=True,
                        help='Use ELU activation in GAN')

    # gpu config

    parser.add_argument('-j', '--num-workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')

    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()

    name = 'prop_admm'

    entity_name = 'generative_stsiva'
    date = datetime.today().strftime('%Y_%m_%d_%H_%M')
    config = (f"{name}_samples_{args.n_images}_bs_{args.batch_size}_std_{args.std}_iter_{args.max_iter}_"
              f"gamma_{args.gamma}_beta_{args.beta}_sigma_{args.sigma}_{date}")

    args.save_name = config
    experiment_setting(__file__, name, args)

    # WandB Logging
    wandb.login(key="Put_your_key_here")
    wandb.Api()
    wandb.init(project="name", entity=entity_name, name=config, config=args)

    print(f'\n {args} \n')

    # dataset

    attacked_test_dataset = torch.load(
        os.path.join(args.data_dir, f"attacked_test_loader_std_{str(args.std).replace('.', '')}.pt"),
        weights_only=False)
    attacked_test_dataset = Subset(attacked_test_dataset, range(0, args.n_images))
    test_loader = data.DataLoader(attacked_test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, 
                                  persistent_workers=True)
    print('Datasets loaded!')

    # model

    encoder = ConvolutionalEncoder(input_dim=128, hidden_dim=64, elu=args.elu_encoder).eval()
    # checkpoint_path = (f'results/prop_init/prop_init_digit_mnist/checkpoints')
    epoch = {0.1: 63, 0.2: 66, 0.25: 36}
    checkpoint_path = (f'results/pretrained_adversarial/'
                           f'autoencoder_digit_mnist_std{args.std}_Encoder_elu_False_GAN_elu_True/'
                           f'checkpoints/epoch={epoch[args.std]}-step={epoch[args.std]}.ckpt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    encoder.load_state_dict(
        {k.replace('encoder.', ''): v for k, v in checkpoint.items() if 'encoder' in k})

    # generator = load_generator(args.generator_path, args.dataset_name, input_dim=args.z_dim, elu=args.elu_gan).eval()
    generator = Decoder_GAN(elu = args.elu_gan).eval()
    model = GenADMM(generator, 'prop', args.z_dim, gamma=args.gamma, beta=args.beta, sigma=args.sigma,
                    max_iter=args.max_iter, encoder=encoder)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')

    # train

    wdb_logger = WandbLogger(project='stsiva', name=args.save_name, save_code=False, log_model=False)
    wdb_logger.experiment.config.update(vars(args))
    csv_logger = CSVLogger('results', name=f'{name}/{args.save_name}/csv')

    trainer = L.Trainer(max_epochs=1, logger=[wdb_logger, csv_logger], precision='32',
                        callbacks=[MyPrintingCallback(), OverrideEpochStepCallback(), lr_monitor],
                        log_every_n_steps=1, enable_progress_bar=False)

    trainer.fit(model, test_loader)
    model.compute_metrics()

if __name__ == '__main__':
    main()
