import argparse
import os
from itertools import product
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from torch.utils.data import Subset

from torchvision.datasets import MNIST
from torchvision import transforms

from autoencoder import Autoencoder_GAN_MNIST
from utils.callbacks import MyPrintingCallback, OverrideEpochStepCallback
from utils.exp_setting import experiment_setting
from src.util.attack import attack_images_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb
from datetime import datetime

torch.set_float32_matmul_precision('high')


def init_parser():
    parser = argparse.ArgumentParser(description='Training script')

    # data args

    parser.add_argument('--save-name', default='prop_init_digit_mnist',
                        help='path to save specific experiment (This will be stored in result_path folder)')

    # datasets and model

    parser.add_argument('--image-size', default=28, type=int,
                        help='image size for training')
    parser.add_argument('--num-bands', default=1, type=int,
                        help='number of bands in the dataset')
    parser.add_argument('--num-channels', default=1, type=int,
                        help='number of channels in the dataset')

    # hyper parameters

    parser.add_argument('--max-epochs', default=101, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--std', type=float, default=0.2,
                        help='standard deviation for noise')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--patient', default=15 * 10, type=int,
                        help='number of epochs to wait before reducing lr')

    # ELU or ReLU Encoder
    parser.add_argument('--elu_encoder', dest='elu_encoder', default=False,
                        help='Use ELU activation in Encoder')
    parser.add_argument('--elu_gan', dest='elu_gan', default=True,
                        help='Use ELU activation in GAN')
    parser.add_argument('--adversarial_attack', type=bool, default=True,
                        help='Use attacked images ')

    # gpu config

    parser.add_argument('-j', '--num-workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')

    # Training or inference

    parser.add_argument('--train', default=True, type=bool,
                        help='False --> To load a pre-trained model and start inference')
    parser.add_argument('--test_examples', default=7, type=int,
                        help='Number of examples test images to show')

    return parser


def main():
    torch.manual_seed(42)

    parser = init_parser()
    args = parser.parse_args()

    print(f"ELU Encoder: {args.elu_encoder}, ELU GAN: {args.elu_gan}")  # Debug

    name = 'prop_init'
    experiment_setting(__file__, name, args)

    entity_name = 'generative_stsiva'
    date = datetime.today().strftime('%Y_%m_%d_%H_%M')
    config = (f"{name}_std{args.std}_Encoder_elu_{args.elu_encoder}_GAN_elu_{args.elu_gan}_{date}")
    wandb.login(key="890516cdb328d76a5ba65e9fd699f5f70696edf5")
    wandb.Api()
    wandb.init(project='gan_inv1', entity=entity_name, name=config, config=vars(args))

    # datasets
    print('Loading datasets...')
    train_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                          transforms.CenterCrop(args.image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                          ])

    full_train_dataset = MNIST(root="data", train=True, transform=train_transform, download=True)
    split_idx = int(0.8 * len(full_train_dataset))  # 80% for training
    train_dataset = Subset(full_train_dataset, range(0, split_idx))
    valid_dataset = Subset(full_train_dataset, range(split_idx, len(full_train_dataset)))

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    train_loader = attack_images_loader(train_loader, args.std, args)
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)
    valid_loader = attack_images_loader(valid_loader, args.std, args)

    test_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                         transforms.CenterCrop(args.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                         ])
    test_dataset = MNIST('data', train=False, transform=test_transform)
    # test_dataset = Subset(test_dataset, range(0, 100))  # testing with first 100 images
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = attack_images_loader(test_loader, args.std, args)
    print('Datasets loaded!')

    # model

    model = Autoencoder_GAN_MNIST(lr=args.lr, patient=args.patient,
                                  elu_gan=args.elu_gan, elu_encoder=args.elu_encoder,
                                  adversarial=args.adversarial_attack)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')

    # train

    wdb_logger = WandbLogger(project='generative_stsiva', name=args.save_name, save_code=False, log_model=False)
    wdb_logger.experiment.config.update(vars(args))
    csv_logger = CSVLogger('results', name=f'{name}/{args.save_name}/csv')
    checkpoint_callback = ModelCheckpoint(dirpath=f'results/{name}/{args.save_name}/checkpoints', save_last=True,
                                          save_top_k=1, monitor='recon/val_psnr', mode='max')

    trainer = L.Trainer(max_epochs=args.max_epochs, logger=[wdb_logger, csv_logger], precision='16-mixed',
                        callbacks=[MyPrintingCallback(), OverrideEpochStepCallback(), checkpoint_callback,
                                   lr_monitor],
                        log_every_n_steps=1, enable_progress_bar=False)

    trainer.fit(model, train_loader, val_dataloaders=valid_loader)
    trainer.test(model, test_loader, ckpt_path=checkpoint_callback.best_model_path)


if __name__ == '__main__':
    main()
