import argparse
import os

import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from autoencoder import Autoencoder_GAN_MNIST
from utils.callbacks import MyPrintingCallback, OverrideEpochStepCallback
from utils.exp_setting import experiment_setting

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb
from datetime import datetime

torch.set_float32_matmul_precision('high')


def init_parser():
    parser = argparse.ArgumentParser(description='Training script')

    # data args

    parser.add_argument('--save-name', default='autoencoder_digit_mnist_test100',
                        help='path to save specific experiment (This will be stored in result_path folder)')

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

    parser.add_argument('--max-epochs', default=101, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--patient', default=15 * 10, type=int,
                        help='number of epochs to wait before reducing lr')
    parser.add_argument('--std', default=0.2, type=float,
                        help='Threshold for the adversarial attack')

    # ELU or ReLU Encoder
    parser.add_argument('--elu_encoder', dest='elu_encoder', action='store_true',
                        help='Use ELU activation in Encoder')
    parser.add_argument('--no-elu_encoder', dest='elu_encoder', action='store_false',
                        help='Use ReLU activation in Encoder')
    parser.set_defaults(elu_encoder=False)

    # ELU or ReLU GAN
    parser.add_argument('--elu_gan', dest='elu_gan', action='store_true',
                        help='Use ELU activation in GAN')
    parser.add_argument('--no-elu_gan', dest='elu_gan', action='store_false',
                        help='Use ReLU activation in GAN')
    parser.set_defaults(elu_gan=True)

    # gpu config

    parser.add_argument('-j', '--num-workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')
    
    return parser


def main():
    torch.manual_seed(42)

    parser = init_parser()
    args = parser.parse_args()
    print(args)

    print(f"ELU Encoder: {args.elu_encoder}, ELU GAN: {args.elu_gan}")  # Debug

    name = 'autoencoder'
    experiment_setting(__file__, name, args)

    entity_name = 'generative_stsiva'
    date = datetime.today().strftime('%Y_%m_%d_%H_%M')
    config = (f"Encoder_elu_{args.elu_encoder}_GAN_elu_{args.elu_gan}_{date}")
    wandb.login(key="890516cdb328d76a5ba65e9fd699f5f70696edf5")
    wandb.Api()
    wandb.init(project='autoencoder_adversarial_mnist', entity=entity_name, name=config, config=args)

    # Datasets
    
    attacked_train_dataset = torch.load(
        os.path.join(args.data_dir, f"attacked_train_loader_std_{str(args.std).replace('.', '')}.pt"),
        weights_only=False)
    train_loader = torch.utils.data.DataLoader(attacked_train_dataset, batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True,
                                                persistent_workers=True)
    print("Attacked train dataset loaded successfully.")

    attacked_valid_dataset = torch.load(
        os.path.join(args.data_dir, f"attacked_valid_loader_std_{str(args.std).replace('.', '')}.pt"),
        weights_only=False)
    valid_loader = torch.utils.data.DataLoader(attacked_valid_dataset, batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True,
                                                persistent_workers=True)
    print("Attacked valid dataset loaded successfully.")

    attacked_test_dataset = torch.load(
        os.path.join(args.data_dir, f"attacked_test_loader_std_{str(args.std).replace('.', '')}.pt"),
        weights_only=False)
    test_loader = torch.utils.data.DataLoader(attacked_test_dataset, batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True,
                                                persistent_workers=True)
    print("Attacked test dataset loaded successfully.")

    print('Datasets loaded!')

    # Model

    model = Autoencoder_GAN_MNIST(lr=args.lr, patient=args.patient,
                                  elu_gan=args.elu_gan, elu_encoder=args.elu_encoder,
                                  adversarial = True)
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
