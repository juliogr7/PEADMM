import argparse
import os
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision import transforms

from autoencoder import Autoencoder_GAN_MNIST
from utils.exp_setting import experiment_setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb
from datetime import datetime

torch.set_float32_matmul_precision('high')


def init_parser():
    parser = argparse.ArgumentParser(description='Test script')

    # datasets and model
    parser.add_argument('--dataset-name', default='mnist', type=str,
                        help='dataset name')
    parser.add_argument('--data-dir', default='data/attacked_mnist',
                        help='path to the data directory')
    parser.add_argument('--image-size', default=28, type=int,
                        help='image size')
    parser.add_argument('--std', default=0.1, type=float,
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

    # Inference

    parser.add_argument('--test_examples', default=16, type=int,
                        help='Number of examples test images to show')

    return parser


def main():
    torch.manual_seed(42)

    parser = init_parser()
    args = parser.parse_args()
    print(args)

    print(f"ELU Encoder: {args.elu_encoder}, ELU GAN: {args.elu_gan}")  # Debug

    # dataset

    attacked_test_dataset = torch.load(
        os.path.join(args.data_dir, f"attacked_test_loader_std_{str(args.std).replace('.', '')}.pt"),
        weights_only=False)
    test_loader = data.DataLoader(attacked_test_dataset, batch_size=100, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    print('Test dataset loaded!')


    # model
    model = Autoencoder_GAN_MNIST(elu_gan=args.elu_gan, elu_encoder=args.elu_encoder, adversarial = True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')


    # Test
    dir = os.path.join('results', 'autoencoder', 'autoencoder_digit_mnist', 'checkpoints',
                        'epoch=84-step=84.ckpt')
    model.load_checkpoint(dir)
    model.eval()

    print('Starting inference...')

    inferences = model.test_step(next(iter(test_loader)))

    originals = inferences["original"]
    reconstructed = inferences["reconstructed"]
    attacked = inferences["transformed"]
    psnr_values = inferences["psnr"]
    ssim_values = inferences["ssim"]

    fig, axes = plt.subplots(3, args.test_examples, figsize=(args.test_examples * 2, 6))

    for i in range(args.test_examples):
        
        axes[0, i].imshow(attacked[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')

        axes[1, i].imshow(originals[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title(f"PSNR: {psnr_values[i]:.2f}\nSSIM: {ssim_values[i]:.2f}")

    fig.text(0.04, 0.83, f'Attacked ({args.std})', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
    fig.text(0.04, 0.5, 'Original', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
    fig.text(0.04, 0.15, 'Reconstructed', va='center', ha='left', fontsize=14, weight='bold', rotation=90)

    plt.tight_layout(rect=[0.06, 0, 1, 1])
    plt.show()



if __name__ == '__main__':
    main()
