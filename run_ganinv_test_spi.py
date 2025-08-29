import argparse
import os
import matplotlib.pyplot as plt
import glob

import torch
import torch.utils.data as data
from torchvision import datasets, transforms

from autoencoder import Autoencoder_GAN_MNIST_SPI
from algorithms.optical_encoders import LinearSPC
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb
from datetime import datetime

torch.set_float32_matmul_precision('high')


def init_parser():
    parser = argparse.ArgumentParser(description='Test script')

    # datasets and model
    parser.add_argument('--data-dir', default='data',
                        help='path to the data directory')
    parser.add_argument('--dataset-name', default='mnist', type=str,
                        help='dataset name')
    parser.add_argument('--image-size', default=32, type=int,
                        help='image size')
    parser.add_argument('--cr', default=0.01, type=float,
                        help='Compression ratio for SPI measurements')

    # SPI specific parameters
    parser.add_argument('--sensing-matrix', default='zig_zag', type=str,
                        help='Type of sensing matrix: zig_zag, cake_cutting, random_binary')
    parser.add_argument('--snr', default=-1, type=float,
                        help='Signal to noise ratio (-1 for no noise)')

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

    parser.add_argument('--test_examples', default=7, type=int,
                        help='Number of examples test images to show')

    return parser


def main():
    torch.manual_seed(42)

    parser = init_parser()
    args = parser.parse_args()
    print(args)

    print(f"ELU Encoder: {args.elu_encoder}, ELU GAN: {args.elu_gan}")  # Debug

    # dataset - Load regular MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    test_dataset = datasets.MNIST(
        root=args.data_dir, 
        train=False, 
        download=True,
        transform=transform
    )
    
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=100, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    print('MNIST test dataset loaded!')

    # Create optical encoder for SPI measurements - using proper format
    input_shape = (args.image_size, args.image_size, 1)  # (H, W, channels) format for MNIST
    
    # try:
    optical_encoder = LinearSPC(
        input_shape=input_shape,
        compression_ratio=args.cr,
        sensing_matrix=args.sensing_matrix,
        snr=args.snr,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Optical encoder created with compression ratio: {args.cr} and sensing matrix: {args.sensing_matrix}')
    # except Exception as e:
    #     print(f"Error creating optical encoder with {args.sensing_matrix}: {e}")
    #     print("Falling back to random_binary sensing matrix...")
    #     optical_encoder = LinearSPC(
    #         input_shape=input_shape,
    #         compression_ratio=args.cr,
    #         sensing_matrix='random_binary',
    #         snr=args.snr,
    #         device='cuda' if torch.cuda.is_available() else 'cpu'
    #     )
    #     print(f'Optical encoder created with compression ratio: {args.cr} and random_binary sensing matrix')

    # model - Use SPI-specific autoencoder
    model = Autoencoder_GAN_MNIST_SPI(
        optical_encoder=optical_encoder,
        elu_gan=args.elu_gan, 
        elu_encoder=args.elu_encoder
    ).to(optical_encoder.H.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')

    # Test
    print('Loading pre-trained model...')
    
    # Dynamic checkpoint path based on parameters
    checkpoint_path = (f'results/prop_init_spi/'
                       f'spc_prop_init_digit_mnist_cr{args.cr}_sm{args.sensing_matrix}'
                       f'_Encoder_elu_{args.elu_encoder}_GAN_elu_{args.elu_gan}_2025_05_04_17_15/'
                       f'checkpoints')
    
    checkpoint_filename = glob.glob(f'{checkpoint_path}/epoch*')[0]
    print(f'Loading checkpoint from: {checkpoint_filename}')
    checkpoint = torch.load(checkpoint_filename, map_location='cpu')['state_dict']
    
    # Load the state dict into the model
    encoder_checkpoint = {}
    decoder_checkpoint = {}
    for key, value in checkpoint.items():
        if key.startswith('encoder'):
            encoder_checkpoint[key.replace('encoder.', '')] = value
        elif key.startswith('decoder'):
            decoder_checkpoint[key.replace('decoder.', '')] = value

    model.encoder.load_state_dict(encoder_checkpoint)
    model.decoder.load_state_dict(decoder_checkpoint)
    print('Model loaded successfully!')
    
    model.eval()

    batch = next(iter(test_loader))
    # Mover el vector de imagen al mismo dispositivo que la matriz H
    batch_on_device = (batch[0].to(optical_encoder.H.device), batch[1]) if isinstance(batch, (list, tuple)) else batch.to(optical_encoder.H.device)
    inferences = model.test_step(batch_on_device)

    originals = inferences["original"]
    reconstructed = inferences["reconstructed"] 
    measurements = inferences["transformed"]  # These are the SPI measurements, not compressed images
    psnr_values = inferences["psnr"]
    ssim_values = inferences["ssim"]

    fig, axes = plt.subplots(3, args.test_examples, figsize=(args.test_examples * 2, 6))

    for i in range(args.test_examples):
        
        # Show original image
        axes[0, i].imshow(originals[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')

        # Show reconstructed image
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"PSNR: {psnr_values[i]:.2f}\nSSIM: {ssim_values[i]:.2f}")
        
        # Show measurement visualization (as a 1D plot since measurements are vectors)
        if i == 0:  # Only show measurements for first image to avoid clutter
            axes[2, i].plot(measurements[i].cpu().numpy().flatten()[:100])  # Show first 100 measurements
            axes[2, i].set_title("SPI Measurements (first 100)")
        else:
            axes[2, i].axis('off')

    fig.text(0.04, 0.84, 'Original', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
    fig.text(0.04, 0.5, 'Reconstructed', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
    fig.text(0.04, 0.17, f'Measurements (CR={args.cr})', va='center', ha='left', fontsize=14, weight='bold', rotation=90)

    plt.tight_layout(rect=[0.06, 0, 1, 1])
    plt.show()



if __name__ == '__main__':
    main()
