import argparse
import os
import glob
import torch
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision import transforms
from algorithms.optical_encoders import LinearSPC
from autoencoder import Autoencoder_GAN_MNIST_SPI
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime
import matplotlib.pyplot as plt

########## THIS IS THE OFFICIAL ONE ###########

def init_parser():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--image-size', default=32, type=int, help='image size for inference')
    parser.add_argument('--num-channels', default=1, type=int, help='number of channels in the dataset')
    parser.add_argument('--cr', default=0.01, type=float, help='Compression ratio')
    parser.add_argument('--sensing-matrix', default='zig_zag', choices=['zig_zag', 'cake_cutting', 'random_binary'], help='sensing matrix type')
    parser.add_argument('--elu_encoder', dest='elu_encoder', default=False, help='Use ELU activation in Encoder')
    parser.add_argument('--elu_gan', dest='elu_gan', default=True, help='Use ELU activation in GAN')
    parser.add_argument('--num-workers', default=6, type=int, metavar='N', help='number of data loading workers (default: 6)')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size for inference')
    parser.add_argument('--test_examples', default=16, type=int, help='Number of examples test images to show')
    return parser

def main():
    torch.manual_seed(42)
    parser = init_parser()
    args = parser.parse_args()
    print(f"ELU Encoder: {args.elu_encoder}, ELU GAN: {args.elu_gan}")
    # Dataset
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    test_dataset = MNIST('data', train=False, transform=test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    print('MNIST test dataset loaded!')
    # Optical encoder
    optical_encoder = LinearSPC((args.image_size, args.image_size, args.num_channels),
                                compression_ratio=args.cr, sensing_matrix=args.sensing_matrix, snr=-1,
                                device='cuda' if torch.cuda.is_available() else 'cpu')
    # Model
    model = Autoencoder_GAN_MNIST_SPI(
        optical_encoder=optical_encoder,
        elu_gan=args.elu_gan,
        elu_encoder=args.elu_encoder
    ).to(optical_encoder.H.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')
    # Load checkpoint
    print('Loading pre-trained model...')
    checkpoint_path = (f'results/prop_init_spi/'
                       f'spc_prop_init_digit_mnist_cr{args.cr}_sm{args.sensing_matrix}'
                       f'_Encoder_elu_{args.elu_encoder}_GAN_elu_{args.elu_gan}_2025_05_04_17_15/'
                       f'checkpoints')
    checkpoint_filename = glob.glob(f'{checkpoint_path}/epoch*')[0]
    print(f'Loading checkpoint from: {checkpoint_filename}')
    checkpoint = torch.load(checkpoint_filename, map_location='cpu')['state_dict']
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
    # Inference
    batch = next(iter(test_loader))
    batch_on_device = (batch[0].to(optical_encoder.H.device), batch[1]) if isinstance(batch, (list, tuple)) else batch.to(optical_encoder.H.device)
    inferences = model.test_step(batch_on_device)
    originals = inferences["original"]
    reconstructed = inferences["reconstructed"]
    psnr_values = inferences["psnr"]
    ssim_values = inferences["ssim"]
    fig, axes = plt.subplots(2, args.test_examples, figsize=(args.test_examples * 2, 4))
    for i in range(args.test_examples):
        axes[0, i].imshow(originals[i].detach().cpu().squeeze().numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].detach().cpu().squeeze().numpy(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"PSNR: {psnr_values[i]:.2f}\nSSIM: {ssim_values[i]:.2f}")
    fig.text(0.04, 0.75, 'Original', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
    fig.text(0.04, 0.25, 'Reconstructed', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
    plt.tight_layout(rect=[0.06, 0, 1, 1])
    plt.show()

if __name__ == '__main__':
    main()
