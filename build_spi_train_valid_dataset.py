import argparse
import os

import torch
import torch.utils.data as data

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset
from src.util.attack import transform_images_loader

from deepinv.physics import SinglePixelCamera

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.set_float32_matmul_precision('high')


def init_parser():
    parser = argparse.ArgumentParser(description='Dataset saver script')

    parser.add_argument('--dataset-name', default='mnist', type=str,
                        help='dataset name')
    parser.add_argument('--image-size', default=32, type=int,
                        help='image size for training')
    parser.add_argument('--num-channels', default=1, type=int,
                        help='number of channels in the dataset')
    parser.add_argument('--cr', default=0.3, type=float,
                        help='Compression ratio')

    parser.add_argument('--batch-size', default=100, type=int,
                        help='batch size')
    parser.add_argument('-j', '--num-workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')

    return parser


def main():
    torch.manual_seed(42)

    parser = init_parser()
    args = parser.parse_args()

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
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    # datasets

    save_dir = os.path.join("data", "single_pixel_mnist")
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, f"spi_train_loader_cr_{str(args.cr).replace('.', '')}.pt")
    valid_path = os.path.join(save_dir, f"spi_valid_loader_cr_{str(args.cr).replace('.', '')}.pt")

    camera = SinglePixelCamera(
        m=int(args.image_size**2 * args.cr), img_shape=(1, args.image_size, args.image_size), fast=True)
    
    print(f'Saving SPI datasets on data folder at {save_dir}...')

    print('Applying Single-Pixel Camera on train images...')
    train_loader = transform_images_loader(train_loader, camera, args)  # Each batch has: (attacked_images, original_images, labels)

    print('Applying Single-Pixel Camera on validation images...')  # Each batch has: (attacked_images, original_images, labels)
    valid_loader = transform_images_loader(valid_loader, camera, args)

    torch.save(train_loader.dataset, train_path)
    print(f"SPI train dataset has been saved in {train_path}")

    torch.save(valid_loader.dataset, valid_path)
    print(f"SPI valid dataset has been saved in {valid_path}")

    print('Datasets saved!')


if __name__ == '__main__':
    main()
