import argparse
import os

import torch
import torch.utils.data as data

from torchvision.datasets import MNIST
from torchvision import transforms
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

    # datasets
    save_dir = os.path.join("data", "single_pixel_mnist")
    os.makedirs(save_dir, exist_ok=True)

    camera = SinglePixelCamera(
        m=int(args.image_size**2 * args.cr), img_shape=(1, args.image_size, args.image_size), fast=True
    )

    # H = torch.randn(args.image_size, args.image_size) / torch.sqrt(torch.tensor(args.image_size, dtype=torch.float)) # Gaussian mask


    print(f'Saving SPI datasets on data folder at {save_dir}...')

    print('Applying SPI Camera on test images...')  # Each batch has: (attacked_images, original_images, labels)
    test_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                         transforms.CenterCrop(args.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                         ])
    test_dataset = MNIST('data', train=False, transform=test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = transform_images_loader(test_loader, camera, args)

    save_path = os.path.join(save_dir, f"spi_test_loader_cr_{str(args.cr).replace('.', '')}.pt")
    torch.save(test_loader, save_path)
    print(f"SPI test dataset has been saved in {save_path}")


if __name__ == '__main__':
    main()
