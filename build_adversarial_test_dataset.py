import argparse
import os

import torch
import torch.utils.data as data

from torchvision.datasets import MNIST
from torchvision import transforms
from src.util.attack import attack_images_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.set_float32_matmul_precision('high')


def init_parser():
    parser = argparse.ArgumentParser(description='Dataset saver script')

    parser.add_argument('--dataset-name', default='mnist', type=str,
                        help='dataset name')
    parser.add_argument('--image-size', default=28, type=int,
                        help='image size for training')
    parser.add_argument('--num-channels', default=1, type=int,
                        help='number of channels in the dataset')
    parser.add_argument('--std', default=0.1, type=float,
                        help='Standard deviation of the noise')

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
    save_dir = os.path.join("data", "attacked_mnist")
    os.makedirs(save_dir, exist_ok=True)

    print('Saving attacked test dataset on data folder...')

    print(
        'Applying adversarial attack on test images...')  # Each batch has: (attacked_images, original_images, labels)
    test_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                         transforms.CenterCrop(args.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                         ])
    test_dataset = MNIST('data', train=False, transform=test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = attack_images_loader(test_loader, args)

    save_path = os.path.join(save_dir, f"attacked_test_loader_std_{str(args.std).replace('.', '')}.pt")
    torch.save(test_loader.dataset, save_path)
    print(f"Attacked test dataset has been saved in {save_path}")


if __name__ == '__main__':
    main()