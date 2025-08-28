import argparse
import os
import glob
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import numpy as np

from torchvision.datasets import MNIST
from torchvision import transforms

from algorithms.optical_encoders import LinearSPC
from autoencoder import Autoencoder_GAN_MNIST_SPI
from image_metrics import compute_psnr, compute_ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_float32_matmul_precision('high')


def init_parser():
    parser = argparse.ArgumentParser(description='Inference script for SPI model')

    # Model and checkpoint args
    parser.add_argument('--checkpoint-path', default="results/prop_init_spi/spc_prop_init_digit_mnist_cr0.1_smzig_zag_Encoder_elu_False_GAN_elu_True_2025_05_04_17_15/checkpoints/epoch=94-step=94.ckpt", type=str,
                        help='Path to the checkpoint file. If not provided, will search automatically.')
    parser.add_argument('--results-dir', default='results/prop_init_spi', type=str,
                        help='Directory where results are stored')
    
    # Data args
    parser.add_argument('--image-size', default=32, type=int,
                        help='image size for inference')
    parser.add_argument('--num-channels', default=1, type=int,
                        help='number of channels in the dataset')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size for inference')
    
    # Model hyperparameters (must match training)
    parser.add_argument('--cr', default=0.1, type=float,
                        help='Compression ratio')
    parser.add_argument('--sensing-matrix', default='zig_zag',
                        choices=['zig_zag', 'cake_cutting', 'random_binary'],
                        help='sensing matrix type')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--patient', default=15 * 10, type=int,
                        help='number of epochs to wait before reducing lr')
    
    # ELU or ReLU settings (must match training)
    parser.add_argument('--elu_encoder', dest='elu_encoder', default=False, type=bool,
                        help='Use ELU activation in Encoder')
    parser.add_argument('--elu_gan', dest='elu_gan', default=True, type=bool,
                        help='Use ELU activation in GAN')
    
    # Inference settings
    parser.add_argument('-j', '--num-workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')
    parser.add_argument('--test_examples', default=30, type=int,
                        help='Number of examples test images to show')
    parser.add_argument('--save-results', action='store_true',
                        help='Save inference results to file')
    parser.add_argument('--output-dir', default='inference_results', type=str,
                        help='Directory to save inference results')
    
    return parser


def find_checkpoint(results_dir, cr, sensing_matrix, elu_encoder, elu_gan):
    """Find the checkpoint file based on model parameters"""
    # Search pattern for the experiment directory
    search_pattern = (f"{results_dir}/"
                     f"spc_prop_init_digit_mnist_cr{cr}_sm{sensing_matrix}"
                     f"_Encoder_elu_{elu_encoder}_GAN_elu_{elu_gan}_*")
    
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        raise FileNotFoundError(f"No experiment directories found matching pattern: {search_pattern}")
    
    # Use the most recent directory (assuming date format in name)
    exp_dir = sorted(matching_dirs)[-1]
    
    # Find checkpoint files
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Return the best checkpoint (usually the one with 'best' in name, or the last one)
    best_checkpoints = [f for f in checkpoint_files if 'best' in f]
    if best_checkpoints:
        return best_checkpoints[0]
    else:
        return sorted(checkpoint_files)[-1]


def load_model(checkpoint_path, cr, sensing_matrix, elu_encoder, elu_gan, lr, patient, image_size, num_channels):
    """Load the model from checkpoint"""
    print(f'Loading model from: {checkpoint_path}')
    
    # Create optical encoder (must match training setup)
    optical_encoder = LinearSPC((image_size, image_size, num_channels),
                               compression_ratio=cr, sensing_matrix=sensing_matrix, snr=-1)
    
    # Create model
    model = Autoencoder_GAN_MNIST_SPI(optical_encoder, lr=lr, patient=patient, 
                                     elu_gan=elu_gan, elu_encoder=elu_encoder)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print('Model loaded successfully!')
    return model


def run_inference(model, test_loader, device='cuda', target_indices=[11, 15]):
    """Run inference on the test dataset, filtering only target indices"""
    model = model.to(device)
    model.eval()
    
    all_results = {
        'originals': [],
        'reconstructed': [],
        'measurements': [],
        'psnr': [],
        'ssim': []
    }
    
    print(f'Running inference for images at indices: {target_indices}...')
    current_index = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            print(f'Processing batch {batch_idx + 1}/{len(test_loader)}')
            
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch_data = batch[0].to(device) if isinstance(batch[0], torch.Tensor) else batch[0]
                batch_labels = batch[1] if len(batch) > 1 else None
            else:
                batch_data = batch.to(device)
                batch_labels = None
            
            batch_size = batch_data.shape[0]
            
            # Check if any target indices are in this batch
            batch_start = current_index
            batch_end = current_index + batch_size
            
            # Find which target indices are in this batch
            indices_in_batch = []
            for target_idx in target_indices:
                if batch_start <= target_idx < batch_end:
                    local_idx = target_idx - batch_start
                    indices_in_batch.append(local_idx)
            
            if indices_in_batch:
                # Run inference on the entire batch
                if batch_labels is not None:
                    results = model.test_step((batch_data, batch_labels))
                else:
                    results = model.test_step(batch_data)
                
                # Extract only the results for target indices
                for local_idx in indices_in_batch:
                    global_idx = batch_start + local_idx
                    print(f'Saving results for image index {global_idx}')
                    
                    all_results['originals'].append(results['original'][local_idx:local_idx+1].cpu())
                    all_results['reconstructed'].append(results['reconstructed'][local_idx:local_idx+1].cpu())
                    all_results['measurements'].append(results['transformed'][local_idx:local_idx+1].cpu())
                    all_results['psnr'].append(results['psnr'][local_idx:local_idx+1].cpu())
                    all_results['ssim'].append(results['ssim'][local_idx:local_idx+1].cpu())
            
            current_index += batch_size
            
            # Break early if we've found all target indices
            if current_index > max(target_indices):
                break
    
    # Concatenate all results
    for key in all_results:
        if all_results[key]:
            all_results[key] = torch.cat(all_results[key], dim=0)
        else:
            all_results[key] = torch.tensor([])
    
    print(f'Collected results for {len(all_results["originals"])} images')
    return all_results


def visualize_results(results, target_indices=[11, 15], save_path=None):
    """Visualize inference results"""
    num_examples = len(results['originals'])
    if num_examples == 0:
        print("No results to visualize")
        return
    
    originals = results['originals']
    reconstructed = results['reconstructed']
    measurements = results['measurements']
    psnr_values = results['psnr']
    ssim_values = results['ssim']
    
    fig, axes = plt.subplots(3, num_examples, figsize=(num_examples * 3, 9))
    
    # Handle case where we have only one image
    if num_examples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_examples):
        image_idx = target_indices[i] if i < len(target_indices) else i
        
        # Show original image
        axes[0, i].imshow(originals[i].squeeze())
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original (Index {image_idx})')
        
        # Show reconstructed image
        axes[1, i].imshow(reconstructed[i].squeeze())
        axes[1, i].axis('off')
        axes[1, i].set_title(f"PSNR: {psnr_values[i]:.2f}\nSSIM: {ssim_values[i]:.4f}")
        
        # Show measurement visualization
        axes[2, i].plot(measurements[i].cpu().numpy().flatten()[:100])
        axes[2, i].set_title(f"SPI Measurements (Index {image_idx})")
        axes[2, i].set_xlabel("Measurement Index")
        axes[2, i].grid(True, alpha=0.3)
    
    fig.text(0.04, 0.84, 'Original', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
    fig.text(0.04, 0.5, 'Reconstructed', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
    fig.text(0.04, 0.17, 'Measurements', va='center', ha='left', fontsize=14, weight='bold', rotation=90)
    
    plt.tight_layout(rect=[0.06, 0, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Results saved to: {save_path}')
    
    plt.show()


def compute_metrics(results):
    """Compute and print overall metrics"""
    psnr_values = results['psnr'].numpy()
    ssim_values = results['ssim'].numpy()
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Number of test images: {len(psnr_values)}")
    print(f"Average PSNR: {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f}")
    print(f"Average SSIM: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
    print(f"Max PSNR: {np.max(psnr_values):.4f}")
    print(f"Min PSNR: {np.min(psnr_values):.4f}")
    print(f"Max SSIM: {np.max(ssim_values):.4f}")
    print(f"Min SSIM: {np.min(ssim_values):.4f}")
    print("="*50)


def main():
    torch.manual_seed(42)
    
    parser = init_parser()
    args = parser.parse_args()
    
    print(f"ELU Encoder: {args.elu_encoder}, ELU GAN: {args.elu_gan}")
    print(f"Compression Ratio: {args.cr}, Sensing Matrix: {args.sensing_matrix}")
    
    # Find checkpoint if not provided
    if args.checkpoint_path is None:
        try:
            args.checkpoint_path = find_checkpoint(
                args.results_dir, args.cr, args.sensing_matrix, 
                args.elu_encoder, args.elu_gan
            )
        except FileNotFoundError as e:
            print(f"Error finding checkpoint: {e}")
            return
    
    # Load test dataset
    print('Loading test dataset...')
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    
    test_dataset = MNIST('data', train=False, transform=test_transform, download=True)
    test_loader = data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print('Test dataset loaded!')
    
    # Load model
    try:
        model = load_model(
            args.checkpoint_path, args.cr, args.sensing_matrix,
            args.elu_encoder, args.elu_gan, args.lr, args.patient,
            args.image_size, args.num_channels
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Only process images at indices 11 and 15
    target_indices = [11, 15]
    results = run_inference(model, test_loader, device, target_indices)
    
    # Compute and print metrics
    compute_metrics(results)
    
    # Visualize results
    save_path = None
    # if args.save_results:
    if True:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, 
                                f'inference_results_indices_11_15_cr{args.cr}_sm{args.sensing_matrix}_'
                                f'elu_enc_{args.elu_encoder}_elu_gan_{args.elu_gan}.svg')
    
    visualize_results(results, target_indices, save_path)
    
    # Save numerical results if requested
    # if args.save_results:
    if True:
        results_file = os.path.join(args.output_dir, 
                                   f'numerical_results_indices_11_15_cr{args.cr}_sm{args.sensing_matrix}_'
                                   f'elu_enc_{args.elu_encoder}_elu_gan_{args.elu_gan}.npz')
        np.savez(results_file, 
                indices=target_indices,
                psnr=results['psnr'].numpy(),
                ssim=results['ssim'].numpy(),
                originals=results['originals'].numpy(),
                reconstructed=results['reconstructed'].numpy(),
                measurements=results['measurements'].numpy())
        print(f'Numerical results saved to: {results_file}')


if __name__ == '__main__':
    main()
