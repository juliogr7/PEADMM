# Learned Latent Space Initialization for ADMM with Generative Priors

Computational Imaging (CI) enables the acquisition of low-dimensional or corrupted observations that can be recovered as high-quality images by solving an inverse problem using computational algorithms. The Alternating Direction Method of Multipliers (ADMM) is widely used due to its robustness in handling constraints and regularization techniques. Recent advances in generative models have established them as powerful priors that can be integrated into ADMM to improve image recovery in challenging ill-posed inverse problems. However, ADMM with generative priors typically starts the recovery process by sampling the latent space from a standard normal distribution, particularly in highly underdetermined settings. This random initialization causes the recovery to begin at arbitrary points in the output space of the generator, hindering convergence and reducing the likelihood of reaching optimal solutions. Therefore, a two-stage approach is proposed for learning the latent space of the generative model to improve recovery performance. In the first stage, the latent space is estimated from the observations using an encoder model that leverages a pre-trained generative model. In the second stage, the learned latent space is used to initialize the ADMM algorithm with the generator model serving as the prior. Experimental results show that the proposed method improves PSNR by approximately 2 dB in the most challenging cases, outperforming the ADMM algorithm with standard normal latent space initialization.

## Key Features
- Learned latent space initialization for ADMM with a frozen pretrained generative decoder (GAN) instead of random sampling.
- ADMM modes: baseline (random latent) and proposed (encoder‑initialized latent) for both adversarially perturbed images and Single-Pixel Imaging (SPI) measurements.
- Encoder training: convolutional encoder + frozen generator (decoder) with optional ELU / ReLU activations.
- Support for adversarial corruption and compressive SPI acquisition (linear measurements with selectable sensing matrices).
- Linear single-pixel encoder (LinearSPC) with zig-zag sensing matrices.

## Repository Structure
Root training / evaluation scripts:
- `run_proposed_initialization.py`       : Train encoder (autoencoder) with adversarial MNIST.
- `run_proposed_initialization_spi.py`   : Train encoder for Single-Pixel Imaging (SPI).
- `run_admm.py`                          : Baseline ADMM (random latent) on adversarial images.
- `run_prop_admm.py`                     : ADMM with proposed encoder-based initialization (adversarial images).
- `run_spc_admm.py`                      : Baseline ADMM on SPI measurements.
- `run_prop_spc_admm.py`                 : ADMM with proposed initialization on SPI.

Core models & algorithms:
- `autoencoder.py`              : Convolutional encoder + frozen GAN decoder + Lightning modules (MNIST / SPI).
- `algorithms/gen_admm.py`      : GenADMM core loop (z updates, metrics, visual artifacts).
- `algorithms/optical_encoders.py` : `LinearSPC` single-pixel forward / inverse operators.
- `src/models/`                 : Generator definitions, blocks, loader utilities.
- `utils/callbacks.py`          : Lightning callbacks (epoch logging, step override).
  
## Installation
1. Clone the repository:
	```bash
	git clone https://github.com/juliogr7/PEADMM.git
	cd PEADMM
	```

## Latent Initialization Modes
- Baseline ADMM: initial `z ~ N(0, I)`; reconstruction may start from arbitrary generator manifold region.
- Proposed ADMM: `z0 = Encoder(x0)` where `x0` is an adversarial image or an SPI backprojection / inverse.

## Citation
If this work is useful, please cite (coming soon).

## License
This repository is released under the MIT License (see `LICENSE`).
