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
- `run_ganinv_train.py` / `run_ganinv_test.py`                        : Standard GAN inversion train / test.
- `run_ganinv_train_adversarial.py` / `run_ganinv_test_adversarial.py`: GAN inversion under adversarial corruption.
- `run_ganinv_train_spi.py` / `run_ganinv_test_spi.py`                : GAN inversion for SPI.
- `build_adversarial_*.py`                                           : Build & save adversarial train/valid/test datasets.
- `build_spi_*.py`                                                   : Build SPI measurement datasets.

Core models & algorithms:
- `autoencoder.py`              : Convolutional encoder + frozen GAN decoder + Lightning modules (MNIST / SPI).
- `algorithms/gen_admm.py`      : GenADMM core loop (z updates, metrics, visual artifacts).
- `algorithms/optical_encoders.py` : `LinearSPC` single-pixel forward / inverse operators.
- `src/models/`                 : Generator definitions, blocks, loader utilities.
- `utils/callbacks.py`          : Lightning callbacks (epoch logging, step override).
- `utils/exp_setting.py`        : Experiment directory + reproducible command capture.

Utilities / supporting code:
- `src/util/attack.py`, `attack_2.py` : Adversarial perturbation generation / dataloader wrapping.
- `src/util/evaluation.py`            : Evaluation helpers (if present).
- `src/functions.py`, `src/proj_l1.py`, `src/augmented_lagrangian.py` : Mathematical operators / optimization pieces for ADMM.
- `algorithms/libs/ordering/*.py`     : Sensing matrix constructions (zig_zag, hadamard, cake_cutting, etc.).

Results layout (examples):
- `results/pretrained_adversarial/...` : Pretrained autoencoder checkpoints (different noise std / activations).
- `results/prop_init_spi/...`          : SPI proposed-init experiments (various compression ratios & matrices).
- `results/<experiment>/<name>/checkpoints/` : Saved weights (`epoch=...step=...ckpt`).
- `results/<experiment>/<name>/csv/`          : Tabular metric logs.
- `results/<experiment>/<name>/model_info.txt`: Reproducible command line.|

## Installation
1. Clone the repository:
	```bash
	git clone https://github.com/juliogr7/PEADMM.git
	cd PEADMM
	```

## Main Arguments (Summary)
Common parameters (see each script for full list):
- `--std`                : Noise std / adversarial severity.
- `--z-dim`              : Latent dimensionality.
- `--gamma`, `--beta`, `--sigma` : ADMM hyperparameters.
- `--max-iter`           : ADMM iterations (encoder training uses epochs separately).
- `--elu_encoder`, `--elu_gan` : Activation choices in encoder / decoder.
- `--cr`                 : Compression ratio (SPI).
- `--sensing-matrix`     : Sensing matrix type (zig_zag, cake_cutting, random_binary).
- `--n-images`           : Number of images processed in ADMM experiments.
- `--batch-size`         : Mini‑batch size (vectorized latent updates).

## Latent Initialization Modes
- Baseline ADMM: initial `z ~ N(0, I)`; reconstruction may start from arbitrary generator manifold region.
- Proposed ADMM: `z0 = Encoder(x0)` where `x0` is an adversarial image or an SPI backprojection / inverse.

## Citation
If this work is useful, please cite (coming soon).

## License
This repository is released under the MIT License (see `LICENSE`).