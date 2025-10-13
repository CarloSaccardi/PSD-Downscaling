
> **Note**: You are on the **`CRPS`** branch. This branch contains the implementation of the probabilistic U-Net trained with CRPS loss. For the regression U-Net and CorrDiff models, switch to the **`CNN-UNet`** branch.

# PSD-Downscaling

This repository contains the official implementation of the paper **"Assessing the Geographic Generalization and Physical Consistency of Generative Models for Climate Downscaling"**, accepted at the AI for Science workshop at NeurIPS 2025.

## Overview

This codebase implements multiple deep learning approaches for statistical downscaling of meteorological data, with a focus on preserving Power Spectral Density (PSD) characteristics. The models are designed to downscale low-resolution ERA5 data to high-resolution CERRA data while maintaining physically meaningful spatial patterns.

### Implemented Methods

1. **UNet-CNN (Regression)**: A deterministic U-Net-based model for direct regression from low-resolution to high-resolution fields. This is the deterministic component of CorrDiff.
3. **CRPS U-Net**: The same U-Net architecture but trained with Continuous Ranked Probability Score (CRPS) loss to make it probabilistic. 
2. **CorrDiff (Diffusion)**: A conditional diffusion model that generates probabilistic downscaled predictions conditioned on the regression model output (UNet-CNN).

## Repository Structure

```
PSD-Downscaling/
├── main.py                 # Main training/evaluation script
├── metrics.py              # Evaluation metrics
├── requirements.txt        # Python dependencies
├── yaml_configs/          # Configuration files
│   ├── UNet/             # UNet regression configs
│   ├── CorrDiff/         # Diffusion model configs
│   └── CRPS/             # CRPS-based UNet configs
├── src/
│   ├── config/           # Configuration system
│   ├── data/             # Dataset implementations
│   │   └── dataset.py    # ERA5-CERRA tiled dataset
│   ├── models/           # Model architectures
│   │   ├── unet.py       # U-Net implementation
│   │   ├── diffusion.py  # Diffusion model
│   │   └── losses/       # Custom loss functions
│   ├── training/         # Training utilities
│   └── utils/            # Utility functions
└── saved_models/         # Model checkpoints (created during training)
```

## Installation

### Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- PyTorch Lightning 2.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/[username]/PSD-Downscaling.git
cd PSD-Downscaling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

The models are trained on ERA5 (low-resolution) and CERRA (high-resolution) meteorological reanalysis data.

**Data preprocessing and download instructions**: A separate repository with detailed instructions for downloading and preprocessing the ERA5 and CERRA datasets will be shared soon. Please check back or contact the authors for access.

## Usage

### Training

To train a model, run `main.py` with a configuration file:

```bash
python main.py --config yaml_configs/[model_type]/[config_file].yaml
```

#### Training Examples

**Train UNet-CNN (Regression Model):**
```bash
python main.py --config yaml_configs/UNet/UNet_train.yaml
```

**Train CorrDiff (Diffusion Model):**
```bash
python main.py --config yaml_configs/CorrDiff/CorrDiff_train.yaml
```

**Train CRPS U-Net:**
```bash
python main.py --config yaml_configs/CRPS/CRPS_train.yaml
```

### Evaluation

To evaluate a trained model on the test set:

```bash
python main.py --config yaml_configs/[model_type]/[config_file]_test.yaml
```

**Example:**
```bash
python main.py --config yaml_configs/UNet/UNet_test.yaml
```

### Configuration

All hyperparameters are specified in YAML configuration files located in the `yaml_configs/` directory. Key parameters include:

- **Dataset paths**: `dataset_cerra`, `dataset_era5`
- **Model architecture**: `model_type`, `model_channels`, `channel_mult`, `attn_resolutions`
- **Training settings**: `lr`, `epochs`, `batch_size`, `precision`
- **Evaluation**: Set `eval: "test"` or `eval: "val"` for evaluation mode

Example configuration snippet:
```yaml
# General options
model: UNet-CNN
seed: 42
precision: "16-mixed"
wandb_project: MyProject

# Model settings
model_channels: 64
channel_mult: [1, 2, 2]
attn_resolutions: [16]

# Training settings
lr: 2.0e-4
epochs: 200
batch_size: 8
```

## Branch Information

This repository has two main branches:

- **`CNN-UNet`**: For training and evaluating the regression U-Net and CorrDiff models
- **`CRPS`**: For training and evaluating the probabilistic U-Net with CRPS loss

Switch to the appropriate branch depending on which model you want to work with:
```bash
git checkout CNN-UNet      # For UNet and CorrDiff
git checkout CRPS          # For CRPS-based probabilistic UNet
```

## Model Checkpoints

Trained model checkpoints are saved in the `saved_models/` directory. Each training run creates a timestamped subdirectory containing:
- `last.ckpt`: Most recent checkpoint
- `min_val_loss.ckpt`: Best model based on validation loss

To resume training or run evaluation, specify the checkpoint path in your config file:
```yaml
load: "saved_models/[run_name]/min_val_loss.ckpt"
```

## Logging

The codebase supports logging with Weights & Biases:

1. **Weights & Biases (W&B)**: Set `wandb_project` in your config file


## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: c.saccardi@tudelft.nl




