# PSD-Downscaling

This repository contains the official implementation of the paper **"Assessing the Geographic Generalization and Physical Consistency of Generative Models for Climate Downscaling"** by Carlo Saccardi et al.

**Authors**: Carlo Saccardi¹'², Maximilian Pierzyna¹, Haitz Sáez de Ocáriz Borde²'³, Simone Monaco⁴, Cristian Meo¹, Pietro Liò², Rudolf Saathof¹, Geethu Joseph¹, Justin Dauwels¹

¹ Delft University of Technology, ² The University of Cambridge, ³ The University of Oxford, ⁴ Politecnico di Torino

> **Note**: You are on the **`CRPS`** branch. This branch contains the implementation of the probabilistic U-Net trained with CRPS loss. For the regression U-Net and CorrDiff models, switch to the **`CNN-UNet`** branch.

## Overview

Kilometer-scale weather data is crucial for real-world applications but remains computationally intensive to produce using traditional weather simulations. This repository provides an emerging deep learning-based solution for climate downscaling, offering a faster alternative to conventional methods.

This codebase implements and benchmarks multiple state-of-the-art deep learning approaches for statistical downscaling of meteorological data from coarse ERA5 resolution (~25 km) to fine CERRA resolution (~5.5 km). A key contribution is the introduction of physics-inspired diagnostics to evaluate model performance, with particular focus on:

- **Geographic Generalization**: How well models trained on one region (e.g., Central Europe) transfer to other regions (e.g., Iberia, Scandinavia)
- **Physical Consistency**: Whether models accurately capture second-order physical variables (divergence, vorticity) and maintain physically meaningful spatial patterns through Power Spectral Density (PSD) preservation

### Key Findings

Our experiments demonstrate that despite strong performance on standard ML metrics, state-of-the-art models like CorrDiff:
- Struggle to generalize geographically when trained on limited regions
- Fail to accurately capture physically-derived second-order variables (divergence, vorticity)
- Show deficiencies in physical consistency even within their training distribution

**Proposed Solution**: We introduce a **Power Spectral Density (PSD) loss function** that empirically improves geographic generalization by encouraging the reconstruction of small-scale physical structures.

### Implemented Methods

1. **UNet-CNN (Regression)**: A deterministic U-Net-based model for direct regression from low-resolution to high-resolution fields (available on `CNN-UNet` branch)
2. **CorrDiff (Diffusion)**: A conditional diffusion model that generates probabilistic downscaled predictions conditioned on the regression model output (based on Song et al.'s approach, available on `CNN-UNet` branch)
3. **CRPS U-Net**: A probabilistic U-Net trained with Continuous Ranked Probability Score (CRPS) loss for ensemble predictions (**this branch**)

### Key Features

- **Physics-Inspired Evaluation**: Includes diagnostics for Power Spectral Density, divergence, and vorticity
- **Multiple Model Architectures**: UNet regression, conditional diffusion (CorrDiff), and CRPS-based probabilistic models
- **PyTorch Lightning**: Modern training framework with built-in support for distributed training
- **Flexible Configuration**: YAML-based config system for easy experimentation
- **Multiple Logging Options**: Support for both Weights & Biases and TensorBoard

## Repository Structure

```
PSD-Downscaling/
├── main.py                 # Main training/evaluation script
├── metrics.py              # Evaluation metrics
├── requirements.txt        # Python dependencies
├── yaml_configs/          # Configuration files
│   ├── UNet/             # UNet regression configs
│   ├── CorrDiff/         # Diffusion model configs
│   └── CRPS/             # CRPS-based UNet configs (for this branch)
├── src/
│   ├── config/           # Configuration system
│   ├── data/             # Dataset implementations
│   │   └── dataset.py    # ERA5-CERRA tiled dataset
│   ├── models/           # Model architectures
│   │   ├── unet.py       # U-Net implementation
│   │   ├── diffusion.py  # Diffusion model
│   │   └── losses/       # Custom loss functions (including CRPS)
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

2. Switch to the CRPS branch:
```bash
git checkout CRPS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Here's a minimal example to get started with the CRPS U-Net:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and preprocess data (see Data section below)
# [Data preprocessing instructions coming soon]

# 3. Train the CRPS U-Net model
python main.py --config yaml_configs/CRPS/CRPS_train.yaml

# 4. Evaluate on test set
python main.py --config yaml_configs/CRPS/CRPS_test.yaml
```

## Data

The models are trained on ERA5 (low-resolution) and CERRA (high-resolution) meteorological reanalysis data.

**Data preprocessing and download instructions**: A separate repository with detailed instructions for downloading and preprocessing the ERA5 and CERRA datasets will be shared soon. Please check back or contact the authors for access.

## Usage

### Training

To train the CRPS U-Net model, run `main.py` with the CRPS configuration file:

```bash
python main.py --config yaml_configs/CRPS/CRPS_train.yaml
```

**Note**: The CRPS model trains a probabilistic U-Net that outputs ensemble predictions. The CRPS (Continuous Ranked Probability Score) loss encourages the model to produce well-calibrated probabilistic forecasts.

### Evaluation

To evaluate a trained model on the test set:

```bash
python main.py --config yaml_configs/CRPS/CRPS_test.yaml
```

### Configuration

All hyperparameters are specified in YAML configuration files located in the `yaml_configs/CRPS/` directory. Key parameters include:

- **Dataset paths**: `dataset_cerra`, `dataset_era5`
- **Model architecture**: `model_type`, `model_channels`, `channel_mult`, `attn_resolutions`
- **Training settings**: `lr`, `epochs`, `batch_size`, `precision`
- **Loss function**: `loss_type` (set for CRPS loss)
- **Evaluation**: Set `eval: "test"` or `eval: "val"` for evaluation mode

Example configuration snippet:
```yaml
# General options
model: UNet-CNN
seed: 42
precision: "16-mixed"
wandb_project: MyProject
loss_type: crps  # CRPS loss for probabilistic training

# Model settings
model_channels: 64
channel_mult: [1, 2, 2]
attn_resolutions: [16]

# Training settings
lr: 2.0e-4
epochs: 300
batch_size: 8
```

## Branch Information

This repository has two main branches:

- **`CNN-UNet`**: For training and evaluating the regression U-Net and CorrDiff models
- **`CRPS`**: For training and evaluating the probabilistic U-Net with CRPS loss (**you are here**)

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

The codebase supports two logging backends:

1. **Weights & Biases (W&B)**: Set `wandb_project` in your config file
2. **TensorBoard**: Used automatically if `wandb_project` is set to `null`

TensorBoard logs are saved to `DebugLogs/`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{saccardi2024assessing,
  title={Assessing the Geographic Generalization and Physical Consistency of Generative Models for Climate Downscaling},
  author={Saccardi, Carlo and Pierzyna, Maximilian and S{\'a}ez de Oc{\'a}riz Borde, Haitz and Monaco, Simone and Meo, Cristian and Li{\`o}, Pietro and Saathof, Rudolf and Joseph, Geethu and Dauwels, Justin},
  year={2024}
}
```

## License

[License information to be added]

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: Carlo Saccardi (Delft University of Technology / University of Cambridge)

## Acknowledgments

This research was conducted at Delft University of Technology in collaboration with the University of Cambridge, University of Oxford, and Politecnico di Torino. We acknowledge the use of ERA5 and CERRA reanalysis data provided by the European Centre for Medium-Range Weather Forecasts (ECMWF) and Copernicus Climate Change Service.
