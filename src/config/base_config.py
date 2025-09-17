"""
Base configuration classes with validation and type safety.

This module provides structured configuration management that maintains
backward compatibility with existing YAML configurations while adding
validation, type safety, and better organization.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import yaml
import torch
from enum import Enum

class ModelType(Enum):
    """Supported model types."""
    UNET_CNN = "UNet-CNN"
    DIFFUSION = "Diffusion"

class PrecisionType(Enum):
    """Supported precision types."""
    FP32 = "32"
    FP16 = "16"
    FP16_MIXED = "16-mixed"
    BF16 = "bf16"

@dataclass
class DatasetConfig:
    """Dataset configuration with validation."""
    cerra_path: str
    era5_path: str
    img_in_channels: int = 5
    img_out_channels: int = 5
    img_resolution: List[int] = field(default_factory=lambda: [300, 300])
    subset_size: Optional[int] = None
    standardize: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate dataset configuration."""
        # Validate paths exist
        if not Path(self.cerra_path).exists():
            raise ValueError(f"CERRA path does not exist: {self.cerra_path}")
        if not Path(self.era5_path).exists():
            raise ValueError(f"ERA5 path does not exist: {self.era5_path}")
        
        # Validate dimensions
        if len(self.img_resolution) != 2:
            raise ValueError("img_resolution must be a list of 2 integers [height, width]")
        if any(dim <= 0 for dim in self.img_resolution):
            raise ValueError("img_resolution dimensions must be positive")
        
        # Validate channels
        if self.img_in_channels <= 0 or self.img_out_channels <= 0:
            raise ValueError("Channel counts must be positive")

@dataclass
class ModelConfig:
    """Model configuration with validation."""
    model_type: str = "UNet-CNN"
    model_channels: int = 64
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 2])
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    embedding_type: str = "zero"
    n_grid_channels: int = 4
    checkpoint_level: int = 0
    
    def __post_init__(self):
        """Validate model configuration."""
        self._validate()
    
    def _validate(self):
        """Validate model configuration."""
        # Validate model type
        try:
            ModelType(self.model_type)
        except ValueError:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Validate embedding type
        valid_embeddings = ["zero", "sinusoidal", "learnable", "linear"]
        if self.embedding_type not in valid_embeddings:
            raise ValueError(f"Invalid embedding type: {self.embedding_type}")
        
        # Validate channel multipliers
        if any(mult <= 0 for mult in self.channel_mult):
            raise ValueError("Channel multipliers must be positive")
        
        # Validate attention resolutions
        if any(res <= 0 for res in self.attn_resolutions):
            raise ValueError("Attention resolutions must be positive")

@dataclass
class TrainingConfig:
    """Training configuration with validation."""
    lr: float = 2e-4
    epochs: int = 200
    batch_size: int = 8
    val_interval: int = 1
    precision: str = "16-mixed"
    n_workers: int = 8
    seed: int = 42
    lr_decay: int = 1
    lr_rampup: int = 0
    grad_clip_threshold: Optional[float] = None
    anneal_epochs: int = 200
    init_lambda: float = 0.0
    max_lambda: float = 0.1
    lambda_psd: float = 0.1
    loss_type: Optional[str] = None
    
    def __post_init__(self):
        """Validate training configuration."""
        self._validate()
    
    def _validate(self):
        """Validate training configuration."""
        # Validate learning rate
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate epochs
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        
        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Validate precision
        try:
            PrecisionType(self.precision)
        except ValueError:
            raise ValueError(f"Unsupported precision: {self.precision}")
        
        # Validate lambda values
        if not 0 <= self.init_lambda <= self.max_lambda:
            raise ValueError("init_lambda must be <= max_lambda and >= 0")

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    
    # Optional settings
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    save_preds_path: Optional[str] = None
    load: Optional[str] = None
    restore_opt: int = 0
    eval: Optional[str] = None
    resume: Optional[str] = None
    regression_net: Optional[str] = None
    gridtype: Optional[str] = None
    hr_mean_conditioning: bool = True
    num_ensembles: int = 32
    output_variables: Optional[List[str]] = None
    model_type: Optional[str] = None  # The actual model class name (e.g., SongUNetPosEmbd)
    
    def __post_init__(self):
        """Validate complete configuration."""
        self._validate()
    
    def _validate(self):
        """Validate experiment configuration."""
        # Validate evaluation mode
        if self.eval is not None and self.eval not in ["val", "test"]:
            raise ValueError("eval must be 'val', 'test', or None")
        
        # Validate checkpoint path if provided
        if self.load and not Path(self.load).exists():
            raise ValueError(f"Checkpoint path does not exist: {self.load}")
        
        # Validate save path if provided
        if self.save_preds_path:
            Path(self.save_preds_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to new structure
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        
        # Create dataset config with proper parameter mapping
        dataset_params = {}
        
        # Map legacy dataset parameters
        if 'dataset_cerra' in config_dict:
            dataset_params['cerra_path'] = config_dict['dataset_cerra']
        if 'dataset_era5' in config_dict:
            dataset_params['era5_path'] = config_dict['dataset_era5']
        if 'subset_ds' in config_dict:
            dataset_params['subset_size'] = config_dict['subset_ds']
        
        # Add other dataset parameters
        for key in ['img_in_channels', 'img_out_channels', 'img_resolution', 'standardize']:
            if key in config_dict:
                dataset_params[key] = config_dict[key]
        
        dataset_config = DatasetConfig(**dataset_params)
        
        # Create model config with proper parameter mapping
        model_params = {}
        
        # Map legacy model parameters
        if 'model' in config_dict:
            model_params['model_type'] = config_dict['model']
        if 'N_grid_channels' in config_dict:
            model_params['n_grid_channels'] = config_dict['N_grid_channels']
        
        # Add other model parameters
        for key in ['model_channels', 'channel_mult', 'attn_resolutions',
                   'embedding_type', 'checkpoint_level']:
            if key in config_dict:
                model_params[key] = config_dict[key]
        
        model_config = ModelConfig(**model_params)
        
        # Create training config
        training_params = {}
        for key in ['lr', 'epochs', 'batch_size', 'val_interval', 'precision', 'n_workers',
                   'seed', 'lr_decay', 'lr_rampup', 'grad_clip_threshold', 'anneal_epochs',
                   'init_lambda', 'max_lambda', 'lambda_psd', 'loss_type']:
            if key in config_dict:
                training_params[key] = config_dict[key]
        
        training_config = TrainingConfig(**training_params)
        
        # Extract remaining settings (excluding all used parameters)
        used_keys = {
            # Dataset keys
            'dataset_cerra', 'dataset_era5', 'subset_ds',
            'img_in_channels', 'img_out_channels', 'img_resolution', 'standardize',
            # Model keys
            'model', 'N_grid_channels', 'model_channels', 'channel_mult', 
            'attn_resolutions', 'embedding_type', 'checkpoint_level',
            # Training keys
            'lr', 'epochs', 'batch_size', 'val_interval', 'precision', 'n_workers',
            'seed', 'lr_decay', 'lr_rampup', 'grad_clip_threshold', 'anneal_epochs',
            'init_lambda', 'max_lambda', 'lambda_psd', 'loss_type'
        }
        
        # Note: model_type is NOT in used_keys so it will be preserved in remaining_config
        
        remaining_config = {
            k: v for k, v in config_dict.items()
            if k not in used_keys
        }
        
        # Map remaining legacy parameters
        if 'savepreds_path' in remaining_config:
            remaining_config['save_preds_path'] = remaining_config.pop('savepreds_path')
        
        return cls(
            dataset=dataset_config,
            model=model_config,
            training=training_config,
            **remaining_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def update_from_args(self, args) -> None:
        """Update configuration from command line arguments."""
        # This method allows updating config from argparse.Namespace
        # while maintaining validation
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        
        # Re-validate after updates
        self.__post_init__()
