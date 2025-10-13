"""
Backward compatibility layer for configuration management.

This module ensures that existing YAML configs and command line arguments
continue to work with the new configuration system.
"""

from typing import Dict, Any
from .base_config import ExperimentConfig

def convert_legacy_config_to_new(legacy_args) -> ExperimentConfig:
    """
    Convert legacy argument parser namespace to new configuration format.
    
    This function ensures that existing YAML configs and command line arguments
    continue to work with the new configuration system.
    """
    # Convert legacy args to flat dictionary first
    config_dict = {}
    
    # Copy all attributes from legacy args, excluding system arguments
    exclude_args = {'config', 'use_new_config', 'validate_config'}
    
    for key, value in vars(legacy_args).items():
        if value is not None and key not in exclude_args:
            config_dict[key] = value
    
    # Use the existing from_dict method which handles the mapping
    return ExperimentConfig.from_dict(config_dict)

def create_legacy_args_from_config(config: ExperimentConfig):
    """
    Create legacy argument namespace from new configuration.
    
    This allows the new configuration system to work with legacy code
    that expects argparse.Namespace objects.
    """
    class LegacyArgs:
        def __init__(self, config: ExperimentConfig):
            # Dataset args
            self.dataset_cerra = config.dataset.cerra_path
            self.dataset_era5 = config.dataset.era5_path
            self.img_in_channels = config.dataset.img_in_channels
            self.img_out_channels = config.dataset.img_out_channels
            self.img_resolution = config.dataset.img_resolution
            self.subset_ds = config.dataset.subset_size or 0
            
            # Model args
            self.model = config.model.model_type
            # model_type should be the actual class name from the YAML file
            # This comes from the top-level model_type field in the YAML
            self.model_type = getattr(config, 'model_type', None)
            self.model_channels = config.model.model_channels
            self.channel_mult = config.model.channel_mult
            self.attn_resolutions = config.model.attn_resolutions
            self.embedding_type = config.model.embedding_type
            self.N_grid_channels = config.model.n_grid_channels
            self.checkpoint_level = config.model.checkpoint_level
            
            # Training args
            self.lr = config.training.lr
            self.epochs = config.training.epochs
            self.batch_size = config.training.batch_size
            self.val_interval = config.training.val_interval
            self.precision = config.training.precision
            self.n_workers = config.training.n_workers
            self.seed = config.training.seed
            self.lr_decay = config.training.lr_decay
            self.lr_rampup = config.training.lr_rampup
            self.grad_clip_threshold = config.training.grad_clip_threshold
            self.anneal_epochs = config.training.anneal_epochs
            self.init_lambda = config.training.init_lambda
            self.max_lambda = config.training.max_lambda
            self.lambda_psd = config.training.lambda_psd
            self.loss_type = config.training.loss_type
            
            # Other args
            self.wandb_project = config.wandb_project
            self.run_name = config.run_name
            self.savepreds_path = config.save_preds_path
            self.save_preds_path = config.save_preds_path  # Add both versions
            self.load = config.load
            self.restore_opt = config.restore_opt
            self.eval = config.eval
            self.resume = config.resume
            self.regression_net = config.regression_net
            self.gridtype = config.gridtype
            self.hr_mean_conditioning = config.hr_mean_conditioning
            self.num_ensembles = config.num_ensembles
            self.output_variables = config.output_variables
    
    return LegacyArgs(config)
