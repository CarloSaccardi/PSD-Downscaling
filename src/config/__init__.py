"""
Configuration management package for PSD-Downscaling.

This package provides structured configuration management with validation,
type safety, and backward compatibility with existing YAML configurations.
"""

from .base_config import (
    ExperimentConfig,
    DatasetConfig, 
    ModelConfig,
    TrainingConfig,
    ModelType,
    PrecisionType
)
from .legacy_compat import (
    convert_legacy_config_to_new,
    create_legacy_args_from_config
)

__all__ = [
    'ExperimentConfig',
    'DatasetConfig',
    'ModelConfig', 
    'TrainingConfig',
    'ModelType',
    'PrecisionType',
    'convert_legacy_config_to_new',
    'create_legacy_args_from_config'
]
