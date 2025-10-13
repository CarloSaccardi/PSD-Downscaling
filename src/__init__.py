"""
PSD-Downscaling Package

This package provides both new modular structure and backward compatibility.
"""

# Import constants and utils directly for easy access
from . import constants
from . import utils as utils_module

# Make utils functions available at package level
utils = utils_module

# Import submodules
from . import config
from . import models
from . import data
from . import training

# Re-export commonly used classes
from .models import UNetWrapper, DiffusionWrapper
from .data import ERA5toCERRA2, ERA5tCERRAStats

__all__ = [
    "constants",
    "utils",
    "config",
    "models",
    "data",
    "training",
    "UNetWrapper",
    "DiffusionWrapper",
    "ERA5toCERRA2",
    "ERA5tCERRAStats",
]
