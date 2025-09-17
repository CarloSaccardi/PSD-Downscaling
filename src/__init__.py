"""
PSD-Downscaling Package

This package provides both new modular structure and backward compatibility.
"""

# Import new modules
from .config import *
from .models import *
from .data import *
from .training import *
from .utils import *

# Backward compatibility imports
import sys
from pathlib import Path

# Add legacy paths to sys.path for backward compatibility
legacy_path = Path(__file__).parent.parent / "neural_lam"
if legacy_path.exists():
    sys.path.insert(0, str(legacy_path))

# Legacy imports for backward compatibility
try:
    from neural_lam.models.unet import UNetWrapper as LegacyUNetWrapper
    from neural_lam.weather_dataset import ERA5toCERRA2 as LegacyERA5toCERRA2
    from neural_lam import utils as legacy_utils
    from neural_lam import constants as legacy_constants
    from neural_lam import vis as legacy_vis
except ImportError:
    # Handle case where legacy modules don't exist
    pass
