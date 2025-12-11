"""
Loss functions for PSD-Downscaling.

This module contains various loss functions used in training.
"""

# Import all loss functions from the original file
from src.models.losses.fourier_losses import (
    FourierLossETH,
    FourierLossDelft, 
    FourierLossHK,
    FourierLossCarlo,
    FourierLossMSEAmpHF
)

__all__ = [
    'FourierLossETH',
    'FourierLossDelft',
    'FourierLossHK', 
    'FourierLossCarlo',
    'FourierLossMSEAmpHF'
]
