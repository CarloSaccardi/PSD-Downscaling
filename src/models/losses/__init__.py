"""
Loss functions for PSD-Downscaling.

This module contains various loss functions used in training.
"""

# Import all loss functions from fourier_losses
from .fourier_losses import (
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
