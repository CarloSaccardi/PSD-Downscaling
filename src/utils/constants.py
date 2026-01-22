# Third-party
import cartopy
import numpy as np

PARAM_NAMES_SHORT_CERRA = [
    "u_wind",
    "v_wind",
    "t2m",
    "sshf",
    "zust",
    "sp",
    "geopotential"
]

PARAM_UNITS_CERRA = [
    "Pa",
    "Pa",
    "K",
    "W/m²",
    "m/s", 
    "Pa/s",
    "m²/s²",
]

GRID_SHAPE_CERRA = (384, 384)  # (y, x)
GRID_SHAPE_ERA5 = (85, 85)  # (y, x)
GRID_SHAPE_CROPPED = (96, 96)  # (y, x)