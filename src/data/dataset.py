# Standard library
import datetime as dt
import glob
import os

# Third-party
import numpy as np
import torch

# First-party
from src import constants, utils

import torch.nn.functional as F

import re
from pathlib import Path
from typing import List, Tuple

import xarray as xr
import pandas as pd


class ERA5tCERRAStats(torch.utils.data.Dataset):
    """
    Statistics-only wrapper: each item is a pair
        (cerra, era5) with shape (C, H*W*T)
    so that downstream code can just take a mean over dim=-1.
    """

    _file_pat = re.compile(r"^nwp_(?P<stem>.+?)\.npy$")

    def __init__(
        self,
        root_cerra: str | Path,
        root_era5: str | Path,
        split: str = "train",
        subset: int | None = None,          # <= None → full data
    ):
        super().__init__()
        assert split in {"train", "val", "test"}

        self.root_cerra = Path(root_cerra).expanduser()
        self.root_era5  = Path(root_era5 ).expanduser()

        self.dir_cerra = self.root_cerra / "samples" / split
        self.dir_era5  = self.root_era5  / "samples" / split

        # ---- Build the canonical list from CERRA and verify ERA5 exists
        self.stems: List[str] = sorted(
            m.group("stem")
            for m in map(lambda p: self._file_pat.match(p.name),  # ← p.name is str
                        self.dir_cerra.iterdir())
            if m is not None
        )

        missing: List[str] = [
            s for s in self.stems if not (self.dir_era5 / f"nwp_{s}.npy").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} ERA5 files not found, e.g. {missing[:3]}"
            )

        if subset is not None:
            self.stems = self.stems[: subset]

    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.stems)

    # --------------------------------------------------------------------- #
    def _load_numpy(self, path: Path) -> torch.Tensor:
        """
        Memory-maps a .npy file and returns a float32 tensor with shape
        (C, T*H*W).  No copy if dtype already float32.
        """
        arr = np.load(path, mmap_mode="r")                  # shape (T, H, W, C′)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)                    # one cheap copy
        # Move channel to front and collapse the rest
        arr = np.moveaxis(arr, -1, 0).reshape(arr.shape[-1], -1)
        return torch.from_numpy(arr)

    # --------------------------------------------------------------------- #
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        cerra = self._load_numpy(self.dir_cerra / f"nwp_{stem}.npy")
        era5  = self._load_numpy(self.dir_era5  / f"nwp_{stem}.npy")
        return cerra, era5
    
    
    
class ERA5toCERRA2(torch.utils.data.Dataset):
    """
    For our dataset:
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    dim_x = 268
    dim_y = 238
    N_grid = 268x238 = 63784
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    """
    
    def __init__(
        self,
        dataset_name_CERRA,
        dataset_name_ERA5,
        split,
        standardize=True,
        subset=False,
    ):
        super().__init__()
        
        assert split in ("train", "val", "test"), "Unknown dataset split"
        
        # Determine mode based on which dataset names are provided.
        if dataset_name_CERRA is None and dataset_name_ERA5 is None:
            raise ValueError("At least one dataset must be provided.")
        elif dataset_name_CERRA is not None and dataset_name_ERA5 is not None:
            self.mode = "both"
        elif dataset_name_CERRA is not None:
            self.mode = "CERRA_only"
        else:
            self.mode = "ERA5_only"
        
        member_file_regexp = "*.npy"
        
        self.split = split
        
        # Load CERRA dataset if available.
        if self.mode in ("both", "CERRA_only"):
            self.sample_dir_path_CERRA = os.path.join("data", dataset_name_CERRA, "samples", split)
            sample_paths_CERRA = glob.glob(os.path.join(self.sample_dir_path_CERRA, member_file_regexp))
            self.sample_names_CERRA = sorted([os.path.basename(path)[4:-4] for path in sample_paths_CERRA])
        
        # Load ERA5 dataset if available.
        if self.mode in ("both", "ERA5_only"):
            self.sample_dir_path_era5 = os.path.join("data", dataset_name_ERA5, "samples", split)
            sample_paths_era5 = glob.glob(os.path.join(self.sample_dir_path_era5, member_file_regexp))
            self.sample_names_era5 = sorted([os.path.basename(path)[4:-4] for path in sample_paths_era5])
        
        # Optionally restrict to a subset of samples.
        if subset:
            if self.mode in ("both", "CERRA_only"):
                self.sample_names_CERRA = self.sample_names_CERRA[:50]
            if self.mode in ("both", "ERA5_only"):
                self.sample_names_era5 = self.sample_names_era5[:50]
        
        # Set up standardization if requested.
        self.standardize = standardize
        if standardize:
            if self.mode in ("both", "CERRA_only"):
                ds_stats_CERRA = utils.load_dataset_stats(dataset_name_CERRA, "cpu")
                self.data_mean_CERRA, self.data_std_CERRA = ds_stats_CERRA["data_mean"], ds_stats_CERRA["data_std"]
            if self.mode in ("both", "ERA5_only"):
                ds_stats_era5 = utils.load_dataset_stats(dataset_name_ERA5, "cpu")
                self.data_mean_era5, self.data_std_era5 = ds_stats_era5["data_mean"], ds_stats_era5["data_std"]
        
        # If subsampling should occur (only during training)
        self.random_subsample = (split == "train")
    
    def __len__(self):
        if self.mode == "both":
            assert len(self.sample_names_CERRA) == len(self.sample_names_era5), "Different number of samples in CERRA and ERA5"
            return len(self.sample_names_CERRA)
        elif self.mode == "CERRA_only":
            return len(self.sample_names_CERRA)
        else:  # ERA5_only
            return len(self.sample_names_era5)
    
    def __getitem__(self, idx):
        if self.mode == "both":
            sample_name_CERRA = self.sample_names_CERRA[idx]
            sample_name_era5 = self.sample_names_era5[idx]
            sample_path_CERRA = os.path.join(self.sample_dir_path_CERRA, f"nwp_{sample_name_CERRA}.npy")
            sample_path_era5 = os.path.join(self.sample_dir_path_era5, f"nwp_{sample_name_era5}.npy")
            try:
                sample_CERRA = torch.tensor(np.load(sample_path_CERRA), dtype=torch.float32)
                sample_era5 = torch.tensor(np.load(sample_path_era5), dtype=torch.float32)
            except ValueError:
                print(f"Failed to load {sample_path_CERRA}")
                print(f"Failed to load {sample_path_era5}")
            # Flatten spatial dimensions.
            sample_CERRA = sample_CERRA.permute(2, 0, 1)
            sample_era5 = sample_era5.permute(2, 0, 1)
            
            sample_era5 = self.upsample(sample_era5, sample_CERRA)
            
            if self.standardize:
                sample_CERRA = (sample_CERRA - self.data_mean_CERRA[:, None, None]) / self.data_std_CERRA[:, None, None]
                sample_era5 = (sample_era5 - self.data_mean_era5[:, None, None]) / self.data_std_era5[:, None, None]
                
            if self.split == "test":
                mean_CERRA = self.data_mean_CERRA[:, None, None]
                std_CERRA = self.data_std_CERRA[:, None, None]
                mean_era5 = self.data_mean_era5[:, None, None]
                std_era5 = self.data_std_era5[:, None, None]
                diz_stats = {
                    "mean_CERRA": mean_CERRA,
                    "std_CERRA": std_CERRA,
                    "mean_era5": mean_era5,
                    "std_era5": std_era5
                }
                #return also the names of the samples
                
                return sample_CERRA, sample_era5, diz_stats, sample_name_CERRA
            
            else:
                return sample_CERRA, sample_era5
        
        elif self.mode == "CERRA_only":
            sample_name_CERRA = self.sample_names_CERRA[idx]
            sample_path_CERRA = os.path.join(self.sample_dir_path_CERRA, f"nwp_{sample_name_CERRA}.npy")
            try:
                sample_CERRA = torch.tensor(np.load(sample_path_CERRA), dtype=torch.float32)
            except ValueError:
                print(f"Failed to load {sample_path_CERRA}")
            sample_CERRA = sample_CERRA.flatten(0, 1)
            if self.standardize:
                sample_CERRA = (sample_CERRA - self.data_mean_CERRA) / self.data_std_CERRA
            return sample_CERRA
        
        else:  # ERA5_only
            sample_name_era5 = self.sample_names_era5[idx]
            sample_path_era5 = os.path.join(self.sample_dir_path_era5, f"nwp_{sample_name_era5}.npy")
            try:
                sample_era5 = torch.tensor(np.load(sample_path_era5), dtype=torch.float32)
            except ValueError:
                print(f"Failed to load {sample_path_era5}")
            sample_era5 = sample_era5.flatten(0, 1)
            if self.standardize:
                sample_era5 = (sample_era5 - self.data_mean_era5) / self.data_std_era5
            return sample_era5
        
        
    def upsample(self, lr_tensor, hr_tensor):
        """
        Upsample the input tensor to match the target tensor's spatial dimensions.
        """
        # 1) add batch dim
        era5_batched = lr_tensor.unsqueeze(0)                # [1, C, H_old, W_old]

        # 2) pick the target spatial size from sample_CERRA
        target_size = hr_tensor.shape[-2:]                  # (H_new, W_new)

        # 3) interpolate
        upsampled = F.interpolate(
            era5_batched,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )                                                      # [1, C, H_new, W_new]

        # 4) drop the batch dim
        return upsampled.squeeze(0)                      # [C, H_new, W_new]
    
    
    

class Era5CropDataset(torch.utils.data.Dataset):
    """
    Refactored custom PyTorch Dataset for ERA5 data.
    - Uses helper functions in __getitem__ for clarity.
    - Loads static data into RAM in __init__ for speed.
    """
    def __init__(self, 
                 path, 
                 split,
                 variables=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp'], 
                 crop_size=85):
        super().__init__()
        
        stats_dir = os.path.join(path, "statistics")
        dynamic_f = os.path.join(path, split, "Eurasia.nc")
        forcing_f = os.path.join(path, split, "static_Eurasia.nc")
        
        # 1. Load statistics
        self.mean_dynamic_vars = np.load(f"{stats_dir}/forcing_mean.npy")
        self.std_dynamic_vars = np.load(f"{stats_dir}/forcing_std.npy")
        self.mean_static_var = np.load(f"{stats_dir}/dynamic_mean.npy")
        self.std_static_var = np.load(f"{stats_dir}/dynamic_std.npy")
        
        # 2. Open datasets
        self.dynamic_f = xr.open_dataset(dynamic_f) #, chunks={'time': 1})
        self.forcing_f = xr.open_dataset(forcing_f)
        
        # 3. Load static data into RAM (This is a key optimization)
        self.geopotential_data = self.forcing_f['geopotential'].values.astype(np.float32)
        
        # 4. Load time axis into RAM (Good for performance)
        self.time_axis = self.dynamic_f.time.values
        
        self.variables = variables
        self.crop_size = crop_size
        
        # 5. Get dimensions
        self.time_len = len(self.dynamic_f.time)
        self.lat_len = len(self.dynamic_f.latitude)
        self.lon_len = len(self.dynamic_f.longitude)
        
        # 6. Pre-calculate max indices for random cropping
        self.max_lat_idx = self.lat_len - self.crop_size
        self.max_lon_idx = self.lon_len - self.crop_size
        
        print(f"Dataset initialized:")
        print(f"  Time steps: {self.time_len}")
        print(f"  Total Channels: {len(self.variables) + 1 + 4}") # dynamic + static + time

    def __len__(self):
        return self.time_len

    def __getitem__(self, idx):
        """
        Fetches one item: a random 81x81 crop from time step `idx`.
        This is now a clean wrapper around helper functions.
        """
        
        # 1. Get random crop start indices
        lat_idx, lon_idx = self._get_random_crop_indices()
        
        # 2. Get 4 time embedding features
        time_features = self._get_time_embedding(idx)
        
        # 3. Load/crop all data channels
        crop_dynamics = self._load_crop_dynamic(idx, lat_idx, lon_idx)
        crop_static = self._crop_static(lat_idx, lon_idx)
        time_channels = self._broadcast_time_features(time_features)
        
        # 4. Normalize
        crop_dynamics = (crop_dynamics - self.mean_dynamic_vars[:, None, None]) / self.std_dynamic_vars[:, None, None]
        crop_static = (crop_static - self.mean_static_var) / self.std_static_var
        
        # 5. Concatenate and return
        all_features = np.concatenate([crop_dynamics, crop_static, time_channels], axis=0)
        
        # 6. Convert to torch tensor and upsample
        all_features_tensor = torch.from_numpy(all_features.copy()).float()
        tensor = self._upsample(all_features_tensor, hr_tensor=(384, 384))
        
        return tensor

    # --- Helper Functions ---

    def _get_random_crop_indices(self):
        """Returns random starting indices for a latitude and longitude."""
        lat_idx = np.random.randint(0, self.max_lat_idx + 1)
        lon_idx = np.random.randint(0, self.max_lon_idx + 1)
        return lat_idx, lon_idx

    def _get_time_embedding(self, idx):
        """Computes the 4 time embedding features for a given time index."""
        datetime = self.time_axis[idx]
        dt_obj = pd.to_datetime(datetime)
        day_of_year = dt_obj.dayofyear
        hour = dt_obj.hour
        
        day_norm = 2 * np.pi * day_of_year / 365.25
        hour_norm = 2 * np.pi * hour / 24.0

        day_sin = (np.sin(day_norm) + 1) / 2
        day_cos = (np.cos(day_norm) + 1) / 2
        hour_sin = (np.sin(hour_norm) + 1) / 2
        hour_cos = (np.cos(hour_norm) + 1) / 2
        
        return [day_sin, day_cos, hour_sin, hour_cos]

    def _load_crop_dynamic(self, idx, lat_idx, lon_idx):
            """
            Loads ONLY the cropped data from disk using lazy xarray slicing.
            """
            
            # 1. Define the spatial slices (this is just metadata)
            lat_slice = slice(lat_idx, lat_idx + self.crop_size)
            lon_slice = slice(lon_idx, lon_idx + self.crop_size)

            # 2. Chain all selections (variables, time, and space)
            #    This is all LAZY. No data is read from disk yet.
            data_crop = self.dynamic_f[self.variables].isel(
                time=idx,
                latitude=lat_slice,
                longitude=lon_slice
            )
            
            # 3. Convert the multi-variable Dataset into a single DataArray
            #    This stacks variables along a new 'variable' dimension.
            #    Still lazy.
            data_array = data_crop.to_array()

            # 4. NOW, call .values.
            #    This executes the read, pulling ONLY the [n_vars, 85, 85]
            #    block of data from the NetCDF file.
            return data_array.values

    def _crop_static(self, lat_idx, lon_idx):
        """Crops the static data (already in RAM)."""
        static_crop = self.geopotential_data[
            lat_idx : lat_idx + self.crop_size,
            lon_idx : lon_idx + self.crop_size
        ]
        # Add a channel dimension
        return static_crop[None, :, :]

    def _broadcast_time_features(self, time_features):
        """Broadcasts the 4 time features to (4, H, W) channels."""
        time_channels = np.zeros((4, self.crop_size, self.crop_size), dtype=np.float32)
        for i, val in enumerate(time_features):
            time_channels[i, :, :] = val
        return time_channels
    
    def _upsample(self, lr_tensor, hr_tensor=(384, 384)):
        """
        Upsample the input tensor to match the target tensor's spatial dimensions.
        """
        # 1) add batch dim
        era5_batched = lr_tensor.unsqueeze(0)                # [1, C, H_old, W_old]
        # 2) pick the target spatial size from sample_CERRA
        target_size = hr_tensor                  # (H_new, W_new)
        # 3) interpolate
        upsampled = F.interpolate(
            era5_batched,
            size=target_size,
            mode='bicubic',
            align_corners=False
        )                                                      # [1, C, H_new, W_new]
        # 4) drop the batch dim
        return upsampled.squeeze(0)                      # [C, H_new, W_new]

    def close(self):
        """Closes the xarray dataset file handle."""
        if self.dynamic_f:
            self.dynamic_f.close()
        if self.forcing_f:
            self.forcing_f.close()
            
