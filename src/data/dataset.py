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

import xarray as xr
import pandas as pd


class CerraEra5SuperResDataset(torch.utils.data.Dataset):
    """
    Super-Resolution Dataset (Upsampled ERA5 + CERRA Forcing).
    
    Logic:
    1. Loads Low-Res ERA5 (85x85) and Time Embeddings.
    2. Upsamples ERA5+Time to High-Res (384x384) using Bicubic interpolation.
    3. Concatenates with High-Res CERRA Forcing (Orography).
    4. Returns (Full_Input, CERRA_Target).
    """
    def __init__(self, 
                 root_dir_cerra,
                 root_dir_era5,
                 split,
                 region,
                 target_size=(384, 384), # Target HR dimension
                 era5_vars=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp'], 
                 cerra_vars=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp']):
        super().__init__()
        
        # Paths
        era5_dyn_path = os.path.join(root_dir_era5, split, f"{region}.nc")
        cerra_dyn_path = os.path.join(root_dir_cerra, split, f"{region}.nc")
        cerra_stat_path = os.path.join(root_dir_cerra, split, f"static_{region}.nc")
        
        # 1. Load Statistics
        self.era5_mean = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_mean.npy"))
        self.era5_std = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_std.npy"))
        
        self.cerra_dyn_mean = np.load(os.path.join(root_dir_cerra, "statistics", "dynamic_mean.npy"))
        self.cerra_dyn_std = np.load(os.path.join(root_dir_cerra, "statistics", "dynamic_std.npy"))
        
        self.cerra_stat_mean = np.load(os.path.join(root_dir_cerra, "statistics", "forcing_mean.npy"))
        self.cerra_stat_std = np.load(os.path.join(root_dir_cerra, "statistics", "forcing_std.npy"))

        # 2. Open Datasets (Lazy Xarray)
        self.era5_dyn_ds = xr.open_dataset(era5_dyn_path, engine="h5netcdf") 
        self.cerra_dyn_ds = xr.open_dataset(cerra_dyn_path, engine="h5netcdf")
        
        # 3. Load Static Data into RAM (Optimization)
        # We perform the static normalization ONCE here to save CPU cycles in __getitem__
        ds_static = xr.open_dataset(cerra_stat_path, engine="h5netcdf")
        raw_static = ds_static['orog'].values.astype(np.float32)
        self.cerra_orography = (raw_static - self.cerra_stat_mean) / self.cerra_stat_std
        ds_static.close()
        
        # 4. Load Time Axis
        self.time_axis = self.era5_dyn_ds.time.values
        self.time_len = len(self.time_axis)
        
        self.era5_vars = era5_vars
        self.cerra_vars = cerra_vars
        self.target_size = target_size


    def __len__(self):
        return self.time_len

    def __getitem__(self, idx):
        # 1. Get Time Embeddings
        time_feats = self._get_time_embedding(idx)
        
        # 2. Load Dynamic Data (Lazy Read)
        # ERA5: [C_era5, 85, 85]
        era5_data = self._load_dynamic_step(self.era5_dyn_ds, self.era5_vars, idx)
        # CERRA Target: [C_cerra, 384, 384]
        cerra_target = self._load_dynamic_step(self.cerra_dyn_ds, self.cerra_vars, idx)
        
        # 3. Normalize Dynamic Data
        era5_data = (era5_data - self.era5_mean[:, None, None]) / self.era5_std[:, None, None]
        cerra_target = (cerra_target - self.cerra_dyn_mean[:, None, None]) / self.cerra_dyn_std[:, None, None]
        
        # 4. Broadcast Time Features to ERA5 (Low Res) and CERRA (High Res)
        # [4, 85, 85] for ERA5
        time_channels_lr = self._broadcast_time_features(time_feats, era5_data.shape[1], era5_data.shape[2])
        
        # 5. Concatenate ERA5 + Time (Low Res)
        lr_combined = np.concatenate([era5_data, time_channels_lr], axis=0)
        
        # 6. Convert to Tensor for Interpolation
        lr_tensor = torch.from_numpy(lr_combined).float()
        
        # 7. --- UPSAMPLING (The key step) ---
        # Interpolate requires [Batch, Channels, H, W], so we unsqueeze(0)
        # Result: [1, C, 384, 384]
        hr_upsampled = F.interpolate(
            lr_tensor.unsqueeze(0), 
            size=self.target_size, 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0) # Remove batch dim -> [C, 384, 384]
        
        # 8. Prepare Static Forcing (Already 384x384 and Normalized in __init__)
        # [1, 384, 384]
        hr_forcing = torch.from_numpy(self.cerra_orography[None, :, :]).float().squeeze(0) #remove first dimension
        
        # 9. Concatenate Upsampled Input + Static Forcing
        # Input: [C_era5 + 4 + 1, 384, 384]
        full_input = torch.cat([hr_upsampled, hr_forcing], dim=0)
        
        return cerra_target, full_input

    # --- Helper Functions (Same as before) ---

    def _load_dynamic_step(self, dataset, variables, idx):
        """Lazy load specific time step."""
        data_sel = dataset[variables].isel(time=idx)
        return data_sel.to_array().values.astype(np.float32)

    def _get_time_embedding(self, idx):
        """Computes 4 time embedding features."""
        datetime = self.time_axis[idx]
        dt_obj = pd.to_datetime(datetime)
        day_of_year = dt_obj.dayofyear
        hour = dt_obj.hour
        
        day_norm = 2 * np.pi * day_of_year / 365.25
        hour_norm = 2 * np.pi * hour / 24.0

        return [
            (np.sin(day_norm) + 1) / 2,
            (np.cos(day_norm) + 1) / 2,
            (np.sin(hour_norm) + 1) / 2,
            (np.cos(hour_norm) + 1) / 2
        ]

    def _broadcast_time_features(self, time_features, height, width):
        time_channels = np.zeros((4, height, width), dtype=np.float32)
        for i, val in enumerate(time_features):
            time_channels[i, :, :] = val
        return time_channels
    
    def close(self):
        if self.era5_dyn_ds: self.era5_dyn_ds.close()
        if self.cerra_dyn_ds: self.cerra_dyn_ds.close()
    
    
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