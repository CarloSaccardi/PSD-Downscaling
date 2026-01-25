# Standard library
import datetime as dt
import os
import numpy as np
import torch
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
        
        self.cerra_dyn_mean = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_mean.npy"))
        self.cerra_dyn_std = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_std.npy"))
        
        self.cerra_stat_mean = np.load(os.path.join(root_dir_era5, "statistics", "forcing_mean.npy"))
        self.cerra_stat_std = np.load(os.path.join(root_dir_era5, "statistics", "forcing_std.npy"))

        # 2. Open Datasets (Lazy Xarray)
        self.era5_dyn_ds = xr.open_dataset(era5_dyn_path, engine="h5netcdf").sortby('latitude')
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
        
        return full_input, cerra_target

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
    
    
    
class Era5CropDataset(torch.utils.data.Dataset):
    """
    Refactored custom PyTorch Dataset for ERA5 data.
    - Uses helper functions in __getitem__ for clarity.
    - Loads static data into RAM in __init__ for speed.
    """
    def __init__(self, 
                 path, 
                 split,
                 mask_ratio,
                 model_patch_size,
                 mask_patch_size,
                 crop_size,
                 variables=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp']):
        super().__init__()
        
        stats_dir = os.path.join(path, "statistics")
        dynamic_f = os.path.join(path, split, "Eurasia.nc")
        forcing_f = os.path.join(path, split, "static_Eurasia.nc")
        
        # 1. Load statistics
        self.mean_dynamic_vars = np.load(f"{stats_dir}/dynamic_mean.npy")
        self.std_dynamic_vars = np.load(f"{stats_dir}/dynamic_std.npy")
        self.mean_static_var = np.load(f"{stats_dir}/forcing_mean.npy")
        self.std_static_var = np.load(f"{stats_dir}/forcing_std.npy")
        
        # 2. Open datasets
        self.dynamic_f = xr.open_dataset(dynamic_f).sortby('latitude') #, chunks={'time': 1})
        self.forcing_f = xr.open_dataset(forcing_f).sortby('latitude')
        
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
        
        #7. Initialize masking generator
        self.mask_generator = MaskGenerator(
            input_size=crop_size,  # Your final upsampled size
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size, # Must match your model's patch size
            mask_ratio=mask_ratio
        )
        
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
        
        # 6. Convert to torch tensor
        all_features_tensor = torch.from_numpy(all_features.copy()).float()
        
        # 7. --- GENERATE MASK ---
        # Call the generator you made in __init__
        mask = self.mask_generator()
        # Convert mask from numpy array to a torch tensor
        mask_tensor = torch.from_numpy(mask).float()
        
        return all_features_tensor, mask_tensor

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
            #    This executes the read, pulling ONLY the [n_vars, 96, 96]
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

    def close(self):
        """Closes the xarray dataset file handle."""
        if self.dynamic_f:
            self.dynamic_f.close()
        if self.forcing_f:
            self.forcing_f.close()
            


class MaskGenerator:
    def __init__(self, input_size, mask_patch_size, model_patch_size, mask_ratio):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask
     
     
     
    
class ConditionsDataset(torch.utils.data.Dataset):
    """
    Refactored custom PyTorch Dataset for ERA5 data.
    - Uses helper functions in __getitem__ for clarity.
    - Loads static data into RAM in __init__ for speed.
    """
    def __init__(self, 
                 path, 
                 split,
                 mask_ratio,
                 model_patch_size,
                 mask_patch_size,
                 crop_size,
                 region,
                 variables=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp']):
        super().__init__()
        
        stats_dir = os.path.join(path, "statistics")
        dynamic_f = os.path.join(path, split, region + ".nc")
        forcing_f = os.path.join(path, split, "static_" + region + ".nc")
        
        # 1. Load statistics
        self.mean_dynamic_vars = np.load(f"{stats_dir}/dynamic_mean.npy")
        self.std_dynamic_vars = np.load(f"{stats_dir}/dynamic_std.npy")
        self.mean_static_var = np.load(f"{stats_dir}/forcing_mean.npy")
        self.std_static_var = np.load(f"{stats_dir}/forcing_std.npy")
        
        # 2. Open datasets
        self.dynamic_f = xr.open_dataset(dynamic_f).sortby('latitude') #, chunks={'time': 1})
        self.forcing_f = xr.open_dataset(forcing_f).sortby('latitude')
        
        # 3. Load static data into RAM (This is a key optimization)
        self.geopotential_data = (self.forcing_f['geopotential'].values.astype(np.float32) - self.mean_static_var) / self.std_static_var
        
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
        
        #7. Initialize masking generator
        self.mask_generator = MaskGenerator(
            input_size=crop_size,  # Your final upsampled size
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size, # Must match your model's patch size
            mask_ratio=mask_ratio
        )
        
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
        
        # 1. Get item
        crop_dynamics = self._load_dynamic_step(self.dynamic_f, self.variables, idx)
        
        # 2. Get 4 time embedding features
        time_features = self._get_time_embedding(idx)
        
        time_channels = self._broadcast_time_features(time_features)
        
        # 4. Normalize
        crop_dynamics = (crop_dynamics - self.mean_dynamic_vars[:, None, None]) / self.std_dynamic_vars[:, None, None]
        crop_static = torch.from_numpy(self.geopotential_data[None, :, :]).float()
        
        # 5. Concatenate and return
        all_features = np.concatenate([crop_dynamics, crop_static, time_channels], axis=0)
        
        # 6. Convert to torch tensor
        all_features_tensor = torch.from_numpy(all_features.copy()).float()
        
        # 7. --- GENERATE MASK ---
        # Call the generator you made in __init__
        mask = self.mask_generator()
        # Convert mask from numpy array to a torch tensor
        mask_tensor = torch.from_numpy(mask).float()
        
        return all_features_tensor, mask_tensor

    # --- Helper Functions ---
    def _load_dynamic_step(self, dataset, variables, idx):
        """Lazy load specific time step."""
        data_sel = dataset[variables].isel(time=idx)
        return data_sel.to_array().values.astype(np.float32)

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
            #    This executes the read, pulling ONLY the [n_vars, 96, 96]
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

    def close(self):
        """Closes the xarray dataset file handle."""
        if self.dynamic_f:
            self.dynamic_f.close()
        if self.forcing_f:
            self.forcing_f.close()
            

