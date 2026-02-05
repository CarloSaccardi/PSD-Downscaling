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
                 root_dir_conditions,
                 split,
                 region,
                 conditioning,
                 crop_size=None,
                 era5_vars=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp'], 
                 cerra_vars=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp'],
                 conditions_vars=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp']):
        super().__init__()
        
        self.era5_vars = era5_vars
        self.cerra_vars = cerra_vars
        self.conditions_vars = conditions_vars
        self.crop_size = crop_size
        self.conditioning = conditioning
        # Paths
        era5_path = os.path.join(root_dir_era5, split, f"{region}.nc")
        cerra_path = os.path.join(root_dir_cerra, split, f"{region}.nc")
        conditions_path = os.path.join(root_dir_conditions, split, f"{region}.nc")
        conditions_orography_path = os.path.join(root_dir_conditions, split, f"static_{region}.nc")
        cerra_orography_path = os.path.join(root_dir_cerra, split, f"static_{region}.nc")
        
        # 1. Load Statistics
        self.eurasia_mean = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_mean.npy"))
        self.eurasia_std = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_std.npy"))
        
        self.eurasia_orography_mean = np.load(os.path.join(root_dir_era5, "statistics", "forcing_mean.npy"))
        self.eurasia_orography_std = np.load(os.path.join(root_dir_era5, "statistics", "forcing_std.npy"))

        # 2. Open Datasets (Lazy Xarray)
        self.era5_ds = xr.open_dataset(era5_path, engine="h5netcdf").sortby('latitude') #wrong order in saved file
        self.conditions_ds = xr.open_dataset(conditions_path, engine="h5netcdf").sortby('latitude')
        self.cerra_ds = xr.open_dataset(cerra_path, engine="h5netcdf")
        
        # 3. Load Static Data into RAM (Optimization)
        # We perform the static normalization ONCE here to save CPU cycles in __getitem__
        ds_static_cerra = xr.open_dataset(cerra_orography_path, engine="h5netcdf")
        raw_static_cerra = ds_static_cerra['orog'].values.astype(np.float32)
        self.cerra_orography = torch.from_numpy((raw_static_cerra - self.eurasia_orography_mean) / self.eurasia_orography_std)
        ds_static_cerra.close()
        
        ds_static_conditions = xr.open_dataset(conditions_orography_path, engine="h5netcdf").sortby('latitude')
        raw_static_conditions = ds_static_conditions['geopotential'].values.astype(np.float32)
        self.conditions_orography = torch.from_numpy((raw_static_conditions - self.eurasia_orography_mean) / self.eurasia_orography_std).unsqueeze(0)
        ds_static_conditions.close()
        
        # 4. Get dimensions
        self.lat_len = len(self.cerra_ds.latitude)
        self.lon_len = len(self.cerra_ds.longitude)
        
        # 6. Pre-calculate max indices for random cropping
        self.max_lat_idx = self.lat_len - self.crop_size
        self.max_lon_idx = self.lon_len - self.crop_size

    def __len__(self):
        time_axis = self.era5_ds.time.values
        return len(time_axis)
    

    def __getitem__(self, idx):
        
        cerra = torch.from_numpy(self._load_dynamic_step(self.cerra_ds, self.cerra_vars, idx))
        era5 = torch.from_numpy(self._load_dynamic_step(self.era5_ds, self.era5_vars, idx))
        
        era5 = F.interpolate(era5.unsqueeze(0), 
                             size=(cerra.shape[-1], cerra.shape[-1]), 
                             mode='bicubic', 
                             align_corners=False).squeeze(0)
        
        # 2. Normalize Dynamic Data
        era5 = (era5 - self.eurasia_mean[:, None, None]) / self.eurasia_std[:, None, None]
        cerra = (cerra - self.eurasia_mean[:, None, None]) / self.eurasia_std[:, None, None]
        cerra_orography = (self.cerra_orography - self.eurasia_orography_mean) / self.eurasia_orography_std
        
        # 3. concatenate conditions with conditions orography 
        era5 =torch.cat([era5, cerra_orography], dim=0)
        
        if self.crop_size is not None:
            lat_idx, lon_idx = self._get_random_crop_indices()
            era5 = self._crop_data(era5, lat_idx, lon_idx)
            cerra = self._crop_data(cerra, lat_idx, lon_idx)
            cerra_orography = self._crop_data(cerra_orography, lat_idx, lon_idx)
            
        if self.conditioning:
            conditions = torch.from_numpy(self._load_dynamic_step(self.conditions_ds, self.conditions_vars, idx))
            conditions = (conditions - self.eurasia_mean[:, None, None]) / self.eurasia_std[:, None, None]
            conditions = torch.cat([conditions, self.conditions_orography], axis=0)
        else:
            conditions = torch.tensor([0.0])
        
        
        return era5, cerra, conditions
    

    def _load_dynamic_step(self, dataset, variables, idx):
        """Lazy load specific time step."""
        data_sel = dataset[variables].isel(time=idx)
        return data_sel.to_array().values.astype(np.float32)
    
    # --- Helper Functions ---

    def _get_random_crop_indices(self):
        """Returns random starting indices for a latitude and longitude."""
        # Sample crop center uniformly across the full grid, then clamp
        lat_idx = np.random.randint(0, self.max_lat_idx + 1)
        lon_idx = np.random.randint(0, self.max_lon_idx + 1)
        return lat_idx, lon_idx

    def _crop_data(self, dataset,lat_idx, lon_idx):
        """Crops the static data (already in RAM)."""
        data_crop = dataset[
            :,
            lat_idx : lat_idx + self.crop_size,
            lon_idx : lon_idx + self.crop_size
        ]
        return data_crop

    
    def close(self):
        if self.era5_ds: self.era5_ds.close()
        if self.cerra_ds: self.cerra_ds.close()
    
    
    
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
        # time_features = self._get_time_embedding(idx)
        
        # 3. Load/crop all data channels
        crop_dynamics = self._load_crop_dynamic(idx, lat_idx, lon_idx)
        crop_static = self._crop_static(lat_idx, lon_idx)
        
        # 4. Normalize
        crop_dynamics = (crop_dynamics - self.mean_dynamic_vars[:, None, None]) / self.std_dynamic_vars[:, None, None]
        crop_static = (crop_static - self.mean_static_var) / self.std_static_var
        
        # 5. Concatenate and return
        all_features = np.concatenate([crop_dynamics, crop_static], axis=0)
        
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
        # Sample crop center uniformly across the full grid, then clamp
        lat_idx = np.random.randint(0, self.max_lat_idx + 1)
        lon_idx = np.random.randint(0, self.max_lon_idx + 1)
        return lat_idx, lon_idx


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
            

