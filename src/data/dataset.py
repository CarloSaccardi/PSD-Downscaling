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