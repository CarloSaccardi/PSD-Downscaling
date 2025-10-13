# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvtx
import torch
import tqdm
import os
from physicsnemo.utils.generative import StackedRandomGenerator
from tueplots import bundles, figsizes
from scipy.stats import ks_2samp  

from typing import Callable, Optional

import torch
from torch import Tensor
from .. import constants
from physicsnemo.utils.patching import GridPatching2D

import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def init_wandb_metrics(wandb_logger):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    for step in constants.VAL_STEP_LOG_ERRORS:
        experiment.define_metric(f"val_loss_unroll{step}", summary="min")


def stochastic_sampler(
    net: torch.nn.Module,
    latents: Tensor,
    img_lr: Tensor,
    class_labels: Optional[Tensor] = None,
    randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
    patching: Optional[GridPatching2D] = None,
    mean_hr: Optional[Tensor] = None,
    lead_time_label: Optional[Tensor] = None,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 800,
    rho: float = 7,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
) -> Tensor:
    """
    Proposed EDM sampler (Algorithm 2) with minor changes to enable
    super-resolution and patch-based diffusion.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model that generates denoised images from noisy
        inputs.
        Expected signature: `net(x, x_lr, t_hat, class_labels,
        lead_time_label=lead_time_label, embedding_selector=embedding_selector)`,
        where:
            x (torch.Tensor): Noisy input of shape (batch_size, C_out, H, W)
            x_lr (torch.Tensor): Conditioning input of shape (batch_size, C_cond, H, W)
            t_hat (torch.Tensor): Noise level of shape (batch_size, 1, 1, 1) or scalar
            class_labels (torch.Tensor, optional): Optional class labels
            lead_time_label (torch.Tensor, optional): Optional lead time labels
            embedding_selector (callable, optional): Function to select
            positional embeddings. Used for patch-based diffusion.
        Returns:
            torch.Tensor: Denoised prediction of shape (batch_size, C_out, H, W)

        Required attributes:
            sigma_min (float): Minimum supported noise level for the model
            sigma_max (float): Maximum supported noise level for the model
            round_sigma (callable): Method to convert sigma values to tensor representation
    latents : Tensor
        The latent variables (e.g., noise) used as the initial input for the
        sampler. Has shape (batch_size, C_out, img_shape_y, img_shape_x).
    img_lr : Tensor
        Low-resolution input image for conditioning the super-resolution
        process. Must have shape (batch_size, C_lr, img_lr_ shape_y,
        img_lr_shape_x).
    class_labels : Optional[Tensor], optional
        Class labels for conditional generation, if required by the model. By
        default None.
    randn_like : Callable[[Tensor], Tensor]
        Function to generate random noise with the same shape as the input
        tensor.
        By default torch.randn_like.
    patching : Optional[GridPatching2D], optional
        A patching utility for patch-based diffusion. Implements methods to
        extract patches from an image and batch the patches along `dim=0`.
        Should also implement a `fuse` method to reconstruct the original image
       from a batch of patches. See
       :class:`physicsnemo.utils.patching.GridPatching2D` for details. By
       default None, in which case non-patched diffusion is used.
    mean_hr : Optional[Tensor], optional
        Optional tensor containing mean high-resolution images for
        conditioning. Must have same height and width as `img_lr`, with shape
        (B_hr, C_hr, img_lr_shape_y, img_lr_shape_x)  where the batch dimension
        B_hr can be either 1, either equal to batch_size, or can be omitted. If
        B_hr = 1 or is omitted, `mean_hr` will be expanded to match the shape
        of `img_lr`. By default None.
    lead_time_label : Optional[Tensor], optional
        Optional lead time labels. By default None.
    num_steps : int
        Number of time steps for the sampler. By default 18.
    sigma_min : float
        Minimum noise level. By default 0.002.
    sigma_max : float
        Maximum noise level. By default 800.
    rho : float
        Exponent used in the time step discretization. By default 7.
    S_churn : float
        Churn parameter controlling the level of noise added in each step. By
        default 0.
    S_min : float
        Minimum time step for applying churn. By default 0.
    S_max : float
        Maximum time step for applying churn. By default float("inf").
    S_noise : float
        Noise scaling factor applied during the churn step. By default 1.

    Returns
    -------
    Tensor
        The final denoised image produced by the sampler. Same shape as
        `latents`: (batch_size, C_out, img_shape_y, img_shape_x).

    See Also
    --------
    :class:`physicsnemo.models.diffusion.EDMPrecondSuperResolution`: A model
        wrapper that provides preconditioning for super-resolution diffusion
        models and implements the required interface for this sampler.
    """

    # Adjust noise levels based on what's supported by the network.
    # Proposed EDM sampler (Algorithm 2) with minor changes to enable
    # super-resolution/
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Safety check on type of patching
    if patching is not None and not isinstance(patching, GridPatching2D):
        raise ValueError("patching must be an instance of GridPatching2D.")

    # Safety check: if patching is used then img_lr and latents must have same
    # height and width, otherwise there is mismatch in the number
    # of patches extracted to form the final batch_size.
    if patching:
        if img_lr.shape[-2:] != latents.shape[-2:]:
            raise ValueError(
                f"img_lr and latents must have the same height and width, "
                f"but found {img_lr.shape[-2:]} vs {latents.shape[-2:]}. "
            )
    # img_lr and latents must also have the same batch_size, otherwise mismatch
    # when processed by the network
    if img_lr.shape[0] != latents.shape[0]:
        raise ValueError(
            f"img_lr and latents must have the same batch size, but found "
            f"{img_lr.shape[0]} vs {latents.shape[0]}."
        )

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    batch_size = img_lr.shape[0]

    # conditioning = [mean_hr, img_lr, global_lr, pos_embd]
    x_lr = img_lr
    if mean_hr is not None:
        if mean_hr.shape[-2:] != img_lr.shape[-2:]:
            raise ValueError(
                f"mean_hr and img_lr must have the same height and width, "
                f"but found {mean_hr.shape[-2:]} vs {img_lr.shape[-2:]}."
            )
        x_lr = torch.cat((mean_hr.expand(x_lr.shape[0], -1, -1, -1), x_lr), dim=1)

    # input and position padding + patching
    if patching:
        # Patched conditioning [x_lr, mean_hr]
        # (batch_size * patch_num, C_in + C_out, patch_shape_y, patch_shape_x)
        x_lr = patching.apply(input=x_lr, additional_input=img_lr)

        # Function to select the correct positional embedding for each patch
        def patch_embedding_selector(emb):
            # emb: (N_pe, image_shape_y, image_shape_x)
            # return: (batch_size * patch_num, N_pe, patch_shape_y, patch_shape_x)
            return patching.apply(emb[None].expand(batch_size, -1, -1, -1))

    else:
        patch_embedding_selector = None

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Increase noise temporarily.
        gamma = S_churn / num_steps if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)

        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step. Perform patching operation on score tensor if patch-based
        # generation is used denoised = net(x_hat, t_hat,
        # class_labels,lead_time_label=lead_time_label).to(torch.float64)

        x_hat_batch = (patching.apply(input=x_hat) if patching else x_hat).to(
            latents.device
        )
        x_lr = x_lr.to(latents.device)

        if lead_time_label is not None:
            denoised = net(
                x_hat_batch,
                x_lr,
                t_hat,
                class_labels,
                lead_time_label=lead_time_label,
                embedding_selector=patch_embedding_selector,
            ).to(torch.float64)
        else:
            denoised = net(
                x_hat_batch,
                x_lr,
                t_hat,
                class_labels,
                embedding_selector=patch_embedding_selector,
            ).to(torch.float64)
        if patching:
            # Un-patch the denoised image
            # (batch_size, C_out, img_shape_y, img_shape_x)
            denoised = patching.fuse(input=denoised, batch_size=batch_size)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            # Patched input
            # (batch_size * patch_num, C_out, patch_shape_y, patch_shape_x)
            x_next_batch = (patching.apply(input=x_next) if patching else x_next).to(
                latents.device
            )

            if lead_time_label is not None:
                denoised = net(
                    x_next_batch,
                    x_lr,
                    t_next,
                    class_labels,
                    lead_time_label=lead_time_label,
                    embedding_selector=patch_embedding_selector,
                ).to(torch.float64)
            else:
                denoised = net(
                    x_next_batch,
                    x_lr,
                    t_next,
                    class_labels,
                    embedding_selector=patch_embedding_selector,
                ).to(torch.float64)
            if patching:
                # Un-patch the denoised image
                # (batch_size, C_out, img_shape_y, img_shape_x)
                denoised = patching.fuse(input=denoised, batch_size=batch_size)

            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next



def diffusion_step(
    net: torch.nn.Module,
    sampler_fn: callable,
    img_shape: tuple,
    img_out_channels: int,
    rank_batches: list,
    img_lr: torch.Tensor,
    rank: int,
    device: torch.device,
    mean_hr: torch.Tensor = None,
    lead_time_label: torch.Tensor = None,
) -> torch.Tensor:

    """
    Generate images using diffusion techniques as described in the relevant paper.

    This function applies a diffusion model to generate high-resolution images based on
    low-resolution inputs. It supports optional conditioning on high-resolution mean
    predictions and lead time labels.

    For each low-resolution sample in `img_lr`, the function generates multiple
    high-resolution samples, with different random seeds, specified in `rank_batches`.
    The function then concatenates these high-resolution samples across the batch dimension.

    Parameters
    ----------
    net : torch.nn.Module
        The diffusion model network.
    sampler_fn : callable
        Function used to sample images from the diffusion model.
    img_shape : tuple
        Shape of the images, (height, width).
    img_out_channels : int
        Number of output channels for the image.
    rank_batches : list
        List of batches of seeds to process.
    img_lr : torch.Tensor
        Low-resolution input image with shape (seed_batch_size, channels_lr, height, width).
    rank : int, optional
        Rank of the current process for distributed processing.
    device : torch.device, optional
        Device to perform computations.
    mean_hr : torch.Tensor, optional
        High-resolution mean tensor to be used as an additional input,
        with shape (1, channels_hr, height, width). Default is None.
    lead_time_label : torch.Tensor, optional
        Lead time label tensor for temporal conditioning,
        with shape (batch_size, lead_time_dims). Default is None.

    Returns
    -------
    torch.Tensor
        Generated images concatenated across batches with shape
        (seed_batch_size * len(rank_batches), out_channels, height, width).
    """

    # Check img_lr dimensions match expected shape
    if img_lr.shape[2:] != img_shape:
        raise ValueError(
            f"img_lr shape {img_lr.shape[2:]} does not match expected shape img_shape {img_shape}"
        )

    # Check mean_hr dimensions if provided
    if mean_hr is not None:
        if mean_hr.shape[2:] != img_shape:
            raise ValueError(
                f"mean_hr shape {mean_hr.shape[2:]} does not match expected shape img_shape {img_shape}"
            )
        if mean_hr.shape[0] != 1:
            raise ValueError(f"mean_hr must have batch size 1, got {mean_hr.shape[0]}")

    img_lr = img_lr.to(memory_format=torch.channels_last)

    # Handling of the high-res mean
    additional_args = {}
    if mean_hr is not None:
        additional_args["mean_hr"] = mean_hr
    if lead_time_label is not None:
        additional_args["lead_time_label"] = lead_time_label

    # Loop over batches
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(rank != 0)):
        with nvtx.annotate(f"generate {len(all_images)}", color="rapids"):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Initialize random generator, and generate latents
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn(
                [
                    img_lr.shape[0],
                    img_out_channels,
                    img_shape[0],
                    img_shape[1],
                ],
                device=device,
            ).to(memory_format=torch.channels_last)

            with torch.inference_mode():
                images = sampler_fn(
                    net, latents, img_lr, randn_like=rnd.randn_like, **additional_args
                )
            all_images.append(images)
    return torch.cat(all_images)


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    bundle = bundles.neurips2023(usetex=False, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


def load_dataset_stats(dataset_name, device="cpu"):
    """
    Load arrays with stored dataset statistics from pre-processing
    """
    static_dir_path = os.path.join(dataset_name, "static")

    def loads_file(fn):
        return torch.load(
            os.path.join(static_dir_path, fn), map_location=device
        )

    data_mean = loads_file("parameter_mean.pt")  # (d_features,)
    data_std = loads_file("parameter_std.pt")  # (d_features,)

    return {
        "data_mean": data_mean,
        "data_std": data_std,
    }



def compute_metrics(
    path_gt: str,
    path_pred: str,
    save_dir: str,
    *,
    var_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compare every .npy file that exists in BOTH `path_gt` and `path_pred`,
    assuming each file has shape (H, W, C) with the same C variables.

    Returns a nested dict: metrics[var_name][metric] = value
    and writes the same information to <save_dir>/metrics.txt.
    """

    path_gt, path_pred = Path(path_gt), Path(path_pred)
    gt_files   = {f.name: f for f in path_gt.glob("*.npy")}
    pred_files = {f.name: f for f in path_pred.glob("*.npy")}
    common     = sorted(gt_files.keys() & pred_files.keys())

    if not common:
        raise FileNotFoundError("No overlapping .npy filenames in the two folders.")

    # ----- discover channel count from the first file
    first = np.load(gt_files[common[0]])
    if first.ndim != 3:
        raise ValueError(
            f"Expected shape (H, W, C). Found {first.shape} in {common[0]!r}"
        )
    C = first.shape[-1]
    if var_names is None:
        var_names = [f"var{c}" for c in range(C)]
    if len(var_names) != C:
        raise ValueError("Length of var_names must equal number of channels (C).")

    # accumulators: metric_sums[var][metric] = running total
    metric_sums = {v: {"MAE": 0.0, "RMSE": 0.0, "SSIM": 0.0, "PSNR": 0.0}
                   for v in var_names}
    n_files = 0

    for fname in common:
        gt  = np.load(gt_files[fname]).astype(np.float32)
        prd = np.load(pred_files[fname]).astype(np.float32)

        gt  = ensure_channels_last(gt,  C)
        prd = ensure_channels_last(prd, C)

        if gt.shape != prd.shape:
            raise ValueError(f"Shape mismatch for {fname}: {gt.shape} vs {prd.shape}")
        if gt.shape[-1] != C:
            raise ValueError(f"Channel count changed in {fname}: got {gt.shape[-1]}, expected {C}")

        for c, vname in enumerate(var_names):
            g = gt[..., c]
            p = prd[..., c]

            metric_sums[vname]["MAE"]    += compute_mae(p, g)
            metric_sums[vname]["RMSE"]   += compute_rmse(p, g)
            metric_sums[vname]["SSIM"]   += compute_ssim_metric(p, g)
            metric_sums[vname]["PSNR"]   += compute_psnr_metric(p, g)
            metric_sums[vname]["Cramer"] += compute_cramer(p, g)

        n_files += 1

    # --- final averages
    metrics = {
        v: {m: total / n_files for m, total in metric_sums[v].items()}
        for v in var_names
    }

    # --- save
    os.makedirs(save_dir, exist_ok=True)
    with open(Path(save_dir) / "metrics_new.txt", "w") as fh:
        for v in var_names:
            fh.write(f"[{v}]\n")
            for m, val in metrics[v].items():
                fh.write(f"  {m}: {val:.6f}\n")
            fh.write("\n")

    return metrics



def compute_mae(p, g):
    return np.abs(p - g).mean()

def compute_rmse(p, g):
    return np.sqrt(np.mean((p - g) ** 2))

def compute_ssim_metric(p, g):
    dr = g.max() - g.min()
    return ssim(g, p, data_range=dr)

def compute_psnr_metric(p, g):
    dr = g.max() - g.min()
    return psnr(g, p, data_range=dr)

def compute_cramer(p, g):
    gx = torch.from_numpy(g.flatten()).float().unsqueeze(1)
    px = torch.from_numpy(p.flatten()).float().unsqueeze(1)
    dxy = torch.cdist(gx, px).mean()
    dgg = torch.cdist(gx, gx).mean()
    dpp = torch.cdist(px, px).mean()
    return (2 * dxy - dgg - dpp).item()

def compute_wind_rmse(u_p, v_p, u_g, v_g):
    """RMSE of wind‑speed magnitude."""
    speed_p = np.hypot(u_p, v_p)
    speed_g = np.hypot(u_g, v_g)
    return np.sqrt(np.mean((speed_p - speed_g) ** 2))

def compute_vorticity_rms(u_p, v_p, u_g, v_g, dx=5500, dy=5500):
    """RMS of vorticity error  ζ = dv/dx − du/dy  (finite‑difference, same grid)."""
    ζ_p = np.gradient(v_p, dx, axis=1) - np.gradient(u_p, dy, axis=0)
    ζ_g = np.gradient(v_g, dx, axis=1) - np.gradient(u_g, dy, axis=0)
    return np.sqrt(np.mean((ζ_p - ζ_g) ** 2))

def compute_ks_metric(p, g):
    """Two‑sample Kolmogorov–Smirnov statistic (two‑sided)."""
    return ks_2samp(p.ravel(), g.ravel(), alternative="two-sided").statistic

def compute_hill_metric(p, g, k=100):
    """Absolute difference of Hill tail indices (heavy‑tail focus)."""
    def hill(x, k):
        x = np.abs(x.ravel()) + 1e-6  # ensure positive
        x_sorted = np.sort(x)[::-1]
        k = min(k, len(x_sorted) - 1)
        x_k = x_sorted[k]
        return (1.0 / k) * np.log(x_sorted[:k] / x_k).sum()
    return abs(hill(p, k) - hill(g, k))


def ensure_channels_last(arr: np.ndarray, C_expected: int) -> np.ndarray:
    """Convert (H,W,C) or (C,H,W) → (H,W,C). Error if neither axis matches C."""
    if arr.shape[-1] == C_expected:      # already (H,W,C)
        return arr
    if arr.shape[0] == C_expected:       # assume (C,H,W)
        return np.moveaxis(arr, 0, -1)   # → (H,W,C)
    raise ValueError(f"Cannot locate channel axis in shape {arr.shape}")



if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from pprint import pprint

    parser = argparse.ArgumentParser(
        description="Compute MAE, RMSE, SSIM, PSNR for two folders of (H,W,C) .npy files."
    )
    parser.add_argument("--path_gt",   required=True, help="Directory with ground-truth .npy files")
    parser.add_argument("--path_pred", required=True, help="Directory with prediction  .npy files")
    parser.add_argument("--save_dir",  required=True, help="Where metrics_new.txt will be written")
    parser.add_argument(
        "--var_names",
        nargs="*",
        default=None,
        help="Optional list of variable names (length must equal channel count, e.g. "
             "--var_names u10 v10 t2m sshf zust).",
    )
    args = parser.parse_args()

    # Resolve paths so the metrics file ends up exactly where you expect
    metrics = compute_metrics(
        Path(args.path_gt).expanduser(),
        Path(args.path_pred).expanduser(),
        Path(args.save_dir).expanduser(),
        var_names=['u10', 'v10', 't2m', 'sshf', 'zust'],
    )

    pprint(metrics)
    
    