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

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import os
from scipy.stats import ks_2samp  
from pprint import pprint

import torch

import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

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
    and writes the same information to <save_dir>/metrics_new.txt.
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
    # if len(var_names) != C:
    #     raise ValueError("Length of var_names must equal number of channels (C).")

    # accumulators: metric_sums[var][metric] = running total
    metric_template = {
        "MAE": 0.0,
        "RMSE": 0.0,
        "SSIM": 0.0,
        "PSNR": 0.0,
        "Cramer": 0.0,
        "KS": 0.0,
        "Hill": 0.0,
    }
    metric_sums = {v: metric_template.copy() for v in var_names}

    # global metrics that combine u and v (channels 0 and 1)
    global_sums: dict[str, float] = {}

    n_files = 0

    # ── per‑file loop ─────────────────────────────────────────────────────────
    for fname in common:
        gt  = np.load(gt_files[fname]).astype(np.float32)
        prd = np.load(pred_files[fname]).astype(np.float32)

        gt  = ensure_channels_last(gt,  C)
        prd = ensure_channels_last(prd, C)
        
        gt_wind_u = gt[..., 0]  # u-component of wind
        gt_wind_v = gt[..., 1]
        prd_wind_u = prd[..., 0]
        prd_wind_v = prd[..., 1]
        wind_speed_prd, wind_speed_gt = compute_wind_speed(prd_wind_u, prd_wind_v, gt_wind_u, gt_wind_v)
        vorticity_prd, vorticity_gt = compute_vorticity(prd_wind_u, prd_wind_v, gt_wind_u, gt_wind_v)
        gt = np.concatenate(
            [gt, wind_speed_gt[..., None], vorticity_gt[..., None]], axis=-1
        )
        prd = np.concatenate(
            [prd, wind_speed_prd[..., None], vorticity_prd[..., None]], axis=-1
        )

        # ── per‑variable metrics ────────────────────────────────────────────
        for c, vname in enumerate(var_names):
            g = gt[..., c]
            p = prd[..., c]

            metric_sums[vname]["MAE"]    += compute_mae(p, g)
            metric_sums[vname]["RMSE"]   += compute_rmse(p, g)
            metric_sums[vname]["SSIM"]   += compute_ssim_metric(p, g)
            metric_sums[vname]["PSNR"]   += compute_psnr_metric(p, g)
            metric_sums[vname]["Cramer"] += 0 #compute_cramer(p, g)
            metric_sums[vname]["KS"]     += compute_ks_metric(p, g)
            metric_sums[vname]["Hill"]   += compute_hill_metric(p, g)

        n_files += 1

    # ── average across files ─────────────────────────────────────────────────
    metrics = {
        v: {m: total / n_files for m, total in metric_sums[v].items()}
        for v in var_names
    }
    if global_sums:
        metrics["_GLOBAL_"] = {m: total / n_files for m, total in global_sums.items()}

    # ── save to disk ─────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / "metrics_new.txt"
    with open(out_path, "w") as fh:
        for v in list(metrics.keys()):
            fh.write(f"[{v}]\n")
            for m, val in metrics[v].items():
                fh.write(f"  {m}: {val:.6f}\n")
            fh.write("\n")

    return metrics


# ─────────────────────── basic metrics helpers ───────────────────────────────
def compute_mae(p: np.ndarray, g: np.ndarray) -> float:
    return np.abs(p - g).mean()


def compute_rmse(p: np.ndarray, g: np.ndarray) -> float:
    return np.sqrt(np.mean((p - g) ** 2))


def compute_ssim_metric(p: np.ndarray, g: np.ndarray) -> float:
    dr = g.max() - g.min()
    return float(ssim(g, p, data_range=dr))


def compute_psnr_metric(p: np.ndarray, g: np.ndarray) -> float:
    dr = g.max() - g.min()
    return float(psnr(g, p, data_range=dr))


def _mean_pairwise_abs(sorted_v: np.ndarray) -> float:
    n      = sorted_v.size
    coeffs = 2 * np.arange(n) - n + 1        # 0‑based version of 2i−n−1
    return (2.0 / n**2) * coeffs.dot(sorted_v)

def _mean_cross_abs(sorted_x: np.ndarray, sorted_y: np.ndarray) -> float:
    n, m   = sorted_x.size, sorted_y.size
    cumsum = np.concatenate(([0.0], np.cumsum(sorted_y)))
    idx    = np.searchsorted(sorted_y, sorted_x, side="left")

    left  = sorted_x * idx             - cumsum[idx]
    right = (cumsum[-1] - cumsum[idx]) - sorted_x * (m - idx)
    return (left + right).sum() / (n * m)

def compute_cramer(p: np.ndarray, g: np.ndarray) -> float:
    p_sorted = np.sort(p.ravel())
    g_sorted = np.sort(g.ravel())

    dxy = _mean_cross_abs(g_sorted, p_sorted)
    dgg = _mean_pairwise_abs(g_sorted)
    dpp = _mean_pairwise_abs(p_sorted)
    return 2.0 * dxy - dgg - dpp


# ───────────────────── physics‑oriented helpers ──────────────────────────────
def compute_wind_speed(u_p: np.ndarray, v_p: np.ndarray,
                      u_g: np.ndarray, v_g: np.ndarray) -> float:
    """RMSE of wind‑speed magnitude."""
    speed_p = np.hypot(u_p, v_p)
    speed_g = np.hypot(u_g, v_g)
    return speed_p, speed_g


def compute_vorticity(u_p: np.ndarray, v_p: np.ndarray,
                           u_g: np.ndarray, v_g: np.ndarray,
                           dx: float = 1.0, dy: float = 1.0) -> float:
    """
    RMS of vorticity error ζ = ∂v/∂x − ∂u/∂y using centred finite differences.
    """
    ζ_p = np.gradient(v_p, dx, axis=1) - np.gradient(u_p, dy, axis=0)
    ζ_g = np.gradient(v_g, dx, axis=1) - np.gradient(u_g, dy, axis=0)
    return ζ_g, ζ_p


def compute_ks_metric(p: np.ndarray, g: np.ndarray) -> float:
    """
    Two‑sample Kolmogorov–Smirnov statistic (two‑sided).
    """
    return ks_2samp(p.ravel(), g.ravel(), alternative="two-sided").statistic


def compute_hill_metric(p: np.ndarray, g: np.ndarray, k: int = 100) -> float:
    """
    Absolute difference of Hill tail indices – focuses on heavy‑tail behaviour.
    """
    def hill(x: np.ndarray, k_: int) -> float:
        x = np.abs(x.ravel()) + 1e-6     # ensure strictly positive
        x_sorted = np.sort(x)[::-1]      # descending
        k_ = min(k_, len(x_sorted) - 1)
        x_k = x_sorted[k_]
        return (1.0 / k_) * np.log(x_sorted[:k_] / x_k).sum()

    return abs(hill(p, k) - hill(g, k))


# ─────────────────────────── utility ─────────────────────────────────────────
def ensure_channels_last(arr: np.ndarray, C_expected: int) -> np.ndarray:
    """
    Convert array to (H, W, C) format if currently (C, H, W).
    Raises if channel axis cannot be found.
    """
    if arr.shape[-1] == C_expected:      # already (H, W, C)
        return arr
    if arr.shape[0] == C_expected:       # assume (C, H, W)
        return np.moveaxis(arr, 0, -1)   # -> (H, W, C)
    raise ValueError(f"Cannot locate channel axis in shape {arr.shape}")


def mass_conservartion(p_s, u, v, T):
        
    Nt, H, W = T.shape                                    # (2090, 400, 400)

    # ---------------------------------------------------------------------
    # 2.  Thermodynamics: density field -----------------------------------
    # ---------------------------------------------------------------------
    R_d = 287.05                                          # J kg‑1 K‑1 (dry‑air gas constant)
    rho = p_s / (R_d * T)                                 # kg m‑3, shape (N_t, H, W)

    # ---------------------------------------------------------------------
    # 3.  Mass‑fluxes through cell faces ----------------------------------
    #      • F_x = ρ u   (zonal  , normal to left/right faces)
    #      • F_y = ρ v   (meridional, normal to top/bottom faces)
    # ---------------------------------------------------------------------
    F_x = rho * u                                         # kg m‑2 s‑1
    F_y = rho * v

    # grid spacing (metres).  If you actually know Δx,Δy, replace the 1.0:
    dx = dy = 5500.0

    # ---------------------------------------------------------------------
    # 4.  Finite‑volume divergence  ∇·(ρu,ρv) ------------------------------
    #     Central difference with cyclic (np.roll) boundaries—good enough
    #     for metric‑style evaluation.  If edges must be excluded, slice
    #     away the first/last row & column after computing `div`.
    # ---------------------------------------------------------------------
    dFdx = (np.roll(F_x, -1, axis=2) - np.roll(F_x,  1, axis=2)) / (2*dx)
    dFdy = (np.roll(F_y, -1, axis=1) - np.roll(F_y,  1, axis=1)) / (2*dy)
    div  = dFdx + dFdy                                    # shape (N_t, H, W)
    #edges must be excluded, slice away the first/last row & column
    div = div[:, 2:-2, 2:-2]                        # shape (N_t, H-2, W-2)

    # ---------------------------------------------------------------------
    # 5.  Scalar error metrics --------------------------------------------
    #     “0 ≈ …”  →  we measure how *far* from zero we are.
    # ---------------------------------------------------------------------
    # summation = div.sum(axis=(1,2))               # |∇·(ρu,ρv)|   kg m‑3 s‑1
    # MAE = np.mean(np.abs(summation))            # mean absolute error

    # print(f"Mean |∇·(ρu,ρv)|  : {MAE: .3e}  kg m⁻³ s⁻¹")
    return div

# ------------------------------------------------------------------
#  Tiny helpers – unchanged
# ------------------------------------------------------------------
def imshow_with_cbar(ax, data, title, *,
                     cmap='bwr', origin='lower',
                     vmin=None, vmax=None, norm=None,
                     cbar_kw=None,
                     add_cbar=False):              # ← NEW flag
    im = ax.imshow(data, cmap=cmap, origin=origin,
                   vmin=vmin, vmax=vmax, norm=norm)
    ax.set_title(title, fontsize=10)

    if add_cbar:                                # ← only when asked
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(im, cax=cax, **(cbar_kw or {}))
    return im

def shared_limits(gt, pred):
    #clip the values to the 0.1 and 99.9 percentiles
    gt = np.clip(gt, np.percentile(gt, 0.1), np.percentile(gt, 99.9))
    pred = np.clip(pred, np.percentile(pred, 0.1), np.percentile(pred, 99.9))
    return float(np.min([gt, pred])), float(np.max([gt, pred]))

# ------------------------------------------------------------------
#  NEW: one‑stop routine for a single variable
# ------------------------------------------------------------------
def plot_triplet(var_name: str,
                 gt2d: np.ndarray,
                 pred2d: np.ndarray,
                 *,
                 residual_kind: str = "diff",  # "diff"  or "sq"
                 out_dir: Path = Path(".")):
    """
    Draws a 1×3 panel [GT | Pred | Residual] for one variable and writes
    <out_dir>/<var_name>_triplet.png.
    """

    # ---------- choose limits & norms ----------
    vmin, vmax = shared_limits(gt2d, pred2d)

    if residual_kind == "diff":
        resid = pred2d - gt2d
        # resid = np.abs(resid)  
        rmax = np.percentile(resid, 99)  # 99th percentile
        rmin = np.percentile(resid, 1)    # 1st percentile
    else:                                  # squared residuals (always ≥0)
        resid = (pred2d - gt2d) ** 2
        rmin, rmax = 0, np.percentile(resid, 99)

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)
    (ax_gt, ax_pd, ax_rs) = axes
    

    im_gt   = imshow_with_cbar(ax_gt, gt2d,  f"{var_name} • GT",
                            cmap='bwr', vmin=vmin, vmax=vmax)
    imshow_with_cbar(ax_pd, pred2d, f"{var_name} • Pred",
                    cmap='bwr', vmin=vmin, vmax=vmax)
    im_rs   = imshow_with_cbar(ax_rs, resid,  f"{var_name} • Residual",
                            cmap='bwr', vmin=rmin, vmax=rmax) 

    fig.colorbar(im_gt, ax=[ax_gt, ax_pd],        # anchors to the first two axes
                orientation='horizontal',
             fraction=0.05, pad=0.08)
    fig.colorbar(im_rs, ax=ax_rs,                 # anchored to the residual axis only
                orientation='horizontal',
                fraction=0.05, pad=0.08)


    fig.savefig(out_dir / f"{var_name}_triplet.png", dpi=300)
    plt.close(fig)



# ──────────────────────────── CLI ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute metrics for two folders of (H,W,C) .npy files."
    )
    parser.add_argument("--path_gt", help="Directory with ground‑truth .npy files")
    parser.add_argument("--path_pred", help="Directory with prediction  .npy files")
    parser.add_argument( "--path_pressure",help="Directory with pressure .npy files for physics metrics")
    parser.add_argument("--physics_metrics", type=bool, help="Compute physics metrics", default=False)
    parser.add_argument("--save_dir", help="Where metrics_new.txt will be written")
    parser.add_argument("--plot_residual", type=bool, help="Plot residuals", default=False)
    parser.add_argument(
        "--var_names",
        nargs="*",
        default=None,
        help=("Optional list of variable names (length must equal channel count), "
              "e.g. --var_names u10 v10 t2m sshf zust"),
    )
    args = parser.parse_args()

    if args.physics_metrics:
    #     div_gt = mass_conservartion(args.path_pressure, args.path_gt, gt=True)
    #     div_pred = mass_conservartion(args.path_pressure, args.path_pred, gt=False)
    #     #compute RMSE of divergence
    #     rmse_div = np.sqrt(np.mean((div_gt - div_pred) ** 2))
    #     print(f"RMSE of divergence: {rmse_div:.6f} kg m⁻³ s⁻¹")
    # else:
        metrics = compute_metrics(
            args.path_gt,
            args.path_pred,
            args.save_dir,
            var_names=['u10', 'v10', 't2m', 'sshf', 'zust', "wind_speed", "vorticity"],
        )

        pprint(metrics)
        
    if args.plot_residual:   
        output_dir = Path(args.save_dir or ".") / "triplet_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = args.path_pred.split("/")[-2]
        
        path_gt, path_pred, path_press = Path(args.path_gt), Path(args.path_pred), Path(args.path_pressure)

        gt_files   = {f.name: f for f in path_gt.glob("*.npy")}
        pred_files = {f.name: f for f in path_pred.glob("*.npy")}
        press_files = {f.name: f for f in path_press.glob("*.npy")}
        common     = sorted(gt_files.keys() & pred_files.keys() & press_files.keys())
        
        u_gt   = np.array([np.load(gt_files[f])[:, :, 0] for f in common])   # (N, H, W) --> (t, y, x)
        v_gt   = np.array([np.load(gt_files[f])[:, :, 1] for f in common])
        T_gt   = np.array([np.load(gt_files[f])[:, :, 2] for f in common])

        u_pred = np.array([np.load(pred_files[f])[0, :, :] for f in common]) # (N, H, W)
        v_pred = np.array([np.load(pred_files[f])[1, :, :] for f in common])
        T_pred = np.array([np.load(pred_files[f])[2, :, :] for f in common])
        
        p_s = np.array([np.load(press_files[f]) for f in common])
        
        mass_conservartion_gt = mass_conservartion(p_s, u_gt, v_gt, T_gt)
        mass_conservartion_pred = mass_conservartion(p_s, u_pred, v_pred, T_pred)
        
        # wind_speed_gt = np.hypot(u_gt, v_gt)
        # wind_speed_pred = np.hypot(u_pred, v_pred)
        
        wind_vorticity_gt = np.gradient(v_gt, 5500.0, axis=2) - np.gradient(u_gt, 5500.0, axis=1)
        wind_vorticity_pred = np.gradient(v_pred, 5500.0, axis=2) - np.gradient(u_pred, 5500.0, axis=1)
        
        wind_divergence_gt = np.gradient(v_gt, 5500.0, axis=1) + np.gradient(u_gt, 5500.0, axis=2)
        wind_divergence_pred = np.gradient(v_pred, 5500.0, axis=1) + np.gradient(u_pred, 5500.0, axis=2)
        
        u_gt_mean   = u_gt.mean(axis=0)
        v_gt_mean   = v_gt.mean(axis=0)
        u_pred_mean = u_pred.mean(axis=0)
        v_pred_mean = v_pred.mean(axis=0)

        # wind_speed_gt_mean      = wind_speed_gt.mean(axis=0)
        # wind_speed_pred_mean    = wind_speed_pred.mean(axis=0)
        
        mass_conservartion_gt_mean = mass_conservartion_gt.mean(axis=0)
        mass_conservartion_pred_mean = mass_conservartion_pred.mean(axis=0)

        wind_vorticity_gt_mean  = wind_vorticity_gt.mean(axis=0)
        wind_vorticity_pred_mean= wind_vorticity_pred.mean(axis=0)

        wind_divergence_gt_mean = wind_divergence_gt.mean(axis=0)
        wind_divergence_pred_mean = wind_divergence_pred.mean(axis=0)

        triplets = [
            # ("u",          u_gt_mean,           u_pred_mean),
            # ("v",          v_gt_mean,           v_pred_mean),
            # ("speed",      wind_speed_gt_mean,  wind_speed_pred_mean),
            ("cons_mass", mass_conservartion_gt_mean, mass_conservartion_pred_mean),
            ("vorticity",  wind_vorticity_gt_mean, wind_vorticity_pred_mean),
            ("divergence", wind_divergence_gt_mean, wind_divergence_pred_mean),
        ]

        for name, gt2d, pred2d in triplets:
            output_dir_model = Path(output_dir) / model_name
            output_dir_model.mkdir(parents=True, exist_ok=True)
            prefixed_name = f"{name}"              # ⇐ add prefix
            plot_triplet(prefixed_name, gt2d, pred2d,
                        residual_kind="diff",
                        out_dir=output_dir_model)
        print(f"✓ wrote one PNG per variable in {output_dir_model}")