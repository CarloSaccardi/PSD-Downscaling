# Third-party
import matplotlib
import matplotlib.pyplot as plt
import torch

# First-party
from . import constants, utils


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_ensemble_prediction(
    prediction, target, mask, title=None, vrange=None
):
    """
    Plot example predictions, ground truth, mean and std.-dev.
    from ensemble forecast

    prediction: (S, N_grid,)
    target: (N_grid,)
    ens_mean: (N_grid,)
    ens_std: (N_grid,)
    obs_mask: (N_grid,)
    (optional) title: title of plot
    (optional) vrange: tuple of length with common min and max of values
        (not for std.)
    """
    
    target_masked = target * (~mask)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(25, 15),
        #subplot_kw={"projection": constants.LAMBERT_PROJ},
    )
    axes = axes.flatten()

    # Plot target, ensemble mean and std.
    gt_im = plot_on_axis(
        axes[0],
        target_masked,
        vmin=target.min().item(),
        vmax=target.max().item(),
        ax_title="Ground Truth - Masked",
    )
    plot_on_axis(
        axes[1],
        target,
        vmin=target.min().item(),
        vmax=target.max().item(),
        ax_title="Ground Truth",
    )
    plot_on_axis(
        axes[2],
        prediction,
        vmin=prediction.min().item(),
        vmax=prediction.max().item(),
        ax_title="Prediction",
    )
    # Turn off unused axes
    for ax in axes[(3 + prediction.shape[0]) :]:
        ax.axis("off")

    # Add colorbars
    values_cbar = fig.colorbar(
        gt_im, ax=axes[:3], aspect=60, location="bottom", shrink=0.9
    )
    values_cbar.ax.tick_params(labelsize=10)
    #std_cbar = fig.colorbar(std_im, aspect=30, location="bottom", shrink=0.9)
    #std_cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


def plot_on_axis(ax, data, vmin=None, vmax=None, ax_title=None):
    """
    Plot weather state on given axis
    """
    #ax.coastlines()  # Add coastline outlines
    data_grid = data.reshape(*constants.GRID_SHAPE_CROPPED).to(torch.float32).cpu().numpy()
    im = ax.imshow(
        data_grid,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )

    if ax_title:
        ax.set_title(ax_title, size=15)
    return im

