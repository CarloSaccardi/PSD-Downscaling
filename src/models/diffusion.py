import importlib
from physicsnemo.utils.patching import RandomPatching2D
from physicsnemo.models.diffusion import EDMPrecond

import wandb
import nvtx
import torch
import pytorch_lightning as pl
from src.utils import vis
from src.utils import constants
import random

import matplotlib.pyplot as plt

import os
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim_func


network_module = importlib.import_module("physicsnemo.models.diffusion")



class DiffusionWrapper(pl.LightningModule):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM).

    This is a variant of `EDMPrecond` that is specifically designed for super-resolution
    tasks. It wraps a neural network that predicts the denoised high-resolution image
    given a noisy high-resolution image, and additional conditioning that includes a
    low-resolution image, and a noise level.

    Parameters
    ----------
    img_resolution : Union[int, Tuple[int, int]]
        Spatial resolution `(H, W)` of the image. If a single int is provided,
        the image is assumed to be square.
    img_in_channels : int
        Number of input channels in the low-resolution input image.
    img_out_channels : int
        Number of output channels in the high-resolution output image.
    use_fp16 : bool, optional
        Whether to use half-precision floating point (FP16) for model execution,
        by default False.
    model_type : str, optional
        Class name of the underlying model. Must be one of the following:
        'SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'.
        Defaults to 'SongUNetPosEmbd'.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.
    sigma_min : float, optional
        Minimum supported noise level, by default 0.0.
    sigma_max : float, optional
        Maximum supported noise level, by default inf.
    **model_kwargs : dict
        Keyword arguments passed to the underlying model `__init__` method.

    See Also
    --------
    For information on model types and their usage:
    :class:`~physicsnemo.models.diffusion.SongUNet`: Basic U-Net for diffusion models
    :class:`~physicsnemo.models.diffusion.SongUNetPosEmbd`: U-Net with positional embeddings
    :class:`~physicsnemo.models.diffusion.SongUNetPosLtEmbd`: U-Net with positional and lead-time embeddings

    Please refer to the documentation of these classes for details on how to call
    and use these models directly.

    Note
    ----
    References:
    - Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    - Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(self, args):
        super().__init__()
        self.img_resolution = args.img_resolution
        self.img_in_channels = args.img_in_channels
        self.img_out_channels = args.img_out_channels
        self.sigma_data: float = 0.5
        self.sigma_min=0.0
        self.sigma_max=float("inf")
        self.savepreds_path = args.savepreds_path
        self.load = args.load
        self.P_mean = 0.0 # mean value for noise level computation
        self.P_std = 1.2 # standard deviation for noise level computation
        self.sigma_data = 0.5 # standard deviation for data weighting
        self.wandb_project = args.wandb_project
        
        self.lr = args.lr
        
        self.model_kwargs = {
            'checkpoint_level': args.checkpoint_level,
            'gridtype': args.gridtype,
            'N_grid_channels': args.N_grid_channels,
            'embedding_type': args.embedding_type,
            'model_channels': args.model_channels,
            'channel_mult': args.channel_mult,
            'attn_resolutions': args.attn_resolutions,
        }

        model_class = getattr(network_module, args.model_type)
        self.model = model_class(
            img_resolution=self.img_resolution,
            in_channels=self.img_in_channels + args.N_grid_channels + self.img_out_channels,
            out_channels=self.img_out_channels,
            **self.model_kwargs,
        )  # TODO needs better handling
        self.scaling_fn = self._scaling_fn
        
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=[0.9, 0.999], eps=1e-8
            )
        return opt

    @staticmethod
    def _scaling_fn(
        x: torch.Tensor, img_lr: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale input tensors by first scaling the high-resolution tensor and then
        concatenating with the low-resolution tensor.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution image of shape (B, C_lr, H, W).
        c_in : torch.Tensor
            Scaling factor of shape (B, 1, 1, 1).

        Returns
        -------
        torch.Tensor
            Scaled and concatenated tensor of shape (B, C_in+C_out, H, W).
        """
        return torch.cat([c_in * x, img_lr.to(x.dtype)], dim=1)

    @nvtx.annotate(message="EDMPrecondSuperResolution", color="orange")
    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """
        Forward pass of the EDMPrecondSuperResolution model wrapper.

        This method applies the EDM preconditioning to compute the denoised image
        from a noisy high-resolution image and low-resolution conditioning image.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W). The number of
            channels `C_hr` should be equal to `img_out_channels`.
        img_lr : torch.Tensor
            Low-resolution conditioning image of shape (B, C_lr, H, W). The number
            of channels `C_lr` should be equal to `img_in_channels`.
        sigma : torch.Tensor
            Noise level of shape (B) or (B, 1) or (B, 1, 1, 1).
        force_fp32 : bool, optional
            Whether to force FP32 precision regardless of the `use_fp16` attribute,
            by default False.
        **model_kwargs : dict
            Additional keyword arguments to pass to the underlying model
            `self.model` forward method.

        Returns
        -------
        torch.Tensor
            Denoised high-resolution image of shape (B, C_hr, H, W).

        Raises
        ------
        ValueError
            If the model output dtype doesn't match the expected dtype.
        """
        # Concatenate input channels
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2) 
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        # if img_lr is None:
        #     arg = c_in * x
        # else:
        arg = self.scaling_fn(x, img_lr, c_in)

        F_x = self.model(
            arg,
            c_noise.flatten(),
            class_labels=None,
            **model_kwargs,
        )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x
    
    def training_step(self, batch, *args):
        img_clean, img_lr = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        latent = img_clean + torch.randn_like(img_clean) * sigma
        
        D_yn = self(
            latent,
            img_lr,
            sigma,
            embedding_selector=None,
            augment_labels=None,
        )
        
        train_loss = torch.mean(weight * (D_yn - img_clean) ** 2)
        
        log_dict = {
            "train_loss": train_loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return train_loss

    def validation_step(self, batch, *args):
        img_clean, img_lr = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        latent = img_clean + torch.randn_like(img_clean) * sigma
        
        D_yn = self(
            latent,
            img_lr,
            sigma,
            embedding_selector=None,
            augment_labels=None,
        )
        
        val_loss = torch.mean(weight * (D_yn - img_clean) ** 2)
        
        # Log loss per time step forward and mean
        val_log_dict = {
            "val_loss": val_loss
        }
        self.log_dict(
            val_log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        
        
        batch_idx = args[0]
        
        # Plot some example predictions using prior and encoder
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.current_epoch % 10 == 0
            and self.wandb_project is not None
        ):
            self.load_metrics_and_plots(D_yn, img_clean, batch_idx, mask=None)
            
            
            
    def load_metrics_and_plots(self, prediction, high_res, batch_idx, mask=None):
        
        #reshap from (B, C, H, W) to (B, num_grid_nodes, C)
        prediction = prediction.permute(0, 2, 3, 1).flatten(1, 2)
        high_res = high_res.permute(0, 2, 3, 1).flatten(1, 2)
        
        if mask is None:
            mask = torch.ones_like(high_res[:, :, 0])
        
        # Plot samples
        log_plot_dict = {}

        var_i = random.randint(0, len(constants.PARAM_NAMES_SHORT_CERRA) - 1)
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]
        
        sample = random.randint(0, prediction.shape[0] - 1)

        pred_states = prediction[
            sample, :, var_i
        ]  # (S, num_grid_nodes)
        
        target_state = high_res[
            sample, :, var_i
        ]  # (num_grid_nodes,)

        plot_title = (
            f"{var_name} ({var_unit})"
        )

        # Make plots
        log_plot_dict[
            f"pred_{var_name}"
        ] = vis.plot_ensemble_prediction(
            pred_states,
            target_state,
            obs_mask = mask[sample],
            title=f"{plot_title} (prior)",
        )

        if not self.trainer.sanity_checking:
            # Log all plots to wandb
            wandb.log(log_plot_dict)

        plt.close("all")   
        
        
    def plot_preds(self, prediction, high_res, img_lr, diz_stats):
        """
        Plot one random sample from the batch (single variable):
        [low‐res input, high‐res target, prediction, residual].
        Here, `prediction` and `high_res` are already un‐normalized. We only
        need to un‐normalize `img_lr` for plotting.
        """
        # If you need statistics to un‐normalize img_lr for plotting:
        low_res_mean = diz_stats["mean_era5"]
        low_res_std  = diz_stats["std_era5"]

        # Un‐normalize img_lr before plotting
        img_lr = img_lr * low_res_std + low_res_mean

        # Select a random sample index and a random variable/channel index
        sample_idx = random.randint(0, prediction.shape[0] - 1)
        var_i      = random.randint(0, prediction.shape[1] - 1)

        # Variable names and units (must match your constants)
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]

        # Extract 2D images for plotting (B, C, H, W → (H, W))
        input_img   = img_lr[sample_idx, var_i, :, :].detach().cpu().numpy()
        target_img  = high_res[sample_idx, var_i, :, :].detach().cpu().numpy()
        pred_img    = prediction[sample_idx, var_i, :, :].detach().cpu().numpy()
        residual_img = target_img - pred_img

        # Build a 1×4 subplot (Input, Target, Prediction, Residual)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(input_img,   cmap='plasma', origin='lower')
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(target_img,  cmap='plasma', origin='lower')
        axes[1].set_title("Target")
        axes[1].axis("off")

        axes[2].imshow(pred_img,    cmap='plasma', origin='lower')
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        axes[3].imshow(residual_img, cmap='plasma', origin='lower')
        axes[3].set_title("Residual")
        axes[3].axis("off")

        # Overall title shows variable name and unit
        fig.suptitle(f"{var_name} ({var_unit})", fontsize=16)

        # Save the figure (e.g. into "plot_tests/")
        save_dir = self.savepreds_path + "/" + self.load.split("/")[-2] + "/pred_plots"
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{var_name}_sample_{sample_idx}.png")
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
    