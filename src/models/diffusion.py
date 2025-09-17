import importlib
from physicsnemo.utils.patching import RandomPatching2D
from physicsnemo.models.diffusion import EDMPrecond
from typing import Callable, Optional, List, Tuple, Union

from torch import Tensor
import nvtx
import torch
from .unet import UNetWrapper
import pytorch_lightning as pl
from ..utils.utils import stochastic_sampler, diffusion_step
from functools import partial

from .. import constants
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
        if args.hr_mean_conditioning:
            self.img_in_channels += self.img_out_channels
        self.sigma_data: float = 0.5
        self.sigma_min=0.0
        self.sigma_max=float("inf")
        self.savepreds_path = args.savepreds_path
        self.load = args.load
        
        self.lr = args.lr
        
        self.regression_net = UNetWrapper.load_from_checkpoint( args.regression_net, args=args)
            
        if args.restore_opt:
            # Save for later
            # Unclear if this works for multi-GPU
            self.regression_net.opt_state = torch.load(args.regression_net)["optimizer_states"][0]
        
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
        
        self.loss_fn = ResidualLoss(
            regression_net=self.regression_net,
            hr_mean_conditioning=args.hr_mean_conditioning
        )
        
    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float, List, torch.Tensor]
            Sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            Tensor representation of sigma values.

        See Also
        --------
        EDMPrecond.round_sigma
        """
        return EDMPrecond.round_sigma(sigma)

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

        if img_lr is None:
            arg = c_in * x
        else:
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
        batch_size = batch[0].shape[0]
        img_clean, img_lr, *rest = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        loss_map = self.loss_fn(
                            net=self,
                            img_clean=img_clean,
                            img_lr=img_lr
                        )
        
        train_loss = loss_map.sum() / batch_size
        
        log_dict = {
            "train_loss": train_loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return train_loss

    def validation_step(self, batch, *args):
        batch_size = batch[0].shape[0]
        img_clean, img_lr = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        loss_map = self.loss_fn(
                            net=self,
                            img_clean=img_clean,
                            img_lr=img_lr
                        )
        
        val_loss = loss_map.sum() / batch_size
        
        # Log loss per time step forward and mean
        val_log_dict = {
            "val_loss": val_loss
        }
        self.log_dict(
            val_log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=[0.9, 0.999], eps=1e-8
            )
        return opt
    
    def test_step(self, batch, *args):
        batch_size = batch[0].shape[0]
        img_clean, img_lr, diz_stats, img_lr_name = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        
        y_mean = self.regression_net(
                torch.zeros_like(img_clean, device=img_clean.device),
                img_lr,
                augment_labels=None,
            )
        
        sampler_fn = partial(stochastic_sampler, patching=None)
        
        # === prepare the seed batches exactly like before ===
        seeds = torch.arange(32, device=self.device)
        num_batches = (
            (len(seeds) - 1) // (batch_size * self.trainer.world_size) + 1
        ) * self.trainer.world_size
        rank_batches = seeds.tensor_split(num_batches)[self.global_rank::self.trainer.world_size]
        
        y_res = diffusion_step(
                            net=self,
                            sampler_fn=sampler_fn,
                            img_shape=tuple(self.img_resolution),
                            img_out_channels=self.img_out_channels,
                            rank_batches=rank_batches,
                            img_lr=img_lr.expand(
                                batch_size, -1, -1, -1
                            ).to(memory_format=torch.channels_last),
                            rank=self.global_rank,
                            device=self.device,
                            mean_hr=y_mean,
                            lead_time_label=None,
                        )
        
        y_res_avg = y_res.mean(dim=0, keepdim=True)
        predictions =  y_res_avg + y_mean
        ground_truth = img_clean
        
        # (2) Un‐normalize both `predictions` and `ground_truth` at once,
        #     so that all metrics and saved files are on the original scale.
        high_res_mean = diz_stats["mean_CERRA"]
        high_res_std  = diz_stats["std_CERRA"]
        # Assuming `ground_truth` was normalized the same way as `prediction`:
        predictions  = predictions * high_res_std + high_res_mean
        ground_truth = ground_truth * high_res_std + high_res_mean

        # (3) If requested, save each sample’s un‐normalized prediction as a .npy file.
        #     We'll save into a folder called "predictions" (create if needed),
        #     and name each file using img_lr_name[i] + ".npy".
        if self.savepreds_path:
            
            savepath = self.savepreds_path + "/" + self.load.split("/")[-2] 
            
            os.makedirs(savepath, exist_ok=True)
            # predictions: Tensor of shape (B, C, H, W) after un‐normalization
            preds_cpu = predictions.detach().cpu().numpy()
            for i in range(batch_size):
                base_name = img_lr_name[i]
                out_path = os.path.join(savepath, f"nwp_{base_name}")
                # Save the multi‐channel array as-is
                # (so downstream you can load with np.load and get shape (C, H, W)).
                np.save(out_path, preds_cpu[i])
                # Alternatively, if you prefer numpy.save:
                # np.save(out_path, preds_cpu[i])
        
        # Overall metrics across all variables.
        mse_all = torch.mean((predictions - ground_truth) ** 2)
        mae_all = torch.mean(torch.abs(predictions - ground_truth))
        rmse_all = torch.sqrt(mse_all)
        
        overall_data_range = (ground_truth.max() - ground_truth.min()).item()
        ssim_all = ssim_func(predictions, ground_truth, data_range=overall_data_range)
        
        # Define variable names in order.
        var_names = ['u10', 'v10', 't2m', 'sshf', 'zust']
        
        mse_vars = {}
        mae_vars = {}
        rmse_vars = {}
        ssim_vars = {}
        for i, var_name in enumerate(var_names):
            # Extract the i-th variable (shape: (B, H, W)).
            pred_i = predictions[:, i, :, :]
            gt_i = ground_truth[:, i, :, :]
            
            mse_val = torch.mean((pred_i - gt_i) ** 2)
            mse_vars[f"test_mse_{var_name}"] = mse_val
            mae_vars[f"test_mae_{var_name}"] = torch.mean(torch.abs(pred_i - gt_i))
            rmse_vars[f"test_rmse_{var_name}"] = torch.sqrt(mse_val)
            
            # SSIM expects the input to be (B, C, H, W); for a single channel, unsqueeze.
            data_range_i = (gt_i.max() - gt_i.min()).item()
            ssim_vars[f"test_ssim_{var_name}"] = ssim_func(pred_i.unsqueeze(1),
                                                            gt_i.unsqueeze(1),
                                                            data_range=data_range_i)
        
        # Combine all metrics.
        log_metrics = {
            "test_mse": mse_all,
            "test_mae": mae_all,
            "test_rmse": rmse_all,
            "test_ssim": ssim_all,
        }
        log_metrics.update(mse_vars)
        log_metrics.update(mae_vars)
        log_metrics.update(rmse_vars)
        log_metrics.update(ssim_vars)
        
        self.log_dict(log_metrics, prog_bar=False, on_epoch=True, sync_dist=True)
        
        self.plot_preds(predictions, ground_truth, img_lr, diz_stats)
        
        return log_metrics
    
    def plot_preds(self, prediction, high_res, img_lr, diz_stats):
        """
        Plot a random sample for a random variable from the batch. The figure includes 
        the low resolution input, target (high_res), prediction, and residual.
        The overall figure title indicates the variable name, and the plot is saved.
        """
        
        # If you need statistics to un‐normalize img_lr for plotting:
        low_res_mean = diz_stats["mean_era5"]
        low_res_std  = diz_stats["std_era5"]

        # Un‐normalize img_lr before plotting
        img_lr = img_lr * low_res_std + low_res_mean

        # Select a random sample from the batch and a random variable index.
        sample = random.randint(0, prediction.shape[0] - 1)
        var_i = random.randint(0, prediction.shape[1] - 1)
        
        # Retrieve variable name and unit from constants.
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]
        
        # Extract the images for the selected variable and sample.
        input_img = img_lr[sample, var_i, :, :].detach().cpu().numpy()
        target_img = high_res[sample, var_i, :, :].detach().cpu().numpy()
        pred_img   = prediction[sample, var_i, :, :].detach().cpu().numpy()
        residual_img = target_img - pred_img
        
        # Create a plot with four subplots.
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(input_img, cmap='plasma', origin='lower')
        axes[0].set_title("Input")
        axes[0].axis("off")
        
        axes[1].imshow(target_img, cmap='plasma', origin='lower')
        axes[1].set_title("Target")
        axes[1].axis("off")
        
        axes[2].imshow(pred_img, cmap='plasma', origin='lower')
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        
        axes[3].imshow(residual_img, cmap='plasma', origin='lower')
        axes[3].set_title("Residual")
        axes[3].axis("off")
        
        # Set overall title with variable name and unit.
        fig.suptitle(f"{var_name} ({var_unit})", fontsize=16)
        
        # Save plot to a given folder.
        save_dir = "plot_tests"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{var_name}_sample_{sample}.png")
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        
class ResidualLoss:
    """
    Mixture loss function for denoising score matching.

    This class implements a loss function that combines deterministic
    regression with denoising score matching. It uses a pre-trained regression
    network to compute residuals before applying the diffusion process.

    Attributes
    ----------
    regression_net : torch.nn.Module
        The regression network used for computing residuals.
    P_mean : float
        Mean value for noise level computation.
    P_std : float
        Standard deviation for noise level computation.
    sigma_data : float
        Standard deviation for data weighting.
    hr_mean_conditioning : bool
        Flag indicating whether to use high-resolution mean for conditioning.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C., Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric
    Downscaling. arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
    ):
        """
        Arguments
        ----------
        regression_net : torch.nn.Module
            Pre-trained regression network used to compute residuals.
            Expected signature: `net(zero_input, y_lr,
            lead_time_label=lead_time_label, augment_labels=augment_labels)` or
            `net(zero_input, y_lr, augment_labels=augment_labels)`, where:
                zero_input (torch.Tensor): Zero tensor of shape (B, C_hr, H, W)
                y_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                lead_time_label (torch.Tensor, optional): Optional lead time labels
                augment_labels (torch.Tensor, optional): Optional augmentation labels
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        P_mean : float, optional
            Mean value for noise level computation, by default 0.0.

        P_std : float, optional
            Standard deviation for noise level computation, by default 1.2.

        sigma_data : float, optional
            Standard deviation for data weighting, by default 0.5.

        hr_mean_conditioning : bool, optional
            Whether to use high-resolution mean for conditioning predicted, by default False.
            When True, the mean prediction from `regression_net` is channel-wise
            concatenated with `img_lr` for conditioning.
        """
        self.regression_net = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hr_mean_conditioning = hr_mean_conditioning

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
    ) -> Tensor:
        """
        Calculate and return the loss for denoising score matching.

        This method computes a mixture loss that combines deterministic
        regression with denoising score matching. It first computes residuals
        using the regression network, then applies the diffusion process to
        these residuals.

        In addition to the standard denoising score matching loss, this method
        also supports optional patching for multi-diffusion. In this case, the spatial
        dimensions of the input are decomposed into `P` smaller patches of shape
        (H_patch, W_patch), that are grouped along the batch dimension, and the
        model is applied to each patch individually. In the following, if `patching`
        is not provided, then the input is not patched and `P=1` and `(H_patch,
        W_patch) = (H, W)`. When patching is used, the original non-patched conditioning is
        interpolated onto a spatial grid of shape `(H_patch, W_patch)` and channel-wise
        concatenated to the patched conditioning. This ensures that each patch
        maintains global information from the entire domain.

        The diffusion model `net` is expected to be conditioned on an input with
        `C_cond` channels, which should be:
            - `C_cond = C_lr` if `hr_mean_conditioning` is `False` and
              `patching` is None.
            - `C_cond = C_hr + C_lr` if `hr_mean_conditioning` is `True` and
              `patching` is None.
            - `C_cond = C_hr + 2*C_lr` if `hr_mean_conditioning` is `True` and
              `patching` is not None.
            - `C_cond = 2*C_lr` if `hr_mean_conditioning` is `False` and
              `patching` is not None.
        Additionally, `C_cond` should also include any embedding channels,
        such as positional embeddings or time embeddings.

        Note: this loss function does not apply any reduction.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model for the diffusion process.
            Expected signature: `net(latent, y_lr, sigma,
            embedding_selector=embedding_selector, lead_time_label=lead_time_label,
            augment_labels=augment_labels)`, where:
                latent (torch.Tensor): Noisy input of shape (B[*P], C_hr, H_patch, W_patch)
                y_lr (torch.Tensor): Conditioning of shape (B[*P], C_cond, H_patch, W_patch)
                sigma (torch.Tensor): Noise level of shape (B[*P], 1, 1, 1)
                embedding_selector (callable, optional): Function to select
                    positional embeddings. Only used if `patching` is provided.
                lead_time_label (torch.Tensor, optional): Lead time labels.
                augment_labels (torch.Tensor, optional): Augmentation labels
            Returns:
                torch.Tensor: Predictions of shape (B[*P], C_hr, H_patch, W_patch)

        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used as ground truth and for data augmentation if 'augment_pipe' is provided.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the regression network and conditioning for the
            diffusion process.

        patching : Optional[RandomPatching2D], optional
            Patching strategy for processing large images, by default None. See
            :class:`physicsnemo.utils.patching.RandomPatching2D` for details.
            When provided, the patching strategy is used for both image patches
            and positional embeddings selection in the diffusion model `net`.
            Transforms tensors from shape (B, C, H, W) to (B*P, C, H_patch,
            W_patch).

        lead_time_label : Optional[torch.Tensor], optional
            Labels for lead-time aware predictions, by default None.
            Shape can vary based on model requirements, typically (B,) or scalar.

        augment_pipe : Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]]
            Data augmentation function.
            Expected signature:
                img_tot (torch.Tensor): Concatenated high and low resolution images
                    of shape (B, C_hr+C_lr, H, W)
            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Augmented images of shape (B, C_hr+C_lr, H, W)
                    - Optional augmentation labels

        Returns
        -------
        torch.Tensor
            If patching is not used:
                A tensor of shape (B, C_hr, H, W) representing the per-sample loss.
            If patching is used:
                A tensor of shape (B*P, C_hr, H_patch, W_patch) representing
                the per-patch loss.

        Raises
        ------
        ValueError
            If patching is provided but is not an instance of RandomPatching2D.
            If shapes of img_clean and img_lr are incompatible.
        """

        # Safety check: enforce patching object
        if patching and not isinstance(patching, RandomPatching2D):
            raise ValueError("patching must be a 'RandomPatching2D' object.")
        # Safety check: enforce shapes
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]

        # form residual
        if lead_time_label is not None:
            y_mean = self.regression_net(
                torch.zeros_like(y, device=img_clean.device),
                y_lr_res,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            y_mean = self.regression_net(
                torch.zeros_like(y, device=img_clean.device),
                y_lr_res,
                augment_labels=augment_labels,
            )

        y = y - y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((y_mean, y_lr), dim=1).contiguous()

        # patchified training
        # conditioning: cat(y_mean, y_lr, input_interp, pos_embd), 4+12+100+4
        if patching:
            # Patched residual
            # (batch_size * patch_num, c_out, patch_shape_y, patch_shape_x)
            y_patched = patching.apply(input=y)
            # Patched conditioning on y_lr and interp(img_lr)
            # (batch_size * patch_num, 2*c_in, patch_shape_y, patch_shape_x)
            y_lr_patched = patching.apply(input=y_lr, additional_input=img_lr)

            # Function to select the correct positional embedding for each
            # patch
            def patch_embedding_selector(emb):
                # emb: (N_pe, image_shape_y, image_shape_x)
                # return: (batch_size * patch_num, N_pe, patch_shape_y, patch_shape_x)
                return patching.apply(emb[None].expand(batch_size, -1, -1, -1))

            y = y_patched
            y_lr = y_lr_patched
        else:
            patch_embedding_selector = None

        # Noise
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Input + noise
        latent = y + torch.randn_like(y) * sigma

        if lead_time_label is not None:
            D_yn = net(
                latent,
                y_lr,
                sigma,
                embedding_selector=patch_embedding_selector,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                latent,
                y_lr,
                sigma,
                embedding_selector=patch_embedding_selector,
                augment_labels=augment_labels,
            )
        loss = weight * ((D_yn - y) ** 2)

        return loss
