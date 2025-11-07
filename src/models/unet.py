import torch
import pytorch_lightning as pl
import importlib
import wandb
import matplotlib.pyplot as plt
from .. import constants
from neural_lam import vis  # Keep this for now, will move later
from typing import Callable, Optional, Tuple
import random
from torchmetrics.functional import structural_similarity_index_measure as ssim_func
import os
import numpy as np
from scipy.fft import fft
import math

from .losses.fourier_losses import FourierLossETH, FourierLossDelft, FourierLossHK, FourierLossCarlo

network_module = importlib.import_module("physicsnemo.physicsnemo.models.diffusion")




class UNetWrapper(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        
        # for compatibility with older versions that took only 1 dimension
        if isinstance(args.img_resolution, int):
            self.img_shape_x = self.img_shape_y = args.img_resolution
        else:
            self.img_shape_y = args.img_resolution[0]
            self.img_shape_x = args.img_resolution[1]

        self.img_in_channels = args.img_in_channels
        self.img_out_channels = args.img_out_channels
        self.lr = args.lr
        self.wandb_project = args.wandb_project
        self.savepreds_path = args.savepreds_path
        self.load = args.load
        
        self.model_kwargs = {
            'checkpoint_level': args.checkpoint_level,
            'N_grid_channels': args.N_grid_channels,
            'embedding_type': args.embedding_type,
            'model_channels': args.model_channels,
            'channel_mult': args.channel_mult,
            'attn_resolutions': args.attn_resolutions,
        }

        model_class = getattr(network_module, args.model_type)
        self.model = model_class(
            img_resolution=args.img_resolution,
            in_channels=args.img_in_channels + args.N_grid_channels + args.img_out_channels,
            out_channels=args.img_out_channels,
            **self.model_kwargs,
        )
        
        loss_ditc = {
            "FourierLossETH": FourierLossETH,
            "FourierLossDelft": FourierLossDelft,
            "FourierLossHK": FourierLossHK,
            "FourierLossCarlo": FourierLossCarlo
        }
        
        loss_func = loss_ditc[args.loss_type]() if args.loss_type in loss_ditc else None

        self.loss_fn = RegressionLoss(args.init_lambda, args.max_lambda, args.anneal_epochs, loss_func)

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """
        Forward pass of the UNet wrapper model.

        This method concatenates the input tensor with the low-resolution conditioning tensor
        and passes the result through the underlying model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor, typically zero-filled, of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution conditioning image of shape (B, C_lr, H, W).
        force_fp32 : bool, optional
            Whether to force FP32 precision regardless of the `use_fp16` attribute,
            by default False.
        **model_kwargs : dict
            Additional keyword arguments to pass to the underlying model
            `self.model` forward method.

        Returns
        -------
        torch.Tensor
            Output tensor (prediction) of shape (B, C_hr, H, W).

        Raises
        ------
        ValueError
            If the model output dtype doesn't match the expected dtype.
        """
        # SR: concatenate input channels
        if img_lr is not None:
            x = torch.cat((x, img_lr), dim=1)


        F_x = self.model(
            x,  # (c_in * x).to(dtype),
            torch.zeros(x.shape[0], device=x.device),  # c_noise.flatten()
            class_labels=None,
            **model_kwargs,
        )

        # skip connection
        D_x = F_x.to(torch.float32)
        return D_x

    def training_step(self, batch, *args):
        batch_size = batch[0].shape[0]
        img_clean, img_lr, *rest = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        loss, _, _, loss_space, loss_amp = self.loss_fn(
                            net=self,
                            img_clean=img_clean,
                            img_lr=img_lr,
                            current_epoch=self.current_epoch,
                        )
        
        log_dict = {
            "train_loss": loss,
            "train_loss_mse": loss_space,
            "train_loss_psd": loss_amp,
        }
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, *args):
        batch_size = batch[0].shape[0]
        img_clean, img_lr = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        val_loss, ground_truth, predictions, loss_space, loss_amp = self.loss_fn(
                                                net=self,
                                                img_clean=img_clean,
                                                img_lr=img_lr,
                                                current_epoch=self.current_epoch
                                            )
        
        # Log loss per time step forward and mean
        val_log_dict = {
            "val_loss": val_loss,
            "val_loss_mse": loss_space,
            "val_loss_psd": loss_amp,
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
            self.load_metrics_and_plots(predictions, ground_truth, batch_idx, mask=None)
    
    
    def test_step(self, batch, batch_idx: int) -> dict:
        """
        Evaluate model on a test batch, save un‐normalized predictions if requested,
        and compute metrics on the un‐normalized data.
        """
        # Unpack batch
        # img_clean: (B, C, H, W)
        # img_lr:    (B, C, H, W)
        # diz_stats: dict with keys "mean_CERRA", "std_CERRA", "mean_era5", "std_era5"
        # img_lr_name: tuple of length B, each entry is a string (base name for saving)
        img_clean, img_lr, diz_stats, img_lr_name = batch

        batch_size = img_clean.shape[0]
        img_clean = img_clean.float()
        img_lr    = img_lr.float()

        # (1) Get raw predictions & ground_truth from your loss_fn
        #     They are assumed to be in normalized space: shape (B, C, H, W)
        _, ground_truth, predictions, _, _ = self.loss_fn(
            net=self,
            img_clean=img_clean,
            img_lr=img_lr,
            current_epoch=self.current_epoch
        )

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
            
            savepath = self.savepreds_path + "/" + self.load.split("/")[-2] + "/files"
            
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

        # (4) Compute all global metrics on the un‐normalized tensors:
        mse_all  = torch.mean((predictions - ground_truth) ** 2)
        mae_all  = torch.mean(torch.abs(predictions - ground_truth))
        rmse_all = torch.sqrt(mse_all)

        # For SSIM, we need the data_range. Since we already un‐normalized:
        data_range = (ground_truth.max() - ground_truth.min()).item()
        ssim_all  = ssim_func(predictions, ground_truth, data_range=data_range)

        # (5) Compute per‐variable metrics. Here var_names must match the channel order.
        var_names = ['u10', 'v10', 't2m', 'sshf', 'zust']
        mse_vars  = {}
        mae_vars  = {}
        rmse_vars = {}
        ssim_vars = {}

        for i, var_name in enumerate(var_names):
            pred_i = predictions[:, i, :, :]  # shape (B, H, W)
            gt_i   = ground_truth[:, i, :, :]

            mse_val        = torch.mean((pred_i - gt_i) ** 2)
            mse_vars[f"test_mse_{var_name}"] = mse_val
            mae_vars[f"test_mae_{var_name}"] = torch.mean(torch.abs(pred_i - gt_i))
            rmse_vars[f"test_rmse_{var_name}"] = torch.sqrt(mse_val)

            # SSIM for single‐channel: add a dummy channel dim
            data_range_i = (gt_i.max() - gt_i.min()).item()
            ssim_vars[f"test_ssim_{var_name}"] = ssim_func(
                pred_i.unsqueeze(1),
                gt_i.unsqueeze(1),
                data_range=data_range_i
            )

        # (6) Log everything
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

        # Write to Lightning’s logger (and optionally sync across GPUs)
        self.log_dict(log_metrics, prog_bar=False, on_epoch=True, sync_dist=True)

        # (7) Pass un‐normalized predictions & high_res into the plotting function.
        #     Note: plot_preds now assumes its inputs are already un‐normalized.
        self.plot_preds(predictions, ground_truth, img_lr, diz_stats)

        return log_metrics
    
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
            
            
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
    
    
class RegressionLoss:
    """
    Regression loss function for the deterministic predictions.
    Note: this loss does not apply any reduction.

    Attributes
    ----------
    sigma_data: float
        Standard deviation for data. Deprecated and ignored.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(self, 
                 init_lambda: float, 
                 max_lambda: float, 
                 anneal_epochs: int, 
                 loss_func: Optional[Callable] = None,
                 eps: float = 1e-12):
        
        self.init_lambda = init_lambda
        self.max_lambda = max_lambda
        self.anneal_epochs = anneal_epochs
        self.loss_func = loss_func
        self.eps = eps                         # to avoid log(0)
        
    
    # @staticmethod
    # def psd2d(a: torch.Tensor, *, dx: float, dy: float,
    #         eps: float = 1e-12) -> torch.Tensor:
    #     H, W = a.shape[-2:]
    #     fft   = torch.fft.rfftn(a, dim=(-2, -1)) / (H * W)
    #     psd   = 2.0 * (fft.real**2 + fft.imag**2)
    #     return psd.clamp_min(eps)     
    
    @staticmethod
    def get_psd_torch(a: torch.Tensor, *, dx: float, dim: int = -1
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        a   : (..., N)   real-valued signal
        dx  : grid spacing (same units as the data’s physical dimension)
        dim : dimension over which to take the FFT
        """
        N  = a.shape[dim]
        dk = 2 * math.pi / dx                      # angular‐wavenumber step
        k  = torch.arange(0, N // 2, device=a.device, dtype=a.dtype) * dk / N

        v_ft  = torch.fft.rfft(a, dim=dim)         # shape (..., N//2+1)
        v_ft  = v_ft.narrow(dim, 0, N // 2)        # drop Nyquist to match k
        psd   = (v_ft.real.square() + v_ft.imag.square()) / (N * dx)

        return k, psd  

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        current_epoch: int,
        augment_pipe: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]
        ] = None,
    ) -> torch.Tensor:
        """
        Calculate and return the regression loss for
        deterministic predictions.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model that will make predictions.
            Expected signature: `net(x, img_lr,
            augment_labels=augment_labels, force_fp32=False)`, where:
                x (torch.Tensor): Tensor of shape (B, C_hr, H, W). Is zero-filled.
                img_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                augment_labels (torch.Tensor, optional): Optional augmentation
                labels, returned by `augment_pipe`.
                force_fp32 (bool, optional): Whether to force the model to use
                fp32, by default False.
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used as ground truth and for data augmentation if 'augment_pipe' is provided.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the neural network.

        augment_pipe : callable, optional
            An optional data augmentation function.
            Expected signature:
                img_tot (torch.Tensor): Concatenated high and low resolution
                    images of shape (B, C_hr+C_lr, H, W)
            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Augmented images of shape (B, C_hr+C_lr, H, W)
                    - Optional augmentation labels

        Returns
        -------
        torch.Tensor
            A tensor representing the per-sample element-wise squared
            difference between the network's predictions and the high
            resolution images `img_clean` (possibly data-augmented by
            `augment_pipe`).
            Shape: (B, C_hr, H, W), same as `img_clean`.
        """

        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        zero_input = torch.zeros_like(y, device=img_clean.device)
        D_yn = net(zero_input, y_lr, force_fp32=False, augment_labels=augment_labels)
        # loss_mse = torch.mean((D_yn - y) ** 2)
        
        # ─── PSD loss ────────────────────────────────────────────────
        # psd_pred = self.psd2d(
        #     D_yn, dx=5.5, dy=5.5)
        # psd_true = self.psd2d(
        #     y,   dx=5.5, dy=5.5)
        
        # k, psd_pred = self.get_psd_torch(
        #     D_yn.permute(0, 2, 3, 1), dx=5.5, dim=2)
        # _, psd_true = self.get_psd_torch(
        #     y.permute(0, 2, 3, 1),   dx=5.5, dim=2)
        
        # freq_weights = (k / k.max()).pow(2)  # shape: (F,)
        # w = freq_weights.view(1, 1, -1, 1)  

        # RMSE between log-PSDs (scalar)
        # diff_log_psd = torch.log(psd_pred + self.eps) - torch.log(psd_true + self.eps)
        # psd_loss = torch.sqrt(torch.mean(w * diff_log_psd ** 2))
        # psd_loss = torch.sqrt(torch.mean(diff_log_psd ** 2))
        
        if self.loss_func is not None:
            term_space, term_amp = self.loss_func(D_yn, y)
            lambda_psd = min(self.init_lambda + (self.max_lambda - self.init_lambda) * (current_epoch / self.anneal_epochs)**2, self.max_lambda)
            loss_amp = lambda_psd * term_amp
            loss_space = term_space
            loss = loss_space + loss_amp
        else:
            loss = torch.mean((D_yn - y) ** 2)
            loss_space = loss #MSE
            loss_amp = torch.tensor(0.0, device=D_yn.device) #just zero
            

        return loss, y, D_yn, loss_space, loss_amp #, lambda_psd, lambda_psd * psd_loss
    