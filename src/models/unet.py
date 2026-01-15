import torch
import pytorch_lightning as pl
import importlib
import wandb
import matplotlib.pyplot as plt
from .. import constants
from ..utils import vis
import random
import os
import torch.nn.functional as F


network_module = importlib.import_module("physicsnemo.models.diffusion")




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
        

    def forward(
        self,
        condition: torch.Tensor,
        x: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """
        Forward pass of the UNet wrapper model.

        This method concatenates the input tensor with the low-resolution conditioning tensor
        and passes the result through the underlying model.

        Parameters
        ----------
        condition : torch.Tensor
            The conditioning tensor, typically zero-filled, of shape (B, C_hr, H, W).
        x : torch.Tensor
            Low-resolution input image of shape (B, C_lr, H, W).
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
        if x is not None:
            x = torch.cat((condition, x), dim=1)


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
        target, x, *rest = batch
        target = target.float()
        x = x.float()
        zero_input = torch.zeros_like(target, device=target.device)
        D_yn = self(zero_input, x, force_fp32=False)
        loss = F.mse_loss(D_yn, target)
        
        log_dict = {
            "train_loss_epoch": loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, *args):
        
        target, x, *rest = batch
        target = target.float()
        x = x.float()
        zero_input = torch.zeros_like(target, device=target.device)
        D_yn = self(zero_input, x, force_fp32=False)
        val_loss = F.mse_loss(D_yn, target)
        
        # Log loss per time step forward and mean
        val_log_dict = {
            "val_loss": val_loss,
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
            self.load_metrics_and_plots(D_yn, target, batch_idx, mask=None)
    
    
    def test_step(self, batch, batch_idx):
        # compute forward pass, compute MSE, MAE, plot PSD, and plot some examples
        target, x = batch 
        zero_input = torch.zeros_like(target, device=target.device)
        D_yn = self(zero_input, x, force_fp32=False)
        mse = F.mse_loss(D_yn, target)
        mae = F.l1_loss(D_yn, target)
        self.log("test_mse", mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_mae", mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"test_mse": mse.detach(), "test_mae": mae.detach()}

    def on_test_epoch_end(self):
        """Print aggregated test metrics (already reduced by Lightning)."""
        if self.trainer.is_global_zero:
            mse = self.trainer.callback_metrics.get("test_mse", None)
            mae = self.trainer.callback_metrics.get("test_mae", None)
            if mse is not None and mae is not None:
                print(f"Test MSE (epoch): {mse:.4f}, Test MAE (epoch): {mae:.4f}")
    
    
    
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
    