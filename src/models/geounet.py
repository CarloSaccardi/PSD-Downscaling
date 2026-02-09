import torch
import pytorch_lightning as pl
import importlib
import wandb
import matplotlib.pyplot as plt
from src.utils import constants
from src.utils import vis
import random
import os
import torch.nn.functional as F
from src.models.swin import SwinV2Wrapper
from argparse import Namespace
from src.data.dataset import MaskGenerator
from typing import List


network_module = importlib.import_module("physicsnemo.models.diffusion")




class GeoUNetWrapper(pl.LightningModule):
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
        self.swin_pretrained_checkpoint = args.swin_pretrained_checkpoint
        ### Generate fixed mask array filled with zeros
        self.mask_generator = MaskGenerator(
            input_size=args.cond_size[0],
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        )
        # Always keep a zero mask (no masking) and expand per batch at runtime.
        zero_mask = torch.zeros_like(torch.from_numpy(self.mask_generator())).float()
        self.register_buffer("zero_mask_base", zero_mask, persistent=False)
        #########################################################
        
        ### Load pretrained SwinV2 model
        if self.swin_pretrained_checkpoint is not None:
            self.swin_pretrained = SwinV2Wrapper.load_from_checkpoint(args.swin_pretrained_checkpoint, args=self.create_args(args))
            for param in self.swin_pretrained.parameters():
                param.requires_grad = False
            self.swin_pretrained.eval()
        #########################################################
        
        self.model_kwargs = {
            'checkpoint_level': args.checkpoint_level,
            # 'N_grid_channels': args.N_grid_channels,
            'swin_pretrained_checkpoint': args.swin_pretrained_checkpoint,
            'embedding_type': args.embedding_type,
            'model_channels': args.model_channels,
            'channel_mult': args.channel_mult,
            'attn_resolutions': args.attn_resolutions,
        }

        model_class = getattr(network_module, args.model_type)
        self.model = model_class(
            img_resolution=args.img_resolution,
            in_channels=args.img_in_channels, #+ args.N_grid_channels,
            out_channels=args.img_out_channels,
            **self.model_kwargs,
        )
        
        
    def create_args(self, conf):
        """Create args object with necessary parameters for model loading."""
        args = Namespace()
        args.img_in_channels = self.img_in_channels
        args.img_out_channels = self.img_out_channels
        args.img_size = conf.cond_size
        args.swin_v2_variant = conf.swin_v2_variant
        args.window_size = conf.window_size
        args.drop_rate = conf.drop_rate
        args.attn_drop_rate = conf.attn_drop_rate
        args.drop_path_rate = conf.drop_path_rate
        args.use_light_decoder = conf.use_light_decoder
        args.wandb_project = conf.wandb_project
        args.lr = conf.lr
        args.batch_size = conf.batch_size
        args.n_workers = conf.n_workers
        return args
        

    def forward(
        self,
        x: torch.Tensor,
        conditions: List[torch.Tensor],
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """
        Forward pass of the UNet wrapper model.
        """

        D_x = self.model(
            x, 
            conditions,
            torch.zeros(x.shape[0], device=x.device),  
            None,
            **model_kwargs,
        )
        return D_x.to(torch.float32)

    def training_step(self, batch, *args):
        era5, cerra, conditions = batch
        
        if self.swin_pretrained_checkpoint is not None:
            zero_mask = self.zero_mask_base.to(device=conditions.device, dtype=conditions.dtype)
            zero_mask = zero_mask.unsqueeze(0).expand(conditions.shape[0], -1, -1)
            features_list, _ = self.swin_pretrained(conditions, zero_mask)
        else:
            features_list = None
            
        D_x = self(era5, features_list)
        
        loss = F.mse_loss(D_x, cerra)
        
        log_dict = {
            "train_loss_epoch": loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, *args):        
        era5, cerra, conditions = batch
        
        if self.swin_pretrained_checkpoint is not None:
            zero_mask = self.zero_mask_base.to(device=conditions.device, dtype=conditions.dtype)
            zero_mask = zero_mask.unsqueeze(0).expand(conditions.shape[0], -1, -1)
            features_list, _ = self.swin_pretrained(conditions, zero_mask)
        else:
            features_list = None
            
        D_x = self(era5, features_list)
        
        val_loss = F.mse_loss(D_x, cerra)
        
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
            self.load_metrics_and_plots(D_x, cerra, mask=None)
    
    
    def test_step(self, batch, batch_idx):
        # batch size is 1. Shapes: 
        # era5_patches: [1, 16, C, 96, 96]
        # cerra_patches: [1, 16, C, 96, 96]
        # conditions: [1, C_cond, 384, 384]
        era5_patches, cerra_patches, conditions = batch
        
        era5_patches = era5_patches.squeeze(0)   # [16, C, 96, 96]
        cerra_patches = cerra_patches.squeeze(0) # [16, C, 96, 96]

        # 1. Process Global Context (Swin) once per image
        if self.swin_pretrained_checkpoint is not None:
            zero_mask = self.zero_mask_base.to(device=conditions.device)
            zero_mask = zero_mask.unsqueeze(0).expand(conditions.shape[0], -1, -1)
            features_list, _ = self.swin_pretrained(conditions, zero_mask)
        else:
            features_list = None

        # 2. Patch-wise Inference
        outputs = []
        for i in range(era5_patches.size(0)):
            # Model forward pass for one patch
            patch_out = self(era5_patches[i:i+1], features_list)
            outputs.append(patch_out)
        
        # 3. Reassemble the patches into the full 384x384 grid
        # Cat the list of [1, C, 96, 96] -> [16, C, 96, 96]
        full_pred = self.reassemble(torch.cat(outputs, dim=0))
        full_target = self.reassemble(cerra_patches)

        # 4. Compute metrics on the FULL RECONSTRUCTED image
        mse = F.mse_loss(full_pred, full_target)
        mae = F.l1_loss(full_pred, full_target)

        # 5. Log as before (Lightning handles the epoch-end aggregation automatically)
        self.log("test_mse", mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_mae", mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return {"test_mse": mse.detach(), "test_mae": mae.detach()}
    
    
    def reassemble(self, patches):
        """
        Input: [16, C, 96, 96]
        Output: [1, C, 384, 384]
        """
        C = patches.shape[1]
        P = 96  # Patch size
        G = 4   # Grid size (384/96)
        
        # 1. Reshape to grid: [4, 4, C, 96, 96]
        out = patches.view(G, G, C, P, P)
        # 2. Permute to align dimensions: [C, 4, 96, 4, 96]
        out = out.permute(2, 0, 3, 1, 4).contiguous()
        # 3. Combine grid and patch dims: [1, C, 384, 384]
        return out.view(1, C, G * P, G * P)
    

    def on_test_epoch_end(self):
        # 1. Concatenate all stored full-image samples
        # Resulting shape: [Total_Samples, C, 384, 384]
        all_preds = torch.cat([x["pred"] for x in self.test_step_outputs], dim=0)
        all_targets = torch.cat([x["target"] for x in self.test_step_outputs], dim=0)

        # 2. Compute Global Metrics
        # Example: Global RMSE
        global_mse = F.mse_loss(all_preds, all_targets)
        global_rmse = torch.sqrt(global_mse)

        # Example: Bias (Mean Error)
        global_bias = torch.mean(all_preds - all_targets)

        # 3. Logging
        self.log("test_global_rmse", global_rmse, sync_dist=True)
        self.log("test_global_bias", global_bias, sync_dist=True)

        # 4. Clear memory for next time
        self.test_step_outputs.clear()
    
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
            
            
    def load_metrics_and_plots(self, prediction, target, mask):
        # Reshape from (B, C, H, W) to (B, num_grid_nodes, C)
        prediction = prediction.permute(0, 2, 3, 1).flatten(1, 2)
        target = target.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Plot samples
        log_plot_dict = {}
        
        var_i = random.randint(0, len(constants.PARAM_NAMES_SHORT_CERRA) - 1)
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]
        
        sample = random.randint(0, prediction.shape[0] - 1)
        
        #select one mask grid
        mask_grid = mask[sample, :, :].flatten(1,2).squeeze(0) if mask is not None else None
        
        
        pred_states = prediction[sample, :, var_i]
        target_state = target[sample, :, var_i]
        
        plot_title = f"{var_name} ({var_unit})"
        
        # Make plots
        log_plot_dict[f"pred_{var_name}"] = vis.plot_ensemble_prediction(
            pred_states,
            target_state,
            mask=mask_grid,
            title=f"{plot_title} (prior)",
        )
        
        if not self.trainer.sanity_checking:
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
    