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
        self.eval_mode = args.eval
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
        # era5_patches: [B, 16, C, 96, 96]
        # cerra_patches: [B, 16, C, 96, 96]
        # conditions: [B, C_cond, 384, 384]
        era5_patches, cerra_patches, conditions = batch
        B = era5_patches.shape[0]

        # 1. Flatten patches: [B*16, C, 96, 96]
        # This allows the model to see B*16 as the batch size
        era5_flat = era5_patches.view(-1, *era5_patches.shape[2:]) 

        # 2. Process Global Context
        if self.swin_pretrained_checkpoint is not None:
            zero_mask = self.zero_mask_base.to(device=conditions.device)
            # Expand mask to current batch size B
            zero_mask = zero_mask.unsqueeze(0).expand(B, -1, -1)
            features_list, _ = self.swin_pretrained(conditions, zero_mask)
            features_list = [f.repeat_interleave(16, dim=0) for f in features_list]
        else:
            features_list = None

        # 3. Vectorized Inference (No more for-loop!)
        # era5_flat is [B*16, C, 96, 96]
        # Ensure your forward method can handle the expanded features_list
        full_pred_patches = self(era5_flat, features_list) 

        # 4. Reassemble for the whole batch
        full_pred = self.reassemble(full_pred_patches, B)
        full_target = self.reassemble(cerra_patches.view(-1, *cerra_patches.shape[2:]), B)
        
        # Log region-grouped images to a WandB table
        self.load_metrics_and_plots(full_pred, full_target, mask=None)

        # 5. Metrics
        mse = F.mse_loss(full_pred, full_target)
        mae = F.l1_loss(full_pred, full_target)

        # Log per-region metrics (aggregated across epoch by Lightning)
        region = getattr(self, "current_region", "unknown")
        self.log(f"test/{region}/mse", mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/{region}/mae", mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return {f"test/{region}/mse": mse.detach(), f"test/{region}/mae": mae.detach()}
    
    
    def reassemble(self, patches, B):
        """
        Input: [B*16, C, 96, 96]
        Output: [B, C, 384, 384]
        """
        C = patches.shape[1]
        P = 96  # Patch size
        G = 4   # Grid size (4x4 = 16 patches)
        
        # 1. Unflatten batch and grid: [B, G, G, C, 96, 96]
        out = patches.view(B, G, G, C, P, P)
        
        # 2. Permute: [B, C, G(row), P(height), G(col), P(width)]
        # Current: (0:B, 1:G_row, 2:G_col, 3:C, 4:P_h, 5:P_w)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        
        # 3. Combine: [B, C, 384, 384]
        return out.view(B, C, G * P, G * P)
    
    
    
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

