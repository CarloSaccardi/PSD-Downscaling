"""
Swin V2 Wrapper for pre-training with SimMIM.
Pure timm + PyTorch implementation with no MONAI dependencies.
Pre-training only - no fine-tuning support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random
from src.utils import constants, vis
import matplotlib.pyplot as plt
import wandb
from physicsnemo.models.swinv2.simMIM import SwinV2Pretrain

class SwinV2Wrapper(pl.LightningModule):
    """
    Wrapper for Swin V2 pre-training with SimMIM.
    Uses timm's Swin Transformer V2 as the backbone.
    Pre-training only - no fine-tuning support.
    """
    
    def __init__(self, args):
        super(SwinV2Wrapper, self).__init__()
        
        # This makes args available as self.hparams
        self.save_hyperparameters(args)
        
        self.wandb_project = args.wandb_project
        self.use_light_decoder = args.use_light_decoder
        self.swin_v2_variant = args.swin_v2_variant
        self.img_size = args.img_size
        self.window_size = args.window_size
        self.img_in_channels = args.img_in_channels
        self.img_out_channels = args.img_out_channels
        
        self.model = SwinV2Pretrain(
            variant=self.swin_v2_variant,
            in_channels=self.img_in_channels,
            out_channels=self.img_out_channels,
            img_size=self.img_size,
            window_size=self.window_size,
            use_light_decoder=self.use_light_decoder,
            pretrained=getattr(args, 'pretrained_backbone', False),
            drop_rate=getattr(args, 'drop_rate', 0.0),
            attn_drop_rate=getattr(args, 'attn_drop_rate', 0.0),
            drop_path_rate=getattr(args, 'drop_path_rate', 0.0),
        )
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr
        )
        return opt
    
    def forward(self, x, patch_mask):
        """
        Forward pass for pre-training.
        
        Returns:
            x_rec: Reconstructed tensor
            mask_bool: Boolean mask tensor for loss computation
        """
        x_rec, mask_bool = self.model(x.contiguous(), patch_mask.contiguous())
        return x_rec, mask_bool
    
    def training_step(self, batch, batch_idx):
        x, mask = batch
        
        x_rec, mask_bool = self.forward(x, patch_mask=mask)
        
        # Loss only on masked regions
        loss = self.get_loss(x, x_rec, mask_bool)
        
        train_log_dict = {
            "train_loss": loss,
        }
        self.log_dict(
            train_log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        
        x_rec, mask_bool = self.forward(x, patch_mask=mask)
        
        loss = self.get_loss(x, x_rec, mask_bool)
        
        val_log_dict = {
            "val_loss": loss,
        }
        self.log_dict(
            val_log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.current_epoch % 10 == 0
            and self.wandb_project is not None
        ):
            self.load_metrics_and_plots(x_rec, x, mask=mask_bool)
    
    def get_loss(self, x, x_rec, mask_bool):
        """
        Compute loss for pre-training.
        Loss is computed only on masked regions.
        """
        # mask_bool: True = MASKED (where we want to compute loss)
        loss_mask = mask_bool.expand_as(x_rec)
        masked_rec = x_rec[loss_mask]
        
        x_vars = x[:, :6, :, :]
        masked_target = x_vars[loss_mask]
        loss = F.mse_loss(masked_rec, masked_target)
        return loss
    
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
        mask_grid = mask[sample, :, :].flatten(1,2).squeeze(0)
        
        
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
        
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs):
        """
        Load checkpoint for pre-training.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        if 'args' not in kwargs:
            raise ValueError("'args' must be provided when calling load_from_checkpoint")
        
        model = cls(kwargs['args'])
        
        # Load state dict
        state_dict = checkpoint['state_dict']
        
        # If self.use_light_decoder is False, then remove the light_decoder from the state dict
        if not kwargs['args'].use_light_decoder:
            keys_to_remove = [k for k in state_dict.keys() if k.startswith('model.light_decoder')]
            for k in keys_to_remove:
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        
        return model

