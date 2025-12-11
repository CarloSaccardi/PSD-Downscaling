# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F # Using F.mse_loss is common

import pytorch_lightning as pl

import random
from src.utils import constants, vis
import matplotlib.pyplot as plt
import wandb
    
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.swin_simMIM import SimMIMSwinUNETR
from monai.utils import ensure_tuple_rep


class SwinUNetrWrapper(pl.LightningModule):
    def __init__(self, args):
        super(SwinUNetrWrapper, self).__init__()
        
        self.wandb_project = args.wandb_project
        self.use_light_decoder = args.use_light_decoder
        
        # This makes args available as self.hparams (e.g., self.hparams.lr)
        self.save_hyperparameters(args)
        
        # For 2D data, ensure window_size is a 2D tuple
        window_size = ensure_tuple_rep(7, len(args.img_resolution))  # (7, 7) for 2D
        
        # Use the original SwinUNETR architecture
        self.swin_unetr = SimMIMSwinUNETR(
            use_light_decoder=args.use_light_decoder,
            freeze_encoder=args.freeze_encoder,
            in_channels=args.img_in_channels,
            out_channels=args.img_out_channels,  # Reconstruction: output same as input
            patch_size=2,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=window_size,  # (7, 7) for 2D
            qkv_bias=True,
            mlp_ratio=4.0,
            feature_size=args.model_channels,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            norm_layer=torch.nn.LayerNorm,
            patch_norm=False,
            use_checkpoint=True if args.load else False,
            spatial_dims=len(args.img_resolution),  # Should be 2 for 2D data
            downsample="merging",
            use_v2=False,
        )
        
        
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.swin_unetr.parameters(), 
            lr=self.hparams.lr
        )
        return opt
    
        
    def forward(self, x, patch_mask):
        """
        Forward pass that handles both pre-training and fine-tuning modes.
        
        Returns:
            x_rec: Reconstructed tensor
            mask_bool: Mask tensor (None for fine-tuning)
        """
        if self.use_light_decoder:
            x_rec, mask_bool = self.swin_unetr(x.contiguous(), patch_mask.contiguous())
            return x_rec, mask_bool
        else:
            x_rec = self.swin_unetr(x.contiguous(), patch_mask=None)
            return x_rec, None
    

    def training_step(self, batch, batch_idx):
        if self.use_light_decoder:
            x, mask = batch 
            x = self._upsample(x)
            x_rec, mask_bool = self.forward(x, patch_mask=mask)
            target = None
            
        else:
            x, target = batch 
            x_rec, mask_bool = self.forward(x, patch_mask=None)
            
        loss = self.get_loss(x, x_rec, target, mask_bool)
        
        train_log_dict = {
            "train_loss": loss,
        }
        self.log_dict(
            train_log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss
    

    def validation_step(self, batch, batch_idx):
        if self.use_light_decoder:
            x, mask = batch 
            x = self._upsample(x)
            x_rec, mask_bool = self.forward(x, patch_mask=mask)
            target = None
            ground_truth = x
            
        else:
            x, target = batch 
            x_rec, mask_bool = self.forward(x, patch_mask=None)
            ground_truth = target
            
        loss = self.get_loss(x, x_rec, target, mask_bool)
        
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
            self.load_metrics_and_plots(x_rec, ground_truth, batch_idx, mask=None)
            
            
    def get_loss(self, x, x_rec, target, mask_bool):
        """
        Compute loss based on the training mode.
        
        Args:
            x: Ground truth input tensor
            x_rec: Reconstructed/predicted tensor
            mask_bool: Boolean mask tensor (only used when use_light_decoder=True)
        
        Returns:
            loss: Computed loss value
        """
        if self.use_light_decoder:
            # Pre-training: compute loss only on masked regions
            loss_mask = ~mask_bool
            loss_mask = loss_mask.expand_as(x)
            masked_rec = x_rec[loss_mask]
            masked_target = x[loss_mask]
            loss = F.mse_loss(masked_rec, masked_target)
            
        else:
            # Fine-tuning: compute loss on entire input (no masking)
            loss = F.mse_loss(target, x_rec)
        
        return loss
    
            
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
        
        sample = random.randint(0, prediction.shape[0] - 1) #random.randint(0, prediction.shape[1] - 1)

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
        
        
    def _upsample(self, lr_tensor, target_size=(384, 384)):
        """
        Upsample the input tensor to match the target tensor's spatial dimensions.
        """
        #interpolate
        return F.interpolate(
                lr_tensor,
                size=target_size,
                mode='bicubic',
                align_corners=False
            )
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs):
        """
        Load checkpoint, keeping only encoder weights when switching from pre-training to fine-tuning.
        """
        # Load checkpoint normally
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Get args - should be provided via kwargs['args'] from main.py
        if 'args' not in kwargs:
            raise ValueError("'args' must be provided when calling load_from_checkpoint")
        
        # Create model with provided args
        model = cls(kwargs['args'])
        
        # Filter state dict: keep only swinViT and mask_token
        state_dict = checkpoint['state_dict']
        filtered = {k: v for k, v in state_dict.items() 
                    if k.startswith('swin_unetr.swinViT.') or k.startswith('swin_unetr.mask_token')}
        
        # Load with strict=False
        model.load_state_dict(filtered, strict=False)
        
        # Handle encoder freezing/unfreezing in fine-tuning mode
        if not model.use_light_decoder:
            if kwargs['args'].freeze_encoder:
                # Freeze encoder parameters
                for param in model.swin_unetr.swinViT.parameters():
                    param.requires_grad = False
                model.swin_unetr.mask_token.requires_grad = False
            else:
                
                for param in model.swin_unetr.swinViT.parameters():
                    param.requires_grad = True
                model.swin_unetr.mask_token.requires_grad = False
        
        return model

 