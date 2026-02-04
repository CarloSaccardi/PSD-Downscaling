"""
Pure Swin V2 pre-training implementation using timm.
Pre-training only - no fine-tuning support.
Uses timm's Swin V2 directly with SimMIM masking and light decoder.
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple
import math


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    Truncated normal initialization for mask token.
    Fills the input Tensor with values drawn from a truncated normal distribution.
    """
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class SwinV2Pretrain(nn.Module):
    """
    Swin V2 model for SimMIM pre-training only.
    Uses timm's Swin V2 directly with light decoder for reconstruction.
    
    This is a pure pre-training implementation - no fine-tuning support.
    """
    
    # Variant mapping: simple names -> timm model names
    VARIANT_MAP = {
        # 'base': 'swinv2_base_window12to24_192to384',
        'base': 'swinv2_base_window8_256',
        'large': 'swinv2_large_window12to24_192to384',
        'huge': 'swinv2_cr_huge_384',
    }
    
    def __init__(
        self,
        variant: str = 'base',
        in_channels: int = 7,
        out_channels: int = 6,
        pretrained: bool = False,
        img_size: Tuple[int, int] = (96, 96),
        window_size: int = 6,
        model_patch_size: int = 2,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_light_decoder: bool = True,
    ):
        super().__init__()
        
        # Validate variant
        if variant not in self.VARIANT_MAP:
            raise ValueError(
                f"Unknown variant: {variant}. "
                f"Choose from {list(self.VARIANT_MAP.keys())}"
            )
        
        model_name = self.VARIANT_MAP[variant]
        
        # 1. Create Backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=img_size,
            window_size=window_size,
            patch_size=model_patch_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            features_only=False
        )
        
        #remove .head from backbone
        del self.backbone.head
        
        # 2. Get dims using feature_info (Verified existing)
        self.feature_info = self.backbone.feature_info
        embed_dim = self.feature_info[0]["num_chs"]  # First stage (patch embedding)
        final_dim = self.feature_info[-1]["num_chs"]  # Last stage (deepest features)
        
        # 3. Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        
        self.patch_size = model_patch_size
        
        # 4. Calculate total downsampling factor
        # timm Swin V2: patch_size=4, 4 stages with 2x downsampling each = 4 * 2^(4-1)= 32
        num_stages = len(self.backbone.layers) - 1
        self.total_downsample = self.patch_size * (2 ** num_stages)
        
        # 5. Decoder
        self.use_light_decoder = use_light_decoder
        if self.use_light_decoder:
            self.light_decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=final_dim, 
                    out_channels=out_channels * (self.total_downsample ** 2), 
                    kernel_size=1
                ),
                nn.PixelShuffle(self.total_downsample) 
            )
    

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor):
            """
            Forward pass for SimMIM pre-training with a Forcing Channel.
            
            Args:
                x: Input tensor (B, C=6, H, W)
                patch_mask: Mask tensor (B, H_p, W_p) where 1=MASKED, 0=UNMASKED
            """
            # ============================================
            # STEP 1: Create pixel-level mask (Moved to Top)
            # ============================================
            # We need this immediately to mask specific input channels
            pixel_mask = patch_mask.repeat_interleave(self.patch_size, 1) \
                                .repeat_interleave(self.patch_size, 2)
            pixel_mask = pixel_mask.unsqueeze(1)  # (B, 1, H, W)

            # ============================================
            # STEP 2: Apply Input-Level Masking (Preserve Channel 6)
            # ============================================
            # Clone x to avoid modifying original tensor
            x_masked_input = x.clone()
            
            # Apply mask ONLY to the first 5 channels (indices 0 to 4)
            # x[:, :-1] selects channels 0-4. 
            # We multiply by (1 - pixel_mask) to zero out masked regions.
            x_masked_input[:, :-1, :, :] = x_masked_input[:, :-1, :, :] * (1.0 - pixel_mask)
            
            # Channel 5 (forcing) is untouched and remains fully visible.

            # ============================================
            # STEP 3: Get patch embeddings
            # ============================================
            # Now we embed the partially zeroed input. 
            # The embedding at masked locations now represents: "Zero Image + Real Forcing"
            x_patch = self.backbone.patch_embed(x_masked_input)  # (B, embed_dim, H_p, W_p)
            
            # ============================================
            # STEP 4: Inject Mask Token (Add, don't Replace)
            # ============================================
            B, H_p, W_p, C = x_patch.shape
            x_flat = x_patch.view(B, H_p * W_p, C)
            
            # Expand mask token: (B, L, C)
            mask_tokens = self.mask_token.expand(B, H_p * W_p, -1)
            
            # Get mask weights: (B, L, 1)
            w = patch_mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            
            # KEY CHANGE: ADD the mask token instead of replacing.
            # Current state at masked pos: Embedding(0_img, forcing)
            # Goal state: Embedding(0_img, forcing) + Mask_Signal
            x_masked = x_flat + (mask_tokens * w)
            
            # ============================================
            # STEP 5: Decode (Standard Logic)
            # ============================================
            x_curr = x_masked.view(B, H_p, W_p, C)
            
            if self.use_light_decoder:
                for layer in self.backbone.layers:
                    x_curr = layer(x_curr)
                x_final = self.backbone.norm(x_curr)
                x_final = x_final.permute(0, 3, 1, 2).contiguous()
                reconstruction = self.light_decoder(x_final)
                return reconstruction, pixel_mask.bool()
            
            else:
                features = []
                for layer in self.backbone.layers:
                    x_curr = layer(x_curr)
                    features.append(x_curr.permute(0, 3, 1, 2).contiguous())
                return features, pixel_mask.bool()
        

