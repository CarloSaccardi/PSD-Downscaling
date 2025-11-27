import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.layers import trunc_normal_
import numpy as np

class SimMIMSwinUNETR(SwinUNETR):
    """
    SwinUNETR subclass for SimMIM pre-training (spatial_dims=2 ONLY)
    using a hybrid masking strategy to prevent information leaks 
    from convolutional skip connections.
    """
    def __init__(self, use_light_decoder: bool = False, **kwargs):
        # Enforce spatial_dims=2
        if "spatial_dims" in kwargs and kwargs["spatial_dims"] != 2:
            raise ValueError("SimMIMSwinUNETR_2D is only for spatial_dims=2.")
        kwargs["spatial_dims"] = 2
        kwargs["use_v2"] = False
        self.use_light_decoder = use_light_decoder
        
        super().__init__(**kwargs)
        
        # Learnable mask token for the Transformer backbone
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.swinViT.embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        
        if self.use_light_decoder:
            # This is the "light decoder" head.
            # The backbone downsamples 5 times (1 patch_embed + 4 layers)
            # So, 2^5 = 32. We need to upsample by 32x.
            # The input will be from encoder10, which has 16 * feature_size channels
            total_upsample_factor = 2**(self.swinViT.num_layers + 1)
            
            self.light_decoder_head = nn.ConvTranspose2d(
                in_channels=16 * 96,
                out_channels=11,
                kernel_size=total_upsample_factor,
                stride=total_upsample_factor
            )
            
            # --- We don't need the heavy U-Net decoder for pre-training ---
            # --- so we can delete them to save memory ---
            del self.decoder5
            del self.decoder4
            del self.decoder3
            del self.decoder2
            del self.decoder1
            del self.out # The light head outputs directly
            del self.encoder1
            del self.encoder2
            del self.encoder3
            del self.encoder4
            
            
        else:
            # Fine-tuning stage: Freeze the pre-trained Swin Transformer backbone
            # Freeze all parameters in the Swin Transformer
            for param in self.swinViT.parameters():
                param.requires_grad = False
            
            # Freeze the mask token (learned during pre-training)
            self.mask_token.requires_grad = False

    def forward(self, x_in, patch_mask):
        """
        Modified forward pass for SimMIM (2D Only).
        
        Args:
            x_in (torch.Tensor): Input image tensor (B, C, H, W).
            patch_mask (torch.Tensor): Mask for tokens (B, H_p, W_p) 
                                    or (B, L) where L is num_patches.
                                    1 = MASKED, 0 = UNMASKED.
                                    Only used when use_light_decoder=True (pre-training).
        """
        if self.use_light_decoder:
            # --- PRE-TRAINING MODE: Apply masking ---
            patch_size_tuple = self.swinViT.patch_size # (patch_h, patch_w)
            
            # --- 1. Create PIXEL-LEVEL mask (for Conv Skips) ---
            # We use (1.0 - patch_mask) because 1 means MASKED, so we multiply by 0
            pixel_mask = (1.0 - patch_mask).repeat_interleave(patch_size_tuple[0], 1) \
                                        .repeat_interleave(patch_size_tuple[1], 2)
            pixel_mask = pixel_mask.unsqueeze(1) # (B, 1, H, W)
            
            # --- 2. Create TOKEN-masked input for TRANSFORMER backbone ---
            #    This uses the *original* x_in to get real embeddings,
            #    which are then replaced by the learnable mask_token.
            x0_unmasked_embed = self.swinViT.patch_embed(x_in)
            x0_unmasked_embed = self.swinViT.pos_drop(x0_unmasked_embed)
            
            x_embed_shape = x0_unmasked_embed.shape # (B, C, H_p, W_p)
            x0_flat = x0_unmasked_embed.flatten(2).transpose(1, 2) # (B, L, C)
            B, L, C = x0_flat.shape
            
            mask_tokens = self.mask_token.expand(B, L, -1)
            # w is (B, L, 1), where 1 = MASKED
            w = patch_mask.flatten(1).unsqueeze(-1).type_as(mask_tokens) 
            
            x_token_masked_flat = x0_flat * (1.0 - w) + mask_tokens * w
            
            # --- 3. Reshape token-masked input for SwinViT layers ---
            x_token_masked = x_token_masked_flat.transpose(1, 2).view(B, C, x_embed_shape[2], x_embed_shape[3])
            
            # --- 4. Run TRANSFORMER backbone on token-masked input ---
            x = x_token_masked
            x1 = self.swinViT.layers1[0](x.contiguous())
            x1_out = self.swinViT.proj_out(x1, self.normalize)
            
            x2 = self.swinViT.layers2[0](x1.contiguous())
            x2_out = self.swinViT.proj_out(x2, self.normalize)
            
            x3 = self.swinViT.layers3[0](x2.contiguous())
            x3_out = self.swinViT.proj_out(x3, self.normalize)
            
            x4 = self.swinViT.layers4[0](x3.contiguous())
            x4_out = self.swinViT.proj_out(x4, self.normalize)
            
            hidden_states_out = [None, x1_out, x2_out, x3_out, x4_out]
            
            # --- 5. Run the light decoder ---
            dec4 = self.encoder10(hidden_states_out[4])
            logits = self.light_decoder_head(dec4)
            
            # Return logits and the pixel_mask (for loss calculation)
            # pixel_mask is 0.0 for masked, 1.0 for unmasked
            return logits, pixel_mask.bool()
        
        else:
            # --- FINE-TUNING MODE: No masking, standard forward pass ---
            # Run the full SwinUNETR forward pass without any masking
            hidden_states_out = self.swinViT(x_in, self.normalize)
            
            # Process skip connections (no masking needed)
            enc0 = self.encoder1(x_in)
            enc1 = self.encoder2(hidden_states_out[0])
            enc2 = self.encoder3(hidden_states_out[1])
            enc3 = self.encoder4(hidden_states_out[2])
            dec4 = self.encoder10(hidden_states_out[4])
            
            # Run the U-Net decoder
            dec3 = self.decoder5(dec4, hidden_states_out[3])
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            dec0 = self.decoder2(dec1, enc1)
            out = self.decoder1(dec0, enc0)
            logits = self.out(out)
            
            # Return logits only (no mask needed for fine-tuning)
            return logits