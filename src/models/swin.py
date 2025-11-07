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

from monai.networks.nets.swin_unetr import SwinUNETR
from monai.utils import ensure_tuple_rep

import pytorch_lightning as pl


class SwinUNetrWrapper(pl.LightningModule):
    def __init__(self, args, upsample="vae", dim=3072):
        super(SwinUNetrWrapper, self).__init__()
        
        # For 2D data, ensure window_size and patch_size are 2D tuples
        window_size = ensure_tuple_rep(7, len(args.img_resolution))  # (7, 7) for 2D
        
        # Use the original SwinUNETR architecture for 2D
        # For reconstruction, out_channels should match in_channels
        self.swin_unetr = SwinUNETR(
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

    def training_step(self, x):
        # SwinUNETR handles the full encoder-decoder with skip connections
        # Returns reconstruction with same shape as input: [B, in_channels, H, W]
        x_rec = self.swin_unetr(x.contiguous())
        return x_rec
