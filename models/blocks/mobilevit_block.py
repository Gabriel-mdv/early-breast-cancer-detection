"""
MobileViT transformer block.
"""

import torch
import torch.nn as nn
from ..layers.conv_bn_act import ConvBNAct
from ..layers.transformer_block import TransformerEncoderBlock

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels: int, transformer_dim: int, patch_size: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.local_conv = ConvBNAct(in_channels, in_channels, 3, padding=1)
        self.patch_size = patch_size
        self.transformer = TransformerEncoderBlock(transformer_dim, num_heads, mlp_ratio)
        self.fusion_conv = ConvBNAct(in_channels + transformer_dim, in_channels, 1)

    def unfold_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        return x

    def fold_patches(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, num_patches, patch_dim = x.shape
        x = x.view(B, H // self.patch_size, W // self.patch_size, patch_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local_conv(x)
        B, C, H, W = local_feat.shape
        patches = self.unfold_patches(local_feat)
        transformer_feat = self.transformer(patches)
        folded = self.fold_patches(transformer_feat, H, W)
        fused = torch.cat([local_feat, folded], dim=1)
        out = self.fusion_conv(fused)
        return out
