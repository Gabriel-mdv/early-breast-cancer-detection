import torch
import torch.nn as nn
from .layers import ConvBNAct, TransformerEncoderBlock


class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, transformer_dim, patch_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.patch_size = patch_size
        self.local_conv = ConvBNAct(in_channels, in_channels, 3, padding=1)
        self.proj_in = nn.Linear(in_channels, transformer_dim)
        self.transformer = TransformerEncoderBlock(transformer_dim, num_heads, mlp_ratio)
        self.proj_out = nn.Linear(transformer_dim, in_channels)
        self.fusion_conv = ConvBNAct(2 * in_channels, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, f"Spatial size ({H},{W}) must be divisible by patch_size {P}"

        local_feat = self.local_conv(x)

        # Unfold: (B, C, H, W) -> (B*num_patches, pixels_per_patch, C)
        # num_patches = (H/P)*(W/P), pixels_per_patch = P*P
        nh, nw = H // P, W // P
        # reshape to (B, C, nh, P, nw, P) then group patches
        xp = local_feat.view(B, C, nh, P, nw, P)
        xp = xp.permute(0, 2, 4, 3, 5, 1).contiguous()   # (B, nh, nw, P, P, C)
        xp = xp.view(B * nh * nw, P * P, C)               # (B*num_patches, pixels, C)

        # Transformer over pixels within each patch
        xp = self.proj_in(xp)                              # (B*num_patches, pixels, transformer_dim)
        xp = self.transformer(xp)
        xp = self.proj_out(xp)                             # (B*num_patches, pixels, C)

        # Fold back: (B*num_patches, P*P, C) -> (B, C, H, W)
        xp = xp.view(B, nh, nw, P, P, C)
        xp = xp.permute(0, 5, 1, 3, 2, 4).contiguous()   # (B, C, nh, P, nw, P)
        xp = xp.view(B, C, H, W)

        # Fuse local conv features with global transformer features
        out = self.fusion_conv(torch.cat([local_feat, xp], dim=1))
        return out
