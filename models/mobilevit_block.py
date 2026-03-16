import torch
import torch.nn as nn
from .layers import ConvBNAct, TransformerEncoderBlock

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, transformer_dim, patch_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.local_conv = ConvBNAct(in_channels, in_channels, 3, padding=1)
        self.patch_size = patch_size
        self.proj_in = nn.Linear(in_channels * patch_size * patch_size, transformer_dim)
        self.transformer = TransformerEncoderBlock(transformer_dim, num_heads, mlp_ratio)
        self.proj_out = nn.Linear(transformer_dim, in_channels * patch_size * patch_size)
        # folded has in_channels * patch_size * patch_size channels after fold_patches
        fusion_in_channels = in_channels + in_channels * patch_size * patch_size
        self.fusion_conv = ConvBNAct(fusion_in_channels, in_channels, 1)

    def unfold_patches(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        return x

    def fold_patches(self, x, H, W):
        B, num_patches, patch_dim = x.shape
        x = x.view(B, H // self.patch_size, W // self.patch_size, patch_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        import torch.nn.functional as F
        local_feat = self.local_conv(x)
        B, C, H, W = local_feat.shape
        patches = self.unfold_patches(local_feat)
        patches_proj = self.proj_in(patches)  # [B, num_patches, transformer_dim]
        transformer_feat = self.transformer(patches_proj)
        patches_recon = self.proj_out(transformer_feat)  # [B, num_patches, patch_dim]
        folded = self.fold_patches(patches_recon, H, W)
        # Upsample folded to match local_feat spatial size if needed
        if folded.shape[2:] != local_feat.shape[2:]:
            folded = F.interpolate(folded, size=local_feat.shape[2:], mode='bilinear', align_corners=False)
        print(f"local_feat.shape: {local_feat.shape}, folded.shape: {folded.shape}")
        assert local_feat.shape[0] == folded.shape[0], f"Batch size mismatch: {local_feat.shape[0]} vs {folded.shape[0]}"
        assert local_feat.shape[2:] == folded.shape[2:], f"Spatial size mismatch: {local_feat.shape[2:]} vs {folded.shape[2:]}"
        fused = torch.cat([local_feat, folded], dim=1)
        out = self.fusion_conv(fused)
        return out
