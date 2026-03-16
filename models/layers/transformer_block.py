"""
Transformer encoder block for MobileViT.
"""

import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.norm2(x)
        x = self.mlp(x)
        return x
