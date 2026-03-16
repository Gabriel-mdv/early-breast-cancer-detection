"""
Attention-based fusion module for combining image and FCM features.
"""

import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        fusion = torch.cat([x1, x2], dim=1)
        attn = self.sigmoid(self.conv(fusion))
        return fusion * attn
