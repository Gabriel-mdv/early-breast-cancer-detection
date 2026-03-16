"""
MobileNet inverted residual block (MBConv).
"""

import torch
import torch.nn as nn
from .conv_bn_act import ConvBNAct

class MBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 4, stride: int = 1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.expand = ConvBNAct(in_channels, hidden_dim, 1)
        self.depthwise = ConvBNAct(hidden_dim, hidden_dim, 3, stride, 1)
        self.project = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        x = self.bn(x)
        return self.act(x)
