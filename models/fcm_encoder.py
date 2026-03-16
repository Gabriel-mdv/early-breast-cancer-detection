import torch
import torch.nn as nn
from .layers import ConvBNAct

class FCMFeatureEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNAct(in_channels, 16, 3, padding=1),
            ConvBNAct(16, 32, 3, padding=1),
            ConvBNAct(32, out_channels, 3, padding=1)
        )
    def forward(self, x):
        return self.encoder(x)
