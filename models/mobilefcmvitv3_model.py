"""
MobileFCMViTv3 model architecture.
"""

import torch
import torch.nn as nn
from models.layers.conv_bn_act import ConvBNAct
from models.layers.mbconv import MBConv
from models.layers.attention_fusion import AttentionFusion
from models.blocks.mobilevit_block import MobileViTBlock
from models.blocks.fcm_feature_encoder import FCMFeatureEncoder

class MobileFCMViTv3(nn.Module):
    def __init__(self, input_channels: int, fcm_channels: int, num_classes: int):
        super().__init__()
        self.stem = ConvBNAct(input_channels, 16, 3, stride=2, padding=1)
        self.mbconv1 = MBConv(16, 32)
        self.mbconv2 = MBConv(32, 64)
        self.mbconv3 = MBConv(64, 128)
        self.mobilevit1 = MobileViTBlock(128, 64, patch_size=8, num_heads=2)
        self.mobilevit2 = MobileViTBlock(128, 64, patch_size=8, num_heads=2)
        self.fcm_encoder = FCMFeatureEncoder(fcm_channels, 64)
        self.attn_fusion = AttentionFusion(128+64, 128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, img: torch.Tensor, fcm_feat: torch.Tensor) -> torch.Tensor:
        x = self.stem(img)
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mobilevit1(x)
        x = self.mobilevit2(x)
        fcm = self.fcm_encoder(fcm_feat)
        fused = self.attn_fusion(x, fcm)
        pooled = self.global_pool(fused)
        pooled = pooled.view(pooled.size(0), -1)
        out = self.classifier(pooled)
        return out
