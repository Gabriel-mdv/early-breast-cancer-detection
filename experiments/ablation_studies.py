"""
Ablation studies for MobileFCMViTv3 experiments.
"""

from models.mobilefcmvitv3_model import MobileFCMViTv3
from models.blocks.mobilevit_block import MobileViTBlock
from models.blocks.fcm_feature_encoder import FCMFeatureEncoder

class MobileViTOnly:
    def __init__(self, input_channels: int, num_classes: int):
        self.model = MobileViTBlock(input_channels, 64, patch_size=8, num_heads=2)
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        feat = self.model(x)
        pooled = nn.AdaptiveAvgPool2d(1)(feat)
        pooled = pooled.view(pooled.size(0), -1)
        return self.classifier(pooled)

class FCMOnly:
    def __init__(self, fcm_channels: int, num_classes: int):
        self.encoder = FCMFeatureEncoder(fcm_channels, 64)
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        feat = self.encoder(x)
        pooled = nn.AdaptiveAvgPool2d(1)(feat)
        pooled = pooled.view(pooled.size(0), -1)
        return self.classifier(pooled)

class MobileFCMViTv3Ablation:
    def __init__(self, input_channels: int, fcm_channels: int, num_classes: int):
        self.model = MobileFCMViTv3(input_channels, fcm_channels, num_classes)
    def forward(self, img, fcm_feat):
        return self.model(img, fcm_feat)
