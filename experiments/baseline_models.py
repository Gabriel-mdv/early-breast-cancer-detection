"""
Baseline models for MobileFCMViTv3 experiments.
"""

import torch.nn as nn
import torchvision.models as models

class MobileNetV3Baseline(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
    def forward(self, x):
        return self.model(x)

class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

class ViTBaseline(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = models.vit_b_16(pretrained=True)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
    def forward(self, x):
        return self.model(x)
