"""
EfficientNet model for ultrasound classification.
"""

import torch.nn as nn
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    def forward(self, x):
        return self.model(x)
