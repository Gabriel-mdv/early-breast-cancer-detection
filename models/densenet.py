"""
DenseNet model for ultrasound classification.
"""

import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
    def forward(self, x):
        return self.model(x)
