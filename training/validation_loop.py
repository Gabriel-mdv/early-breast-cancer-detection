"""
Validation loop for MobileFCMViTv3.
"""

import torch
from typing import Any

class ValidationLoop:
    def __init__(self, model: torch.nn.Module, val_loader: Any, loss_fn: Any, device: torch.device):
        self.model = model
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device

    def run(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                imgs = batch['image'].to(self.device)
                fcm_feats = batch['fcm_feat'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(imgs, fcm_feats)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
        return val_loss
