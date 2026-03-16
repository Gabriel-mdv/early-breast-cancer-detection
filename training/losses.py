"""
Loss functions for training.
"""

import torch.nn as nn

def get_loss(loss_name: str = 'cross_entropy', class_weights=None, label_smoothing: float = 0.0) -> nn.Module:
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")
