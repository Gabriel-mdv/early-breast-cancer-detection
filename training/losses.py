"""
Loss functions for training.
"""

import torch.nn as nn

def get_loss(loss_name: str = 'cross_entropy') -> nn.Module:
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")
