"""
Learning rate scheduler factory.
"""

import torch.optim as optim
from typing import Any

def get_scheduler(optimizer: optim.Optimizer, config: Any):
    if config.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
