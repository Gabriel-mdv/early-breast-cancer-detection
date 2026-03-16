"""
Learning rate scheduler factory.
"""

import math
import torch.optim as optim
from typing import Any

def get_scheduler(optimizer: optim.Optimizer, config: Any):
    if config.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif config.scheduler == 'warmup_cosine':
        warmup_epochs = getattr(config, 'warmup_epochs', 5)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, config.epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
