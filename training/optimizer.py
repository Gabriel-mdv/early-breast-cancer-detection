"""
Optimizer factory for training.
"""

import torch
import torch.optim as optim
from typing import Any

def get_optimizer(model: torch.nn.Module, config: Any) -> optim.Optimizer:
    if config.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
