
"""
Trainer class for MobileFCMViTv3.
"""

import torch
import wandb
from typing import Any
from .training_loop import TrainingLoop

class Trainer:
    def __init__(self, model: torch.nn.Module, train_loader: Any, val_loader: Any, config: Any, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        wandb.init(project=config.wandb['project'])
        self.loop = TrainingLoop(model, train_loader, val_loader, config, device)

    def train(self):
        self.loop.run()
        wandb.finish()
