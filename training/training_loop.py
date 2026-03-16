"""
Training loop for MobileFCMViTv3.
"""

import torch
from typing import Any
from .losses import get_loss
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .callbacks import EarlyStopping, ModelCheckpoint

class TrainingLoop:
    def __init__(self, model: torch.nn.Module, train_loader: Any, val_loader: Any, config: Any, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.loss_fn = get_loss()
        self.optimizer = get_optimizer(model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        self.early_stopping = EarlyStopping(config.early_stopping['patience'], config.early_stopping['min_delta'])
        self.checkpoint = ModelCheckpoint(config.checkpoint_dir)

    def run(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0
            for imgs, fcm_feats, labels in self.train_loader:
                imgs, fcm_feats, labels = imgs.to(self.device), fcm_feats.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs, fcm_feats)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            val_loss = self.validate()
            self.scheduler.step()
            if val_loss < self.early_stopping.best_loss:
                self.checkpoint.save(self.model)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print('Early stopping triggered.')
                break

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, fcm_feats, labels in self.val_loader:
                imgs, fcm_feats, labels = imgs.to(self.device), fcm_feats.to(self.device), labels.to(self.device)
                outputs = self.model(imgs, fcm_feats)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
        return val_loss
