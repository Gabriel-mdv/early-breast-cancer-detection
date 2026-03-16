"""
Training loop for MobileFCMViTv3.
"""

import torch
import wandb
from typing import Any
from tqdm import tqdm
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
        # Use class weights if available from train_loader.dataset
        class_weights = None
        if hasattr(train_loader.dataset, 'get_class_weights'):
            class_weights = train_loader.dataset.get_class_weights().to(device)
        self.loss_fn = get_loss(
            class_weights=class_weights,
            label_smoothing=getattr(config, 'label_smoothing', 0.0)
        )
        self.optimizer = get_optimizer(model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        es_cfg = config.early_stopping
        self.early_stopping = EarlyStopping(es_cfg['patience'], es_cfg['min_delta']) if es_cfg.get('enabled', True) else None
        self.checkpoint = ModelCheckpoint(config.checkpoint_dir)

    def run(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}', unit='batch', leave=True)
            for batch in pbar:
                imgs = batch['image'].to(self.device)
                fcm_feats = batch['fcm_feat'].to(self.device)
                labels = batch['label'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs, fcm_feats)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix(train_loss=f'{loss.item():.4f}')
            val_loss = self.validate()
            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            pbar.set_postfix(train_loss=f'{avg_train_loss:.4f}', val_loss=f'{avg_val_loss:.4f}')
            wandb.log({'train/loss': avg_train_loss, 'val/loss': avg_val_loss, 'epoch': epoch + 1})
            self.scheduler.step()
            if val_loss < (self.early_stopping.best_loss if self.early_stopping else float('inf')):
                self.checkpoint.save(self.model)
            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print('Early stopping triggered.')
                    break

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='  Validating', unit='batch', leave=False):
                imgs = batch['image'].to(self.device)
                fcm_feats = batch['fcm_feat'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(imgs, fcm_feats)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
        return val_loss
