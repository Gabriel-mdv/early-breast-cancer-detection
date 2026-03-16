import json
import os
import numpy as np
import torch
import wandb
from typing import Any
from .metrics import compute_metrics
from .confusion_matrix import plot_confusion_matrix
from .roc_curve import plot_roc_curve


class EvaluationPipeline:
    def __init__(self, model: Any, dataloader: Any, class_names: list,
                 device: torch.device, save_dir: str = 'results', wandb_run=None):
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names
        self.device = device
        self.save_dir = save_dir
        self.wandb_run = wandb_run
        os.makedirs(save_dir, exist_ok=True)

    def evaluate(self) -> dict:
        self.model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for batch in self.dataloader:
                imgs = batch['image'].to(self.device)
                fcm_feats = batch['fcm_feat'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(imgs, fcm_feats)
                probs = outputs.softmax(dim=1).cpu().numpy()   # (B, n_classes)
                preds = outputs.argmax(dim=1).cpu().numpy()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)
                y_prob.extend(probs)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)   # (N, n_classes)

        metrics = compute_metrics(y_true, y_pred, y_prob, self.class_names)

        cm_path = plot_confusion_matrix(y_true, y_pred, self.class_names, self.save_dir)
        roc_path = plot_roc_curve(y_true, y_prob, self.class_names, self.save_dir)

        # Save metrics to JSON
        json_path = os.path.join(self.save_dir, 'metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'Metrics saved to {json_path}')

        # Log to WandB
        if self.wandb_run is not None:
            self.wandb_run.log({f'test/{k}': v for k, v in metrics.items() if v is not None})
            self.wandb_run.log({
                'test/confusion_matrix': wandb.Image(cm_path),
                'test/roc_curve': wandb.Image(roc_path)
            })

        return metrics
