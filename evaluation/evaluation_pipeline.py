from typing import Any
from .metrics import compute_metrics
from .confusion_matrix import plot_confusion_matrix
from .roc_curve import plot_roc_curve
from .gradcam import GradCAM

class EvaluationPipeline:
    def __init__(self, model: Any, dataloader: Any, class_names: list):
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names

    def evaluate(self):
        y_true, y_pred, y_prob = [], [], []
        for imgs, fcm_feats, labels in self.dataloader:
            outputs = self.model(imgs, fcm_feats)
            probs = outputs.softmax(dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        plot_confusion_matrix(y_true, y_pred, self.class_names)
        plot_roc_curve(y_true, y_prob)
        return metrics
