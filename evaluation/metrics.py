from typing import List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, jaccard_score

def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: np.ndarray, class_names: List[str]) -> dict:
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['iou'] = float(jaccard_score(y_true, y_pred, average='macro', zero_division=0))
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        metrics['roc_auc'] = None
    return metrics
