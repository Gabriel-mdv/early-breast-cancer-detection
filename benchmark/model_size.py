"""
Model size benchmarking for PyTorch models.
"""

import os

def get_model_size(model_path: str) -> float:
    size = os.path.getsize(model_path)
    return size / 1024 / 1024  # MB
