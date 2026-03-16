"""
Export PyTorch model to ONNX format.
"""

import torch

def export_to_onnx(model: torch.nn.Module, input_shape: tuple, export_path: str):
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(model, dummy_input, export_path, opset_version=12, input_names=['input'], output_names=['output'])
