"""
Memory usage benchmarking for models.
"""

import torch
import psutil

def measure_memory(model: torch.nn.Module, input_shape: tuple, device: torch.device) -> float:
    model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    process = psutil.Process()
    mem_before = process.memory_info().rss
    with torch.no_grad():
        _ = model(dummy_input)
    mem_after = process.memory_info().rss
    return (mem_after - mem_before) / 1024 / 1024  # MB
