"""
Inference latency benchmarking for models.
"""

import torch
import time

def measure_latency(model: torch.nn.Module, input_shape: tuple, device: torch.device, n_runs: int = 100) -> float:
    model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            times.append(end - start)
    avg_latency = sum(times) / n_runs
    return avg_latency
