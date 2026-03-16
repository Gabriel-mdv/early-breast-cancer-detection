
"""
CLI script for benchmarking MobileFCMViTv3 model.
"""

import argparse
from benchmark.latency_test import measure_latency
from benchmark.memory_usage_test import measure_memory
from benchmark.model_size import get_model_size

def main():
    parser = argparse.ArgumentParser(description='Benchmark MobileFCMViTv3 model')
    parser.add_argument('--model_path', type=str, default='export/model.onnx')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[1, 1, 224, 224])
    args = parser.parse_args()

    # Load configs
    from mobilefcmvitv3.config_loader import ConfigLoader
    config_loader = ConfigLoader('config/')
    model_config = config_loader.load_model_config()

    # Model setup
    from models.mobilefcmvitv3 import MobileFCMViTv3
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileFCMViTv3(model_config).to(device)

    # Benchmarking steps
    from benchmark.latency_test import measure_latency
    from benchmark.memory_usage_test import measure_memory
    from benchmark.model_size import get_model_size

    latency = measure_latency(model, tuple(args.input_shape), device)
    memory = measure_memory(model, tuple(args.input_shape), device)
    size = get_model_size(args.model_path)
    print(f'Benchmarking finished.\nLatency: {latency}\nMemory: {memory}\nModel size: {size}')

if __name__ == '__main__':
    main()
