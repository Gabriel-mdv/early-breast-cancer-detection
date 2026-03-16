
"""
CLI script for evaluating MobileFCMViTv3.
"""

import argparse
from mobilefcmvitv3.config_loader import ConfigLoader
from evaluation.evaluation_pipeline import EvaluationPipeline
import torch

def main():
    parser = argparse.ArgumentParser(description='Evaluate MobileFCMViTv3')
    parser.add_argument('--config_dir', type=str, default='config/', help='Path to config directory')
    args = parser.parse_args()
    config_loader = ConfigLoader(args.config_dir)
    # ...model and dataloader setup...
    # pipeline = EvaluationPipeline(model, dataloader, class_names)
    # metrics = pipeline.evaluate()
    print('Evaluation started.')

if __name__ == '__main__':
    main()
