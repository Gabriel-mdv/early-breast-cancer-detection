
"""
CLI script for training MobileFCMViTv3.
"""

import argparse
from mobilefcmvitv3.config_loader import ConfigLoader
from training.trainer import Trainer
import torch

def main():
    parser = argparse.ArgumentParser(description='Train MobileFCMViTv3')
    parser.add_argument('--config_dir', type=str, default='config/', help='Path to config directory')
    args = parser.parse_args()
    config_loader = ConfigLoader(args.config_dir)
    dataset_config = config_loader.load_dataset_config()
    training_config = config_loader.load_training_config()
    model_config = config_loader.load_model_config()
    # ...dataset and dataloader setup...
    # model = MobileFCMViTv3(...)
    # trainer = Trainer(model, train_loader, val_loader, training_config, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # trainer.train()
    print('Training started.')

if __name__ == '__main__':
    main()
