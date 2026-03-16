
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

    # Load configs
    config_loader = ConfigLoader(args.config_dir)
    dataset_config = config_loader.load_dataset_config()
    training_config = config_loader.load_training_config()
    model_config = config_loader.load_model_config()

    # Dataloader setup
    from preprocessing.dataloader import BUSIDataLoader
    processed_dir = dataset_config.output_dir
    manifest_dir = f"{processed_dir}/manifests"
    batch_size = training_config.batch_size
    dataloader = BUSIDataLoader(processed_dir=f"{processed_dir}/images", batch_size=batch_size)
    loaders = dataloader.create_splits(
        train_split_file=f"{manifest_dir}/train_manifest.txt",
        val_split_file=f"{manifest_dir}/val_manifest.txt",
        test_split_file=f"{manifest_dir}/test_manifest.txt"
    )
    train_loader = loaders['train']
    val_loader = loaders['val']

    # Model setup
    from models.mobilefcmvitv3 import MobileFCMViTv3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileFCMViTv3(model_config).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model)

    # Training
    trainer = Trainer(model, train_loader, val_loader, training_config, device)
    trainer.train()
    print('Training finished.')

if __name__ == '__main__':
    main()
