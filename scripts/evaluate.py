"""
CLI script for evaluating MobileFCMViTv3.
"""

import argparse
import torch
import wandb
from mobilefcmvitv3.config_loader import ConfigLoader
from preprocessing.dataloader import BUSIDataLoader
from models.mobilefcmvitv3 import MobileFCMViTv3
from evaluation.evaluation_pipeline import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(description='Evaluate MobileFCMViTv3')
    parser.add_argument('--config_dir', type=str, default='config/')
    parser.add_argument('--checkpoint', type=str, default='training/checkpoints/best_model.pt')
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    config_loader = ConfigLoader(args.config_dir)
    dataset_config = config_loader.load_dataset_config()
    training_config = config_loader.load_training_config()
    model_config = config_loader.load_model_config()

    # Dataloader
    processed_dir = dataset_config.output_dir
    manifest_dir = f"{processed_dir}/manifests"
    dataloader = BUSIDataLoader(processed_dir=f"{processed_dir}/images", batch_size=training_config.batch_size)
    test_loader = dataloader.create_splits(
        train_split_file=f"{manifest_dir}/train_manifest.txt",
        val_split_file=f"{manifest_dir}/val_manifest.txt",
        test_split_file=f"{manifest_dir}/test_manifest.txt"
    )['test']

    # Model + checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileFCMViTv3(model_config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    # Strip DataParallel 'module.' prefix if present
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f'Loaded checkpoint: {args.checkpoint}')

    # WandB
    run = wandb.init(project=training_config.wandb['project'], job_type='evaluation')

    # Evaluate
    class_names = ['benign', 'malignant', 'normal']
    pipeline = EvaluationPipeline(model, test_loader, class_names, device,
                                  save_dir=args.save_dir, wandb_run=run)
    metrics = pipeline.evaluate()
    print('Evaluation complete. Metrics:', metrics)

    wandb.finish()


if __name__ == '__main__':
    main()
