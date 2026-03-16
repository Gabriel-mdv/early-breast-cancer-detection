import yaml
from dataclasses import dataclass, field
from typing import Any, Dict
import os

@dataclass
class DatasetConfig:
    name: str
    input_dir: str
    output_dir: str
    formats: list
    split: dict
    random_seed: int

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    scheduler: str
    mixed_precision: bool
    early_stopping: dict
    checkpoint_dir: str
    log_dir: str
    wandb: dict

@dataclass
class ModelConfig:
    input_channels: int
    fcm_channels: int
    num_classes: int
    image_size: int
    mobilenet_blocks: int
    mobilevit_blocks: int
    attention_fusion: bool

class ConfigLoader:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.config_dir, filename)
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_dataset_config(self) -> DatasetConfig:
        data = self.load_yaml('dataset_config.yaml')
        return DatasetConfig(**data)

    def load_training_config(self) -> TrainingConfig:
        data = self.load_yaml('training_config.yaml')
        return TrainingConfig(**data)

    def load_model_config(self) -> ModelConfig:
        data = self.load_yaml('model_config.yaml')
        return ModelConfig(**data)
