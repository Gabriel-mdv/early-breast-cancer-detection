"""
DataLoader Module
PyTorch and TensorFlow data loading with preprocessing
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, List
from torch.utils.data import Dataset, DataLoader
import torch
import json


class BUSIDataset(Dataset):
    """PyTorch Dataset for BUSI breast cancer images"""
    
    def __init__(self, 
                 image_dir: Union[str, Path],
                 mask_dir: Optional[Union[str, Path]] = None,
                 class_labels: Optional[Dict[str, int]] = None,
                 split: str = 'train',
                 split_file: Optional[Union[str, Path]] = None,
                 transform=None,
                 include_mask: bool = False):
        """
        Initialize dataset
        
        Args:
            image_dir: Directory containing preprocessed images
            mask_dir: Directory containing masks
            class_labels: Map class names to labels (e.g., {'benign': 0, 'malignant': 1, 'normal': 2})
            split: 'train', 'val', or 'test'
            split_file: Path to split manifest file
            transform: Optional transforms to apply
            include_mask: Whether to include masks in output
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.split = split
        self.include_mask = include_mask
        self.transform = transform
        
        # Default class labels
        self.class_labels = class_labels or {
            'benign': 0,
            'malignant': 1,
            'normal': 2
        }
        
        self.class_names = {v: k for k, v in self.class_labels.items()}
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        
        self._load_dataset(split_file)
    
    def _load_dataset(self, split_file: Optional[Union[str, Path]]):
        """Load image paths and labels"""
        if split_file:
            # Load from manifest file
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        image_path = self.image_dir / parts[0]
                        label = int(parts[1])
                        
                        if image_path.exists():
                            self.image_paths.append(image_path)
                            self.labels.append(label)
                            
                            if self.mask_dir:
                                mask_path = self.mask_dir / f"{image_path.stem}_mask.npy"
                                self.mask_paths.append(mask_path if mask_path.exists() else None)
        else:
            # Load from directory structure
            for class_name, class_id in self.class_labels.items():
                class_dir = self.image_dir / class_name
                if class_dir.exists():
                    for img_file in sorted(class_dir.glob('*.npy')):
                        self.image_paths.append(img_file)
                        self.labels.append(class_id)
                        
                        if self.mask_dir:
                            mask_path = self.mask_dir / class_name / f"{img_file.stem}_mask.npy"
                            self.mask_paths.append(mask_path if mask_path.exists() else None)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get item at index
        
        Returns:
            Dict with 'image' and optionally 'mask' and 'label'
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = np.load(image_path)
        
        # Convert to torch tensor
        image_tensor = torch.from_numpy(image).float()
        
        # Apply transform if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        result = {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # Load mask if requested
        if self.include_mask and self.mask_paths and self.mask_paths[idx]:
            mask = np.load(self.mask_paths[idx])
            result['mask'] = torch.from_numpy(mask).float()
        
        return result
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced dataset
        
        Returns:
            Tensor of class weights
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.from_numpy(weights).float()


class BUSIDataLoader:
    """Wrapper for creating PyTorch DataLoaders"""
    
    def __init__(self, 
                 processed_dir: Union[str, Path],
                 mask_dir: Optional[Union[str, Path]] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 shuffle_train: bool = True,
                 class_labels: Optional[Dict[str, int]] = None):
        """
        Initialize DataLoader wrapper
        
        Args:
            processed_dir: Directory containing preprocessed images
            mask_dir: Directory containing masks
            batch_size: Batch size
            num_workers: Number of workers for data loading
            shuffle_train: Whether to shuffle training data
            class_labels: Class label mapping
        """
        self.processed_dir = Path(processed_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_labels = class_labels or {
            'benign': 0,
            'malignant': 1,
            'normal': 2
        }
        self.shuffle_train = shuffle_train
    
    def create_dataset(self, split: str = 'train',
                      split_file: Optional[Union[str, Path]] = None) -> BUSIDataset:
        """
        Create dataset for split
        
        Args:
            split: 'train', 'val', or 'test'
            split_file: Optional manifest file for this split
        
        Returns:
            BUSIDataset instance
        """
        dataset = BUSIDataset(
            image_dir=self.processed_dir,
            mask_dir=self.mask_dir,
            class_labels=self.class_labels,
            split=split,
            split_file=split_file,
            include_mask=False
        )
        return dataset
    
    def create_dataloader(self, split: str = 'train',
                         split_file: Optional[Union[str, Path]] = None,
                         shuffle: Optional[bool] = None) -> DataLoader:
        """
        Create PyTorch DataLoader
        
        Args:
            split: 'train', 'val', or 'test'
            split_file: Optional manifest file
            shuffle: Whether to shuffle (if None, uses self.shuffle_train for train split)
        
        Returns:
            PyTorch DataLoader
        """
        if shuffle is None:
            shuffle = (split == 'train') and self.shuffle_train
        
        dataset = self.create_dataset(split, split_file)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    def create_splits(self, train_split_file: Union[str, Path],
                     val_split_file: Union[str, Path],
                     test_split_file: Union[str, Path]) -> Dict:
        """
        Create all three dataloaders from split manifests
        
        Args:
            train_split_file: Path to training split manifest
            val_split_file: Path to validation split manifest
            test_split_file: Path to test split manifest
        
        Returns:
            Dict with 'train', 'val', 'test' dataloaders
        """
        return {
            'train': self.create_dataloader('train', train_split_file, shuffle=True),
            'val': self.create_dataloader('val', val_split_file, shuffle=False),
            'test': self.create_dataloader('test', test_split_file, shuffle=False)
        }


def create_image_batch_tensor(images: List[np.ndarray]) -> torch.Tensor:
    """
    Convert list of numpy images to batch tensor
    
    Args:
        images: List of images (each H, W, C)
    
    Returns:
        Batch tensor (N, H, W, C) as torch.Tensor float32
    """
    image_array = np.array(images)
    return torch.from_numpy(image_array).float()


def get_class_distribution(dataset: BUSIDataset) -> Dict:
    """
    Get class distribution in dataset
    
    Args:
        dataset: BUSIDataset instance
    
    Returns:
        Dict with class distributions
    """
    unique, counts = np.unique(dataset.labels, return_counts=True)
    distribution = {}
    for class_id, count in zip(unique, counts):
        class_name = dataset.class_names[class_id]
        distribution[class_name] = {
            'count': int(count),
            'percentage': float(count / len(dataset) * 100)
        }
    return distribution


def save_dataloader_config(config: Dict, save_path: Union[str, Path]):
    """
    Save dataloader configuration to JSON
    
    Args:
        config: Configuration dict
        save_path: Path to save JSON
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_dataloader_config(config_path: Union[str, Path]) -> Dict:
    """
    Load dataloader configuration from JSON
    
    Args:
        config_path: Path to config JSON
    
    Returns:
        Configuration dict
    """
    with open(config_path, 'r') as f:
        return json.load(f)
