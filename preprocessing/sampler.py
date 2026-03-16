"""
Class-Aware Sampler for Augmentation-Based Oversampling
Implements inverse class frequency weighting for imbalanced datasets
"""

import numpy as np
from torch.utils.data import Sampler
from typing import Iterator, Optional


class ClassAwareSampler(Sampler):
    """
    Sampler that applies class-aware sampling probabilities.
    
    Minority classes (malignant, normal) are sampled more frequently than
    the majority class (benign) using inverse class frequency weighting.
    
    Each epoch sees the full dataset but distributed according to inverse
    class frequencies, allowing augmentation to expose the model to more
    minority class variations.
    
    Example:
        >>> sampler = ClassAwareSampler(train_dataset)
        >>> train_loader = DataLoader(train_dataset, sampler=sampler)
    """
    
    def __init__(self, dataset, num_samples: Optional[int] = None):
        """
        Initialize class-aware sampler.
        
        Args:
            dataset: BUSIDataset instance (must have .labels attribute)
            num_samples: Total samples to draw per epoch (None = len(dataset))
                        When None, draws len(dataset) samples with replacement
        """
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        
        # Get unique labels and their counts
        labels_array = np.array(self.dataset.labels)
        unique_labels = np.unique(labels_array)
        class_counts = np.bincount(labels_array)[unique_labels]
        
        # Inverse class frequency weighting
        # Classes with fewer samples get higher sampling probability
        weights = 1.0 / class_counts.astype(float)
        weights = weights / weights.sum()  # Normalize to sum=1
        
        # Assign sampling weight to each sample based on its class
        self.weights = np.array([weights[label] for label in labels_array])
        self.weights = self.weights / self.weights.sum()  # Re-normalize
        
        # Class distribution for logging
        self.class_distribution = dict(zip(unique_labels, class_counts))
        self.class_weights = dict(zip(unique_labels, weights[unique_labels]))
    
    def __iter__(self) -> Iterator:
        """
        Yields indices sampled according to class weights.
        
        Samples with replacement to allow minority classes to appear
        multiple times in the same epoch.
        """
        indices = np.random.choice(
            len(self.dataset),
            size=self.num_samples,
            replace=True,  # Allow repeats for upsampling
            p=self.weights
        )
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        """Return total number of samples per epoch."""
        return self.num_samples
    
    def get_sampling_probabilities(self) -> dict:
        """
        Returns class-wise sampling probabilities.
        
        Returns:
            Dict mapping class_id to sampling probability
        """
        return self.class_weights
    
    def get_class_distribution(self) -> dict:
        """
        Returns original class distribution in dataset.
        
        Returns:
            Dict mapping class_id to count
        """
        return self.class_distribution


class WeightedRandomSampler(Sampler):
    """
    Alternative: Weighted random sampler based on pre-computed weights.
    
    Useful when you want to specify custom weights directly.
    """
    
    def __init__(self, weights: np.ndarray, num_samples: int, replacement: bool = True):
        """
        Initialize weighted sampler.
        
        Args:
            weights: Array of weights (length = dataset size)
            num_samples: Number of samples to draw per epoch
            replacement: Whether to allow repeats
        """
        self.weights = weights / weights.sum()  # Normalize
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self) -> Iterator:
        """Yields indices sampled according to weights."""
        indices = np.random.choice(
            len(self.weights),
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights
        )
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.num_samples
