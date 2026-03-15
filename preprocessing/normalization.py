"""
Normalization Module - ImageNet Standardization
Normalizes images to ImageNet statistics (mean, std)
Parameters: mean=0.485, std=0.229 (single channel: using average)
Input: uint8 image (H, W) [0-255]
Output: float32 normalized (H, W) [-2.2 to 2.2]
"""

import numpy as np
from typing import Union, Tuple
from pathlib import Path
from PIL import Image


class Normalizer:
    """ImageNet normalization"""
    
    def __init__(self, mean: float = 0.485, std: float = 0.229):
        """
        Initialize normalization parameters
        
        Args:
            mean: Mean for normalization (default: 0.485 - ImageNet R channel)
            std: Standard deviation for normalization (default: 0.229 - ImageNet R channel)
        """
        self.mean = mean
        self.std = std
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply ImageNet normalization
        
        Args:
            image: Input image uint8 (H, W) [0-255]
        
        Returns:
            Normalized float32 (H, W) in range [-mean/std to (255-mean)/std]
        """
        # Ensure float32
        img = image.astype(np.float32)
        
        # Normalize to [0, 1] range first
        img = img / 255.0
        
        # Apply ImageNet normalization: (x - mean) / std
        normalized = (img - self.mean) / self.std
        
        return normalized
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Reverse normalization to get original image
        
        Args:
            image: Normalized float32 image
        
        Returns:
            Original uint8 image [0-255]
        """
        # Reverse: x = (normalized * std) + mean
        img = (image * self.std) + self.mean
        
        # Convert back to [0, 255]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        return img
    
    def normalize_from_file(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file and normalize
        
        Args:
            image_path: Path to image file
        
        Returns:
            Normalized float32 image
        """
        img = Image.open(image_path)
        image_array = np.array(img)
        
        # Convert to greyscale if needed
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2).astype(np.uint8)
        
        return self.normalize(image_array)
    
    def normalize_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Normalize batch of images
        
        Args:
            images: Batch of images (N, H, W) uint8
        
        Returns:
            Normalized batch (N, H, W) float32
        """
        normalized = np.zeros_like(images, dtype=np.float32)
        for i in range(images.shape[0]):
            normalized[i] = self.normalize(images[i])
        return normalized
    
    def set_parameters(self, mean: float = None, std: float = None) -> None:
        """Update normalization parameters"""
        if mean is not None:
            self.mean = mean
        if std is not None:
            self.std = std
    
    def get_parameters(self) -> dict:
        """Get current parameters"""
        return {
            "mean": self.mean,
            "std": self.std
        }
    
    @staticmethod
    def get_imagenet_stats() -> dict:
        """
        Get standard ImageNet normalization values
        
        Returns:
            Dict with R, G, B means and stds
        """
        return {
            "mean": {
                "R": 0.485,
                "G": 0.456,
                "B": 0.406,
                "average": 0.449  # For single channel
            },
            "std": {
                "R": 0.229,
                "G": 0.224,
                "B": 0.225,
                "average": 0.226  # For single channel
            }
        }
