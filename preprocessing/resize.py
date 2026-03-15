"""
Resize Module - Image Resizing
Resizes images to target size using bilinear interpolation
Parameters: target_size=(224, 224)
Input: Float32 or uint8 image (H, W)
Output: Resized float32 (224, 224)
"""

import numpy as np
from PIL import Image
from typing import Tuple, Union
from pathlib import Path


class Resizer:
    """Image resizing to target size"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize resizer
        
        Args:
            target_size: Target size (height, width) (default: (224, 224))
        """
        self.target_size = target_size
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size using bilinear interpolation
        
        Args:
            image: Input image (H, W) or (H, W, 3)
        
        Returns:
            Resized image to (target_h, target_w) or (target_h, target_w, 3)
        """
        target_h, target_w = self.target_size
        
        # Ensure uint8 for PIL
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            else:
                image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        # Convert to PIL Image
        if len(image.shape) == 2:
            # Greyscale
            pil_img = Image.fromarray(image_uint8, mode='L')
        else:
            # Color image
            pil_img = Image.fromarray(image_uint8)
        
        # Resize using bilinear interpolation
        pil_img_resized = pil_img.resize((target_w, target_h), Image.BILINEAR)
        
        # Convert back to numpy
        resized = np.array(pil_img_resized, dtype=image.dtype)
        
        # If input was float, convert back to float
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                resized = resized.astype(np.float32) / 255.0
            else:
                resized = resized.astype(np.float32)
        
        return resized
    
    def resize_from_file(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file and resize
        
        Args:
            image_path: Path to image file
        
        Returns:
            Resized image uint8
        """
        img = Image.open(image_path)
        image_array = np.array(img)
        return self.resize(image_array)
    
    def resize_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Resize batch of images
        
        Args:
            images: Batch of images (N, H, W) or (N, H, W, C)
        
        Returns:
            Resized batch
        """
        if len(images.shape) == 3:
            # Greyscale batch (N, H, W)
            n = images.shape[0]
            target_h, target_w = self.target_size
            resized = np.zeros((n, target_h, target_w), dtype=images.dtype)
            
            for i in range(n):
                resized[i] = self.resize(images[i])
            
            return resized
        
        elif len(images.shape) == 4:
            # Color batch (N, H, W, C)
            n = images.shape[0]
            c = images.shape[3]
            target_h, target_w = self.target_size
            resized = np.zeros((n, target_h, target_w, c), dtype=images.dtype)
            
            for i in range(n):
                resized[i] = self.resize(images[i])
            
            return resized
        
        else:
            raise ValueError(f"Unexpected batch shape: {images.shape}")
    
    def set_target_size(self, target_size: Tuple[int, int]) -> None:
        """Update target size"""
        self.target_size = target_size
    
    def get_target_size(self) -> Tuple[int, int]:
        """Get current target size"""
        return self.target_size
