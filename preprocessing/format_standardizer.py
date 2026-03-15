"""
Format Standardizer
Standardizes image formats: RGB to greyscale conversion, format consistency
Input: Raw PNG image file (may be 3-channel or 1-channel)
Output: Single-channel greyscale uint8 (H, W)
"""

import numpy as np
from PIL import Image
from typing import Tuple, Union
from pathlib import Path


class FormatStandardizer:
    """Convert and standardize image formats to greyscale"""
    
    def __init__(self):
        self.channels_processed = {
            "single_channel": 0,
            "three_channel": 0,
            "four_channel": 0,
            "other": 0
        }
    
    def standardize(self, image: np.ndarray) -> np.ndarray:
        """
        Standardize image to single-channel greyscale uint8
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
        
        Returns:
            Single-channel uint8 greyscale image (H, W)
        """
        # Ensure image is numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Handle different input formats
        if len(image.shape) == 2:
            # Already single channel
            self.channels_processed["single_channel"] += 1
            greyscale = image
        
        elif len(image.shape) == 3:
            channels = image.shape[2]
            
            if channels == 1:
                # Single channel with explicit dimension
                self.channels_processed["single_channel"] += 1
                greyscale = image[:, :, 0]
            
            elif channels == 3:
                # RGB to greyscale using luminosity formula
                self.channels_processed["three_channel"] += 1
                greyscale = self._rgb_to_greyscale(image)
            
            elif channels == 4:
                # RGBA - convert to greyscale (ignore alpha)
                self.channels_processed["four_channel"] += 1
                rgb_image = image[:, :, :3]
                greyscale = self._rgb_to_greyscale(rgb_image)
            
            else:
                # Unexpected number of channels
                self.channels_processed["other"] += 1
                # Average all channels
                greyscale = np.mean(image, axis=2)
        
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Ensure uint8 format
        if greyscale.dtype != np.uint8:
            if greyscale.max() <= 1.0:
                greyscale = (greyscale * 255).astype(np.uint8)
            else:
                greyscale = np.clip(greyscale, 0, 255).astype(np.uint8)
        
        return greyscale
    
    def standardize_from_file(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file and standardize
        
        Args:
            image_path: Path to image file
        
        Returns:
            Single-channel uint8 greyscale image
        """
        img = Image.open(image_path)
        image_array = np.array(img)
        return self.standardize(image_array)
    
    @staticmethod
    def _rgb_to_greyscale(rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to greyscale using luminosity formula
        Weights: R=0.299, G=0.587, B=0.114
        
        Args:
            rgb_image: RGB image (H, W, 3)
        
        Returns:
            Greyscale image (H, W)
        """
        # Extract channels
        r = rgb_image[:, :, 0]
        g = rgb_image[:, :, 1]
        b = rgb_image[:, :, 2]
        
        # Apply luminosity formula
        greyscale = 0.299 * r + 0.587 * g + 0.114 * b
        
        return greyscale.astype(np.float32)
    
    def get_statistics(self) -> dict:
        """Return processing statistics"""
        return self.channels_processed.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics"""
        self.channels_processed = {
            "single_channel": 0,
            "three_channel": 0,
            "four_channel": 0,
            "other": 0
        }
