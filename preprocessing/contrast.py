"""
Contrast Enhancement Module - CLAHE
Contrast Limited Adaptive Histogram Equalization
Parameters: clip_limit=2.0, tile_grid=(8, 8)
Input: Single-channel uint8 (H, W)
Output: Enhanced contrast uint8 (H, W)
"""

import numpy as np
from skimage import exposure
from typing import Tuple, Union
from pathlib import Path
from PIL import Image


class ContrastEnhancer:
    """CLAHE contrast enhancement"""
    
    def __init__(self, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)):
        """
        Initialize CLAHE parameters
        
        Args:
            clip_limit: Clip limit for histogram (default: 2.0)
            tile_grid: Number of tiles (rows, cols) (default: (8, 8))
        """
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE contrast enhancement
        
        Args:
            image: Input greyscale image uint8 (H, W)
        
        Returns:
            Enhanced image uint8 (H, W)
        """
        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Apply CLAHE
        enhanced = exposure.equalize_adapthist(
            image,
            kernel_size=self.tile_grid,
            clip_limit=self.clip_limit,
            nbins=256
        )
        
        # Convert back to uint8 [0, 255]
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def enhance_from_file(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file and enhance contrast
        
        Args:
            image_path: Path to image file
        
        Returns:
            Enhanced image uint8
        """
        img = Image.open(image_path)
        image_array = np.array(img)
        
        # Convert to greyscale if needed
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2).astype(np.uint8)
        
        return self.enhance(image_array)
    
    def set_parameters(self, clip_limit: float = None, 
                      tile_grid: Tuple[int, int] = None) -> None:
        """Update CLAHE parameters"""
        if clip_limit is not None:
            self.clip_limit = clip_limit
        if tile_grid is not None:
            self.tile_grid = tile_grid
    
    def get_parameters(self) -> dict:
        """Get current parameters"""
        return {
            "clip_limit": self.clip_limit,
            "tile_grid": self.tile_grid
        }
