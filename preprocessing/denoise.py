"""
Denoise Module - Anisotropic Diffusion Filter
Reduces noise while preserving edges using anisotropic diffusion
Parameters: iterations=10, conductance=30, time_step=0.1
Input: Single-channel uint8 greyscale (H, W)
Output: Denoised uint8 (H, W)
"""

import numpy as np
from scipy import ndimage
from typing import Union
from pathlib import Path
from PIL import Image


class Denoiser:
    """Anisotropic diffusion denoising"""
    
    def __init__(self, iterations: int = 10, conductance: float = 30, time_step: float = 0.1):
        """
        Initialize denoiser parameters
        
        Args:
            iterations: Number of diffusion iterations (default: 10)
            conductance: Conductance parameter for edge detection (default: 30)
            time_step: Time step for diffusion (default: 0.1)
        """
        self.iterations = iterations
        self.conductance = conductance
        self.time_step = time_step
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply anisotropic diffusion denoising
        
        Args:
            image: Input greyscale image uint8 (H, W)
        
        Returns:
            Denoised image uint8 (H, W)
        """
        # Convert to float for processing
        img = image.astype(np.float32)
        
        # Apply iterative anisotropic diffusion
        for iteration in range(self.iterations):
            # Compute gradients in 4 directions (N, S, E, W)
            n = ndimage.shift(img, 1, mode='constant', cval=0)  # North
            s = ndimage.shift(img, -1, mode='constant', cval=0)  # South
            e = ndimage.shift(img, (0, -1), mode='constant', cval=0)  # East
            w = ndimage.shift(img, (0, 1), mode='constant', cval=0)  # West
            
            # Compute differences
            dn = n - img
            ds = s - img
            de = e - img
            dw = w - img
            
            # Conductance-controlled diffusion
            # g(∇I) = 1 / (1 + (∇I / κ)^2)
            cn = 1.0 / (1.0 + (dn / self.conductance) ** 2)
            cs = 1.0 / (1.0 + (ds / self.conductance) ** 2)
            ce = 1.0 / (1.0 + (de / self.conductance) ** 2)
            cw = 1.0 / (1.0 + (dw / self.conductance) ** 2)
            
            # Diffusion step
            img = img + self.time_step * (
                cn * dn + cs * ds + ce * de + cw * dw
            )
        
        # Convert back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def denoise_from_file(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file and denoise
        
        Args:
            image_path: Path to image file
        
        Returns:
            Denoised image uint8
        """
        img = Image.open(image_path)
        image_array = np.array(img)
        
        # Convert to greyscale if needed
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2).astype(np.uint8)
        
        return self.denoise(image_array)
    
    def set_parameters(self, iterations: int = None, conductance: float = None, 
                      time_step: float = None) -> None:
        """Update denoising parameters"""
        if iterations is not None:
            self.iterations = iterations
        if conductance is not None:
            self.conductance = conductance
        if time_step is not None:
            self.time_step = time_step
    
    def get_parameters(self) -> dict:
        """Get current parameters"""
        return {
            "iterations": self.iterations,
            "conductance": self.conductance,
            "time_step": self.time_step
        }
