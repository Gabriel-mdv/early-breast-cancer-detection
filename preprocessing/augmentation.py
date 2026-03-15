"""
Data Augmentation Module
Training data augmentation transformations
Supported: horizontal flip, rotation, brightness/contrast, translation
"""

import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Union, Optional
from pathlib import Path


class DataAugmenter:
    """Data augmentation for training"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize augmenter
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
    
    def horizontal_flip(self, image: np.ndarray, probability: float = 0.5) -> np.ndarray:
        """
        Randomly flip image horizontally
        
        Args:
            image: Input image (H, W) or (H, W, C)
            probability: Probability of flipping (0-1)
        
        Returns:
            Flipped or original image
        """
        if np.random.random() < probability:
            return np.fliplr(image)
        return image
    
    def vertical_flip(self, image: np.ndarray, probability: float = 0.5) -> np.ndarray:
        """Randomly flip image vertically"""
        if np.random.random() < probability:
            return np.flipud(image)
        return image
    
    def rotate(self, image: np.ndarray, angle_range: Tuple[float, float] = (-15, 15),
              probability: float = 0.5) -> np.ndarray:
        """
        Randomly rotate image
        
        Args:
            image: Input image uint8
            angle_range: Range of rotation angles in degrees
            probability: Probability of rotation
        
        Returns:
            Rotated or original image
        """
        if np.random.random() < probability:
            angle = np.random.uniform(angle_range[0], angle_range[1])
            
            # Convert to PIL Image
            if image.dtype != np.uint8:
                img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            else:
                img_uint8 = image
            
            if len(image.shape) == 2:
                pil_img = Image.fromarray(img_uint8, mode='L')
            else:
                pil_img = Image.fromarray(img_uint8)
            
            # Rotate
            pil_img_rotated = pil_img.rotate(angle, expand=False, fillcolor=0)
            rotated = np.array(pil_img_rotated, dtype=image.dtype)
            
            return rotated
        return image
    
    def brightness_contrast(self, image: np.ndarray, 
                          factor_range: Tuple[float, float] = (0.8, 1.2),
                          probability: float = 0.5) -> np.ndarray:
        """
        Randomly adjust brightness and contrast
        
        Args:
            image: Input image uint8
            factor_range: Range of adjustment factors
            probability: Probability of adjustment
        
        Returns:
            Adjusted or original image
        """
        if np.random.random() < probability:
            brightness_factor = np.random.uniform(factor_range[0], factor_range[1])
            contrast_factor = np.random.uniform(factor_range[0], factor_range[1])
            
            # Convert to PIL Image
            if image.dtype != np.uint8:
                img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            else:
                img_uint8 = image
            
            if len(image.shape) == 2:
                pil_img = Image.fromarray(img_uint8, mode='L')
            else:
                pil_img = Image.fromarray(img_uint8)
            
            # Adjust brightness
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(brightness_factor)
            
            # Adjust contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(contrast_factor)
            
            adjusted = np.array(pil_img, dtype=image.dtype)
            return adjusted
        return image
    
    def translate(self, image: np.ndarray, 
                 max_shift: Union[float, Tuple[float, float]] = 0.1,
                 probability: float = 0.5) -> np.ndarray:
        """
        Randomly translate (shift) image
        
        Args:
            image: Input image
            max_shift: Maximum shift as fraction of image size
            probability: Probability of translation
        
        Returns:
            Translated or original image
        """
        if np.random.random() < probability:
            h, w = image.shape[:2]
            
            if isinstance(max_shift, (int, float)):
                max_shift_h = max_shift_w = max_shift
            else:
                max_shift_h, max_shift_w = max_shift
            
            shift_h = int(np.random.uniform(-max_shift_h * h, max_shift_h * h))
            shift_w = int(np.random.uniform(-max_shift_w * w, max_shift_w * w))
            
            # Convert to PIL for affine transformation
            if image.dtype != np.uint8:
                img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            else:
                img_uint8 = image
            
            if len(image.shape) == 2:
                pil_img = Image.fromarray(img_uint8, mode='L')
            else:
                pil_img = Image.fromarray(img_uint8)
            
            # Apply translation using affine transform
            from PIL import ImageOps
            translated_pil = ImageOps.expand(pil_img, border=0)
            
            # Simple numpy-based translation
            translated = np.roll(np.roll(image, shift_h, axis=0), shift_w, axis=1)
            
            return translated
        return image
    
    def augment(self, image: np.ndarray, augmentation_config: dict) -> np.ndarray:
        """
        Apply multiple augmentations based on config
        
        Args:
            image: Input image
            augmentation_config: Dict with augmentation parameters
                {
                    'horizontal_flip': {'probability': 0.5},
                    'rotate': {'angle_range': (-15, 15), 'probability': 0.5},
                    'brightness_contrast': {'factor_range': (0.8, 1.2), 'probability': 0.3},
                    'translate': {'max_shift': 0.1, 'probability': 0.3}
                }
        
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        if 'horizontal_flip' in augmentation_config:
            prob = augmentation_config['horizontal_flip'].get('probability', 0.5)
            augmented = self.horizontal_flip(augmented, prob)
        
        if 'rotate' in augmentation_config:
            config = augmentation_config['rotate']
            angle_range = config.get('angle_range', (-15, 15))
            prob = config.get('probability', 0.5)
            augmented = self.rotate(augmented, angle_range, prob)
        
        if 'brightness_contrast' in augmentation_config:
            config = augmentation_config['brightness_contrast']
            factor_range = config.get('factor_range', (0.8, 1.2))
            prob = config.get('probability', 0.3)
            augmented = self.brightness_contrast(augmented, factor_range, prob)
        
        if 'translate' in augmentation_config:
            config = augmentation_config['translate']
            max_shift = config.get('max_shift', 0.1)
            prob = config.get('probability', 0.3)
            augmented = self.translate(augmented, max_shift, prob)
        
        return augmented
    
    @staticmethod
    def get_default_augmentation_config() -> dict:
        """
        Get default augmentation configuration
        
        Returns:
            Dict with default augmentation parameters
        """
        return {
            'horizontal_flip': {'probability': 0.5},
            'rotate': {'angle_range': (-15, 15), 'probability': 0.5},
            'brightness_contrast': {'factor_range': (0.8, 1.2), 'probability': 0.3},
            'translate': {'max_shift': 0.1, 'probability': 0.3}
        }
