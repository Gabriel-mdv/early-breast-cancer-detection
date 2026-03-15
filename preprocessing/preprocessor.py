"""
Main Preprocessor Module  
Orchestrates complete preprocessing pipeline
Combines all preprocessing steps into single pipeline
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
import json
import sys

# Handle both relative and direct imports
try:
    from .format_standardizer import FormatStandardizer
    from .denoise import Denoiser
    from .contrast import ContrastEnhancer
    from .normalization import Normalizer
    from .resize import Resizer
    from .augmentation import DataAugmenter
    from .utils import ensure_dir, load_image, save_image, log_message
except ImportError:
    from format_standardizer import FormatStandardizer
    from denoise import Denoiser
    from contrast import ContrastEnhancer
    from normalization import Normalizer
    from resize import Resizer
    from augmentation import DataAugmenter
    from utils import ensure_dir, load_image, save_image, log_message


class ImagePreprocessor:
    """Main preprocessing pipeline orchestrator"""
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 seed: Optional[int] = None):
        """
        Initialize preprocessor with all modules
        
        Args:
            config: Configuration dict with module parameters
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize all modules
        self.standardizer = FormatStandardizer()
        self.denoiser = Denoiser(
            iterations=10,
            conductance=30,
            time_step=0.1
        )
        self.contrast_enhancer = ContrastEnhancer(
            clip_limit=2.0,
            tile_grid=(8, 8)
        )
        self.normalizer = Normalizer()
        self.resizer = Resizer(target_size=(224, 224))
        self.augmenter = DataAugmenter(seed=seed)
        
        # Store config
        self.config = config or self._get_default_config()
        self.execution_log = []
    
    def _get_default_config(self) -> Dict:
        """Get default preprocessing configuration"""
        return {
            'standardize': True,
            'denoise': True,
            'enhance_contrast': True,
            'normalize': True,
            'resize': True,
            'augment': False,  # Only during training
            'augmentation_config': DataAugmenter.get_default_augmentation_config(),
            'return_format': 'float32',  # 'float32' or 'uint8'
            'replicate_to_3channel': True  # Replicate greyscale to 3 channels
        }
    
    def preprocess(self, image: np.ndarray, augment: bool = False) -> np.ndarray:
        """
        Apply complete preprocessing pipeline
        
        Args:
            image: Input image (any format)
            augment: Whether to apply augmentation (for training)
        
        Returns:
            Preprocessed image (224, 224, 3) float32 or uint8
        """
        processed = image.copy()
        
        # Step 1: Standardize to greyscale uint8
        if self.config['standardize']:
            processed = self.standardizer.standardize(processed)
        
        # Step 2: Denoise
        if self.config['denoise']:
            processed = self.denoiser.denoise(processed)
            # Ensure uint8 after denoising
            processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        # Step 3: Enhance contrast
        if self.config['enhance_contrast']:
            processed = self.contrast_enhancer.enhance(processed)
        
        # Step 4: Normalize
        if self.config['normalize']:
            processed = self.normalizer.normalize(processed)
        
        # Step 5: Resize
        if self.config['resize']:
            processed = self.resizer.resize(processed)
        
        # Step 6: Augmentation (only if requested)
        if augment and self.config['augment']:
            # Need to convert back to uint8 for augmentation
            if processed.dtype != np.uint8:
                processed_uint8 = np.clip(processed * 255, 0, 255).astype(np.uint8)
            else:
                processed_uint8 = processed
            
            processed_uint8 = self.augmenter.augment(
                processed_uint8,
                self.config['augmentation_config']
            )
            
            # Normalize again after augmentation
            if self.config['normalize']:
                processed = self.normalizer.normalize(processed_uint8)
            else:
                processed = processed_uint8.astype(np.float32) / 255.0
        
        # Step 7: Replicate to 3 channels if currently single channel
        if self.config['replicate_to_3channel']:
            if len(processed.shape) == 2 or processed.shape[-1] == 1:
                if len(processed.shape) == 3:
                    processed = processed[:, :, 0]
                # Replicate to 3 channels
                processed = np.stack([processed] * 3, axis=-1)
        
        # Step 8: Convert to requested format
        if self.config['return_format'] == 'uint8':
            if processed.dtype != np.uint8:
                processed = np.clip(processed * 255, 0, 255).astype(np.uint8)
        else:  # float32
            if processed.dtype == np.uint8:
                processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def preprocess_from_file(self, image_path: Union[str, Path], 
                            augment: bool = False) -> np.ndarray:
        """
        Load image and apply preprocessing
        
        Args:
            image_path: Path to image file
            augment: Whether to apply augmentation
        
        Returns:
            Preprocessed image
        """
        image = load_image(image_path)
        return self.preprocess(image, augment=augment)
    
    def preprocess_batch(self, images: np.ndarray, augment: bool = False) -> np.ndarray:
        """
        Preprocess batch of images
        
        Args:
            images: Batch of images (N, H, W) or (N, H, W, C)
            augment: Whether to apply augmentation
        
        Returns:
            Batch of preprocessed images (N, 224, 224, 3)
        """
        batch_size = images.shape[0]
        processed_batch = []
        
        for i in range(batch_size):
            processed = self.preprocess(images[i], augment=augment)
            processed_batch.append(processed)
        
        return np.array(processed_batch)
    
    def preprocess_directory(self, input_dir: Union[str, Path],
                            output_dir: Union[str, Path],
                            augment: bool = False,
                            save_format: str = 'npy') -> Dict:
        """
        Preprocess all images in directory
        
        Args:
            input_dir: Input directory with images
            output_dir: Output directory for preprocessed images
            augment: Whether to apply augmentation
            save_format: 'npy', 'jpg', or 'png'
        
        Returns:
            Statistics dict
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        # Support common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        for image_path in image_files:
            try:
                # Load and preprocess
                image = load_image(image_path)
                processed = self.preprocess(image, augment=augment)
                
                # Save
                output_name = image_path.stem
                if save_format == 'npy':
                    output_path = output_dir / f'{output_name}.npy'
                    np.save(output_path, processed)
                else:
                    output_path = output_dir / f'{output_name}.{save_format}'
                    save_image(output_path, processed)
                
                stats['successful'] += 1
                self.execution_log.append({
                    'file': str(image_path),
                    'status': 'success',
                    'output': str(output_path),
                    'shape': processed.shape
                })
                
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append(str(e))
                self.execution_log.append({
                    'file': str(image_path),
                    'status': 'failed',
                    'error': str(e)
                })
            
            stats['total_processed'] += 1
        
        return stats
    
    def set_config(self, config: Dict):
        """Update configuration"""
        self.config = config
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.copy()
    
    def get_execution_log(self) -> list:
        """Get execution log"""
        return self.execution_log.copy()
    
    def clear_execution_log(self):
        """Clear execution log"""
        self.execution_log = []
    
    def save_config(self, config_path: Union[str, Path]):
        """Save configuration to JSON"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_config(self, config_path: Union[str, Path]):
        """Load configuration from JSON"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_pipeline_info(self) -> Dict:
        """Get information about current pipeline configuration"""
        return {
            'standardizer': self.standardizer.get_parameters(),
            'denoiser': self.denoiser.get_parameters(),
            'contrast_enhancer': self.contrast_enhancer.get_parameters(),
            'normalizer': self.normalizer.get_parameters(),
            'resizer': self.resizer.get_parameters(),
            'config': self.get_config()
        }
