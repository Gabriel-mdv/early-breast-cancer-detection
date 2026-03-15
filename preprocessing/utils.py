"""
Utility Functions for Preprocessing Pipeline
Common helpers, file operations, and utility functions
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Dict, Any
from PIL import Image


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load image from file"""
    img = Image.open(image_path)
    return np.array(img)


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """Save image array to file"""
    ensure_dir(Path(output_path).parent)
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    Image.fromarray(image).save(output_path)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 range [0, 255]"""
    if image.dtype == np.uint8:
        return image
    
    if image.max() <= 1.0:
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        return np.clip(image, 0, 255).astype(np.uint8)


def normalize_to_float(image: np.ndarray) -> np.ndarray:
    """Normalize image to float range [0, 1]"""
    if image.dtype == np.uint8 or image.max() > 1.0:
        return image.astype(np.float32) / 255.0
    else:
        return image.astype(np.float32)


def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
    """Get image height and width"""
    if len(image.shape) == 2:
        return image.shape[0], image.shape[1]
    else:
        return image.shape[0], image.shape[1]


def validate_image_shape(image: np.ndarray, expected_shape: Tuple) -> bool:
    """Validate if image matches expected shape"""
    return image.shape == expected_shape


def log_message(message: str, level: str = "INFO") -> None:
    """Print formatted log message"""
    print(f"[{level:8s}] {message}")


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save dictionary to JSON file"""
    ensure_dir(Path(filepath).parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def count_files_in_directory(path: Union[str, Path], extension: str = "*.png") -> int:
    """Count files with specific extension in directory"""
    path = Path(path)
    return len(list(path.glob(extension)))


def get_class_directories(root_path: Union[str, Path]) -> Dict[str, Path]:
    """Get all class subdirectories from root path"""
    root_path = Path(root_path)
    classes = {}
    for class_dir in sorted(root_path.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            classes[class_dir.name] = class_dir
    return classes
