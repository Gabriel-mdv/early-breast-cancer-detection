"""
Phase 4: Execute Preprocessing Pipeline
Process all 780 raw BUSI images through complete pipeline
Outputs: preprocessed images (224x224x3) + resized masks
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# Import only what we need for Phase 4
import sys
sys.path.insert(0, str(Path(__file__).parent))

from preprocessor import ImagePreprocessor
from utils import ensure_dir, load_image, save_image, log_message


def execute_preprocessing_pipeline():
    """Execute complete preprocessing for all 780 images"""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    datasets_dir = project_root / 'datasets'
    processed_dir = project_root / 'datasets' / 'processed'
    output_dir = project_root / 'preprocessing' / 'outputs'
    
    # Create output directories
    ensure_dir(processed_dir / 'images' / 'benign')
    ensure_dir(processed_dir / 'images' / 'malignant')
    ensure_dir(processed_dir / 'images' / 'normal')
    ensure_dir(processed_dir / 'masks' / 'benign')
    ensure_dir(processed_dir / 'masks' / 'malignant')
    ensure_dir(processed_dir / 'masks' / 'normal')
    ensure_dir(output_dir)
    
    # Initialize preprocessor
    log_message("Phase 4", "Initializing ImagePreprocessor...")
    preprocessor = ImagePreprocessor(seed=42)
    
    # Statistics
    stats = {
        'start_time': time.time(),
        'total_images': 0,
        'successful': 0,
        'failed': 0,
        'by_class': defaultdict(lambda: {'total': 0, 'successful': 0, 'failed': 0}),
        'errors': [],
        'class_file_counts': {}
    }
    
    # Process each class
    classes = ['benign', 'malignant', 'normal']
    
    for class_name in classes:
        log_message("Phase 4", f"\n{'='*60}")
        log_message("Phase 4", f"Processing class: {class_name.upper()}")
        log_message("Phase 4", f"{'='*60}")
        
        class_dir = datasets_dir / class_name
        
        if not class_dir.exists():
            log_message("Phase 4", f"ERROR: Class directory not found: {class_dir}")
            continue
        
        # Get all image files
        image_files = sorted(list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')))
        
        # Filter out mask files (they end with _mask)
        image_files = [f for f in image_files if '_mask' not in f.stem]
        
        log_message("Phase 4", f"Found {len(image_files)} images in {class_name}")
        stats['class_file_counts'][class_name] = len(image_files)
        
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            try:
                # Load image
                image = load_image(image_path)
                
                # Preprocess
                preprocessed = preprocessor.preprocess(image, augment=False)
                
                # Save preprocessed image
                output_image_path = processed_dir / 'images' / class_name / f"{image_path.stem}.npy"
                np.save(output_image_path, preprocessed.astype(np.float32))
                
                # Process and save mask if exists
                mask_path = image_path.parent / f"{image_path.stem}_mask.png"
                if mask_path.exists():
                    # Load mask (binary)
                    mask = load_image(mask_path)
                    
                    # Convert to binary if needed
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    
                    # Normalize to [0, 1]
                    if mask.max() > 1:
                        mask = mask.astype(np.float32) / 255.0
                    else:
                        mask = mask.astype(np.float32)
                    
                    # Resize mask to 224x224 using nearest neighbor
                    from PIL import Image
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    pil_mask = Image.fromarray(mask_uint8)
                    pil_mask_resized = pil_mask.resize((224, 224), Image.NEAREST)
                    mask_resized = np.array(pil_mask_resized).astype(np.float32) / 255.0
                    
                    # Save resized mask
                    mask_output_path = processed_dir / 'masks' / class_name / f"{image_path.stem}_mask.npy"
                    np.save(mask_output_path, mask_resized)
                
                stats['successful'] += 1
                stats['by_class'][class_name]['successful'] += 1
                
                if idx % 50 == 0 or idx == len(image_files):
                    progress = (idx / len(image_files)) * 100
                    log_message("Phase 4", f"  [{class_name}] {idx}/{len(image_files)} ({progress:.1f}%)")
                
            except Exception as e:
                stats['failed'] += 1
                stats['by_class'][class_name]['failed'] += 1
                error_msg = f"{image_path.name}: {str(e)}"
                stats['errors'].append(error_msg)
                log_message("Phase 4", f"  ERROR - {error_msg}")
            
            stats['total_images'] += 1
            stats['by_class'][class_name]['total'] += 1
        
        log_message("Phase 4", f"Completed {class_name}: {stats['by_class'][class_name]['successful']}/{len(image_files)} successful")
    
    # End timing
    stats['end_time'] = time.time()
    stats['execution_time_seconds'] = stats['end_time'] - stats['start_time']
    
    # Summary
    log_message("Phase 4", f"\n{'='*60}")
    log_message("Phase 4", "PREPROCESSING COMPLETE")
    log_message("Phase 4", f"{'='*60}")
    log_message("Phase 4", f"Total images: {stats['total_images']}")
    log_message("Phase 4", f"Successful: {stats['successful']}")
    log_message("Phase 4", f"Failed: {stats['failed']}")
    log_message("Phase 4", f"Success rate: {stats['successful']/stats['total_images']*100:.2f}%")
    log_message("Phase 4", f"Execution time: {stats['execution_time_seconds']:.2f}s")
    log_message("Phase 4", f"Average time per image: {stats['execution_time_seconds']/stats['total_images']:.3f}s")
    
    log_message("Phase 4", f"\nBreakdown by class:")
    for class_name in classes:
        class_stats = stats['by_class'][class_name]
        if class_stats['total'] > 0:
            success_rate = (class_stats['successful'] / class_stats['total']) * 100
            log_message("Phase 4", f"  {class_name:10s}: {class_stats['successful']:3d}/{class_stats['total']:3d} ({success_rate:5.1f}%)")
    
    if stats['errors']:
        log_message("Phase 4", f"\nRecorded {len(stats['errors'])} errors")
    
    # Save statistics
    stats_file = output_dir / 'phase4_preprocessing_log.json'
    
    # Convert to serializable format
    stats_serializable = {
        'start_time': stats['start_time'],
        'end_time': stats['end_time'],
        'total_images': stats['total_images'],
        'successful': stats['successful'],
        'failed': stats['failed'],
        'success_rate': stats['successful'] / stats['total_images'] if stats['total_images'] > 0 else 0,
        'execution_time_seconds': stats['execution_time_seconds'],
        'avg_time_per_image': stats['execution_time_seconds'] / stats['total_images'] if stats['total_images'] > 0 else 0,
        'by_class': {
            class_name: {
                'total': stats['by_class'][class_name]['total'],
                'successful': stats['by_class'][class_name]['successful'],
                'failed': stats['by_class'][class_name]['failed'],
                'success_rate': (stats['by_class'][class_name]['successful'] / stats['by_class'][class_name]['total'] 
                                if stats['by_class'][class_name]['total'] > 0 else 0)
            }
            for class_name in classes
        },
        'class_file_counts': stats['class_file_counts'],
        'errors': stats['errors']
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    log_message("Phase 4", f"\nStatistics saved to: {stats_file}")
    
    # Verify outputs
    log_message("Phase 4", f"\n{'='*60}")
    log_message("Phase 4", "VERIFYING OUTPUTS")
    log_message("Phase 4", f"{'='*60}")
    
    for class_name in classes:
        img_count = len(list((processed_dir / 'images' / class_name).glob('*.npy')))
        mask_count = len(list((processed_dir / 'masks' / class_name).glob('*_mask.npy')))
        log_message("Phase 4", f"{class_name:10s}: {img_count:3d} images, {mask_count:3d} masks")
    
    total_processed = sum(len(list((processed_dir / 'images' / cn).glob('*.npy'))) for cn in classes)
    log_message("Phase 4", f"\nTotal preprocessed images: {total_processed}")
    
    return stats_serializable


if __name__ == '__main__':
    log_message("Phase 4", "Starting Phase 4: Execute Preprocessing Pipeline")
    log_message("Phase 4", f"Python version: {os.sys.version}")
    
    # Change to preprocessing directory for proper imports
    os.chdir(Path(__file__).parent)
    
    try:
        stats = execute_preprocessing_pipeline()
        log_message("Phase 4", "\n✓ Phase 4 completed successfully")
    except Exception as e:
        log_message("Phase 4", f"\n✗ Phase 4 failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
