"""
Phase 5: Dataset Splitting
Stratified split of 780 preprocessed images into train (70%) / val (15%) / test (15%)
Output: Split manifests and verification report
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    from utils import ensure_dir, log_message
except ImportError:
    from preprocessing.utils import ensure_dir, log_message


def stratified_split(class_counts: Dict[str, int], 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     seed: int = 42) -> Dict[str, Dict[str, List[int]]]:
    """
    Generate stratified train/val/test split indices
    
    Args:
        class_counts: Dict with class names and image counts
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        Dict with split indices for each class
    """
    np.random.seed(seed)
    
    splits = {'train': {}, 'val': {}, 'test': {}}
    
    for class_name, count in class_counts.items():
        # Create indices
        indices = np.arange(count)
        np.random.shuffle(indices)
        
        # Calculate split points
        train_size = int(count * train_ratio)
        val_size = int(count * val_ratio)
        
        # Split
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        splits['train'][class_name] = train_idx.tolist()
        splits['val'][class_name] = val_idx.tolist()
        splits['test'][class_name] = test_idx.tolist()
    
    return splits


def execute_dataset_splitting():
    """Execute stratified dataset splitting"""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'datasets' / 'processed'
    manifest_dir = processed_dir / 'manifests'
    output_dir = project_root / 'preprocessing' / 'outputs'
    
    ensure_dir(manifest_dir)
    ensure_dir(output_dir)
    
    log_message("Phase 5", "Starting Phase 5: Dataset Splitting")
    log_message("Phase 5", f"{'='*60}")
    
    # Class counts
    class_counts = {
        'benign': 437,
        'malignant': 210,
        'normal': 133
    }
    
    log_message("Phase 5", "\n[Dataset Information]")
    for class_name, count in class_counts.items():
        log_message("Phase 5", f"  {class_name:10s}: {count:3d} images")
    log_message("Phase 5", f"  {'Total':10s}: {sum(class_counts.values()):3d} images")
    
    # Generate stratified split
    log_message("Phase 5", "\n[Generating Stratified Split]")
    log_message("Phase 5", "  Train: 70% (546 images)")
    log_message("Phase 5", "  Val:   15% (117 images)")
    log_message("Phase 5", "  Test:  15% (117 images)")
    
    splits = stratified_split(class_counts, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Verify split sizes
    split_stats = {'train': 0, 'val': 0, 'test': 0}
    for split_name in ['train', 'val', 'test']:
        for class_name, indices in splits[split_name].items():
            split_stats[split_name] += len(indices)
    
    log_message("Phase 5", "\n[Actual Split Sizes]")
    log_message("Phase 5", f"  Train: {split_stats['train']} images ({split_stats['train']/780*100:.1f}%)")
    log_message("Phase 5", f"  Val:   {split_stats['val']} images ({split_stats['val']/780*100:.1f}%)")
    log_message("Phase 5", f"  Test:  {split_stats['test']} images ({split_stats['test']/780*100:.1f}%)")
    
    # Write manifest files
    log_message("Phase 5", f"\n[Writing Manifest Files]")
    
    manifests = {}
    
    # First, collect actual filenames from the processed directory
    log_message("Phase 5", f"\n[Scanning Preprocessed Images]")
    actual_files = {}
    for class_name in ['benign', 'malignant', 'normal']:
        class_dir = processed_dir / 'images' / class_name
        if class_dir.exists():
            files = sorted(list(class_dir.glob('*.npy')))
            actual_files[class_name] = [f.name for f in files]
            log_message("Phase 5", f"  {class_name:10s}: {len(files)} files")
    
    for split_name in ['train', 'val', 'test']:
        manifest_lines = []
        
        # Create mapping of class names to label IDs
        class_to_id = {'benign': 0, 'malignant': 1, 'normal': 2}
        
        for class_name in ['benign', 'malignant', 'normal']:
            for img_idx in splits[split_name][class_name]:
                # Use actual filename from the processed directory
                if img_idx < len(actual_files[class_name]):
                    filename = actual_files[class_name][img_idx]
                    image_path = f"images/{class_name}/{filename}"
                    label = class_to_id[class_name]
                    manifest_lines.append(f"{image_path},{label},{class_name}")
        
        manifests[split_name] = manifest_lines
        
        # Write manifest file
        manifest_path = manifest_dir / f'{split_name}_manifest.txt'
        with open(manifest_path, 'w') as f:
            f.write('\n'.join(manifest_lines))
        
        log_message("Phase 5", f"  Wrote {split_name} manifest: {manifest_path.name} ({len(manifest_lines)} lines)")
    
    # Generate split statistics
    split_report = {
        'split_ratios': {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        },
        'split_sizes': split_stats,
        'by_class': {}
    }
    
    # Calculate per-class split
    log_message("Phase 5", f"\n[Split Distribution by Class]")
    for class_name in ['benign', 'malignant', 'normal']:
        class_id = {'benign': 0, 'malignant': 1, 'normal': 2}[class_name]
        total = class_counts[class_name]
        train_count = len(splits['train'][class_name])
        val_count = len(splits['val'][class_name])
        test_count = len(splits['test'][class_name])
        
        split_report['by_class'][class_name] = {
            'total': total,
            'train': train_count,
            'val': val_count,
            'test': test_count,
            'stratification_ratios': {
                'train': train_count / total if total > 0 else 0,
                'val': val_count / total if total > 0 else 0,
                'test': test_count / total if total > 0 else 0
            }
        }
        
        log_message("Phase 5", f"  {class_name:10s} | Total: {total:3d} | Train: {train_count:3d} ({train_count/total*100:5.1f}%) | Val: {val_count:3d} ({val_count/total*100:5.1f}%) | Test: {test_count:3d} ({test_count/total*100:5.1f}%)")
    
    # Verify stratification (ratios should be similar across classes)
    log_message("Phase 5", f"\n[Stratification Verification]")
    train_ratios = [split_report['by_class'][c]['stratification_ratios']['train'] for c in split_report['by_class']]
    mean_train_ratio = np.mean(train_ratios)
    std_train_ratio = np.std(train_ratios)
    log_message("Phase 5", f"  Train ratio: {mean_train_ratio:.3f} ± {std_train_ratio:.4f} (std dev)")
    log_message("Phase 5", f"  Stratification quality: {'GOOD' if std_train_ratio < 0.01 else 'ACCEPTABLE'}")
    
    # Save statistics
    stats_file = output_dir / 'phase5_split_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(split_report, f, indent=2)
    
    log_message("Phase 5", f"\n[Output Files]")
    log_message("Phase 5", f"  Train manifest: {manifest_dir / 'train_manifest.txt'}")
    log_message("Phase 5", f"  Val manifest:   {manifest_dir / 'val_manifest.txt'}")
    log_message("Phase 5", f"  Test manifest:  {manifest_dir / 'test_manifest.txt'}")
    log_message("Phase 5", f"  Statistics:     {stats_file}")
    
    log_message("Phase 5", f"\n{'='*60}")
    log_message("Phase 5", "✓ Phase 5 completed successfully")
    
    return split_report


if __name__ == '__main__':
    try:
        stats = execute_dataset_splitting()
        print("\nPhase 5 completed successfully!")
    except Exception as e:
        log_message("Phase 5", f"✗ Phase 5 failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
