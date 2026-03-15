"""
Phase 2: Class Distribution Analysis & Class Weights Computation
Analyzes dataset class imbalance and computes loss function weights
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("PHASE 2: CLASS DISTRIBUTION ANALYSIS & CLASS WEIGHTS COMPUTATION")
print("=" * 80)

# Configuration
DATASET_ROOT = Path("datasets")
OUTPUT_DIR = Path("preprocessing/outputs")
CLASSES = ["benign", "malignant", "normal"]

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2.1: CLASS DISTRIBUTION ANALYSIS
# ============================================================================
print("\n[2.1] CLASS DISTRIBUTION ANALYSIS")
print("-" * 80)

class_counts = {}
total_images = 0

# Count images per class (from raw dataset)
for class_name in CLASSES:
    class_dir = DATASET_ROOT / class_name
    if not class_dir.exists():
        print(f"ERROR: {class_dir} does not exist!")
        continue
    
    # Count image files (exclude masks)
    image_files = [f for f in class_dir.glob("*.png") if not ("_mask" in f.name)]
    count = len(image_files)
    class_counts[class_name] = count
    total_images += count
    print(f"{class_name:12} : {count:4d} images")

print(f"\nTotal images: {total_images}")

# ============================================================================
# COMPUTE CLASS WEIGHTS
# ============================================================================
print("\n[2.2] CLASS WEIGHTS COMPUTATION")
print("-" * 80)

# Formula: weight_c = total_images / (num_classes * count_c)
num_classes = len(CLASSES)
class_weights = {}

print(f"\nFormula: weight_c = {total_images} / (3 * count_c)\n")

for class_name in CLASSES:
    count = class_counts[class_name]
    weight = total_images / (num_classes * count)
    class_weights[class_name] = round(weight, 4)
    percentage = (count / total_images) * 100
    
    print(f"{class_name:12} : weight = {total_images} / ({num_classes} * {count}) = {weight:.4f}")
    print(f"              Percentage: {percentage:.1f}% of dataset")
    print()

# ============================================================================
# CREATE COMPREHENSIVE REPORT
# ============================================================================
print("\n[2.3] GENERATING COMPREHENSIVE REPORT")
print("-" * 80)

# Calculate imbalance ratio
benign_count = class_counts["benign"]
malignant_count = class_counts["malignant"]
normal_count = class_counts["normal"]

imbalance_ratio = f"{benign_count/normal_count:.2f}:{malignant_count/normal_count:.2f}:1.00"

class_distribution_report = {
    "class_distribution": {
        "benign": {
            "count": benign_count,
            "percentage": round((benign_count / total_images) * 100, 1),
            "weight": class_weights["benign"]
        },
        "malignant": {
            "count": malignant_count,
            "percentage": round((malignant_count / total_images) * 100, 1),
            "weight": class_weights["malignant"]
        },
        "normal": {
            "count": normal_count,
            "percentage": round((normal_count / total_images) * 100, 1),
            "weight": class_weights["normal"]
        }
    },
    "total_images": total_images,
    "num_classes": num_classes,
    "imbalance_ratio": imbalance_ratio,
    "class_weights_dict": {
        "benign": class_weights["benign"],
        "malignant": class_weights["malignant"],
        "normal": class_weights["normal"]
    },
    "class_weights_list": [
        class_weights["benign"],
        class_weights["malignant"],
        class_weights["normal"]
    ]
}

# Save JSON report
output_file = OUTPUT_DIR / "phase2_class_distribution.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(class_distribution_report, f, indent=2)

print(f"Saved: {output_file}")

# ============================================================================
# 2.2: LESION COVERAGE ANALYSIS (OPTIONAL)
# ============================================================================
print("\n[2.4] OPTIONAL: LESION COVERAGE ANALYSIS")
print("-" * 80)

from PIL import Image

coverage_stats = {}

for class_name in CLASSES:
    class_dir = DATASET_ROOT / class_name
    mask_files = [f for f in sorted(class_dir.glob("*.png")) if "_mask.png" in f.name and "_mask_" not in f.name]
    
    coverages = []
    
    for mask_file in mask_files:
        try:
            mask = Image.open(mask_file)
            mask_array = np.array(mask)
            
            # Normalize to 0-1 range
            if mask_array.max() > 1:
                mask_array = mask_array.astype(float) / 255.0
            
            # Calculate coverage percentage
            coverage = (mask_array > 0.5).sum() / mask_array.size * 100
            coverages.append(coverage)
        except:
            pass
    
    if coverages:
        coverage_stats[class_name] = {
            "mean_coverage_pct": round(np.mean(coverages), 2),
            "min_coverage_pct": round(np.min(coverages), 2),
            "max_coverage_pct": round(np.max(coverages), 2),
            "std_coverage_pct": round(np.std(coverages), 2),
            "total_masks_analyzed": len(coverages)
        }
        
        print(f"\n{class_name.capitalize()}:")
        print(f"  Mean lesion coverage: {coverage_stats[class_name]['mean_coverage_pct']:.2f}%")
        print(f"  Min coverage: {coverage_stats[class_name]['min_coverage_pct']:.2f}%")
        print(f"  Max coverage: {coverage_stats[class_name]['max_coverage_pct']:.2f}%")
        print(f"  Std deviation: {coverage_stats[class_name]['std_coverage_pct']:.2f}%")

# Add lesion coverage to report
class_distribution_report["lesion_coverage_analysis"] = coverage_stats

# Update JSON file with lesion coverage
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(class_distribution_report, f, indent=2)

print(f"\nUpdated: {output_file}")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2 SUMMARY")
print("=" * 80)

summary_text = f"""
CLASS DISTRIBUTION:
-------------------
Benign    : {benign_count:4d} images ({class_distribution_report['class_distribution']['benign']['percentage']:.1f}%) - Weight: {class_weights['benign']:.4f}
Malignant : {malignant_count:4d} images ({class_distribution_report['class_distribution']['malignant']['percentage']:.1f}%) - Weight: {class_weights['malignant']:.4f}
Normal    : {normal_count:4d} images ({class_distribution_report['class_distribution']['normal']['percentage']:.1f}%) - Weight: {class_weights['normal']:.4f}
-------------------
Total     : {total_images:4d} images

IMBALANCE RATIO: {imbalance_ratio}
- Benign is {benign_count/normal_count:.1f}x more frequent than Normal
- Malignant is {malignant_count/normal_count:.1f}x more frequent than Normal

CLASS WEIGHTS (For Loss Function):
-----------------------------------
Dict format:
  'benign': {class_weights['benign']:.4f},
  'malignant': {class_weights['malignant']:.4f},
  'normal': {class_weights['normal']:.4f}

List format (0=benign, 1=malignant, 2=normal):
  [{class_weights['benign']:.4f}, {class_weights['malignant']:.4f}, {class_weights['normal']:.4f}]

RECOMMENDATIONS:
----------------
1. Use class weights in weighted loss function (CrossEntropyLoss, etc.)
2. Consider weighted sampling during data augmentation
3. Malignant class is most underrepresented - may benefit from oversampling
4. Normal class is least represented - consider giving it highest weight

OUTPUT:
-------
File: {output_file}
Contains: Complete class distribution, weights, and lesion coverage analysis

PHASE 2 STATUS: COMPLETE
Next: Phase 3 - Preprocessing Infrastructure
"""

print(summary_text)

print("=" * 80)
