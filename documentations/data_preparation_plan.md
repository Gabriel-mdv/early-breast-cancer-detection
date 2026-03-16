# Data Preparation Plan - Early Breast Cancer Detection
## MobileViTv3 + BUSI Dataset

**Status**: Phases 1-7 COMPLETE - Ready for model training  
**Target**: Split dataset ready for training model  
**Timeline**: 7 days completed (Phases 1-7)
**Architecture**: 3-channel tensors (NO FCM yet)

---

## QUICK REFERENCE: FINAL FOLDER STRUCTURE (CORRECTED!)

### Phase Execution & Code Location
```
eda/
└── Phase 1 Assets (Notebook + Analysis)
    ├── phase1_eda_analysis.ipynb  ← Main EDA notebook
    └── outputs/
        ├── file_count_report.txt
        ├── dimension_analysis.json
        ├── intensity_stats.json
        ├── intensity_histograms.png
        ├── artifact_report.txt
        ├── mask_validation_report.txt
        └── sample_images_grid.png

preprocessing/  ← ALL PREPROCESSING PHASES EXECUTED HERE
├── Core Modules (existing):
│   ├── __init__.py
│   ├── dataloader.py
│   ├── denoise.py
│   ├── contrast.py
│   ├── normalization.py
│   ├── resize.py
│   └── documentation.md
├── New Core Modules (Phase 3):
│   ├── format_standardizer.py
│   ├── augmentation.py
│   ├── preprocessor.py (orchestrator)
│   ├── sampler.py (Phase 7)
│   └── utils.py
├── Phase Scripts (Execution):
│   ├── phase2_class_weights.py ← Phase 2: Compute class weights
│   ├── phase3_setup_pipeline.py ← Phase 3: Setup modules
│   ├── phase4_execute_preprocessing.py ← Phase 4: Run pipeline
│   ├── phase5_dataset_splitting.py ← Phase 5: Create splits
│   └── phase6_validate_dataset.py ← Phase 6: Validation
└── outputs/ ← All preprocessing outputs (logs, stats, reports)
    ├── phase2_class_distribution.json
    ├── phase4_preprocessing_log.json
    ├── phase5_split_statistics.json
    └── phase6_validation_report.json

datasets/
├── benign/, malignant/, normal/  ← RAW DATA (INPUT)
└── processed/  ← PROCESSED DATA OUTPUT (created by preprocessing/)
    ├── images/[benign|malignant|normal]/ ← Preprocessed images
    ├── masks/[benign|malignant|normal]/  ← Resized masks
    └── manifests/
        ├── train_manifest.txt
        ├── val_manifest.txt
        └── test_manifest.txt

results/ ← FINAL REPORTS & ANALYSIS (OUTPUT)
├── split_statistics.json
├── sample_batches_visualization.png
├── final_data_report.json
└── preprocessing_metrics.json
```

**KEY RULE**: 
- CODE & EXECUTION: `preprocessing/` folder
- RAW INPUT DATA: `datasets/[benign|malignant|normal]/`
- PROCESSED DATA: `datasets/processed/`
- FINAL REPORTS: `results/` and `preprocessing/outputs/`

---

# PHASE 1: EDA & DATASET VERIFICATION
**Duration**: 1 day  
**Deliverables**: Understanding of raw data + artifact report

## 1.1 - File Count Verification
**Purpose**: Ensure dataset is complete and uncorrupted

**Task**: Count all files in each directory
```
Expected:
- benign/: 436 images
- malignant/: 210 images
- normal/: 134 images
Total images: 780
```

**Output Location**: `eda/file_count_report.txt`
**Note**: Results saved to existing `eda/` folder at root
```
[Class] [Image Count] [Mask Count] [Total] [Match Expected?]
benign: 436 images, 436 masks = 872 files ✓
malignant: 210 images, 210 masks = 420 files ✓
normal: 134 images, 134 masks = 268 files ✓
TOTAL: 780 images, 780 masks = 1560 files ✓
```

---

## 1.2 - Image Dimension Analysis
**Purpose**: Know actual image sizes, plan for resizing

**Task**: Load every image, record width and height
- Find: min, max, mean, median dimensions
- Check: RGB vs greyscale (single channel vs 3 channels)
- Verify: Pixel data type (uint8, float32, etc.)

**Output Location**: `eda/dimension_analysis.json`
**Note**: Results saved to existing `eda/` folder at root
```json
{
  "dimension_stats": {
    "width": {"min": 480, "max": 500, "mean": 498.5, "median": 500},
    "height": {"min": 480, "max": 500, "mean": 498.5, "median": 500},
    "total_unique_sizes": 15
  },
  "channel_analysis": {
    "single_channel_images": 450,
    "three_channel_images": 330,
    "other": 0
  },
  "data_type_check": {
    "uint8": 750,
    "float32": 30,
    "other": 0
  }
}
```

---

## 1.3 - Pixel Intensity Distribution Analysis
**Purpose**: Understand image brightness/contrast characteristics

**Task**: For each class, compute:
- Mean pixel intensity
- Standard deviation
- Min/Max pixel values
- Plot histograms for sample images

**Output Files Location**: 
- `eda/intensity_stats.json` - Numerical statistics
- `eda/intensity_histograms.png` - Visualization (3 subplots, one per class)
**Note**: All outputs in existing `eda/` folder at root

**Expected output format**:
```json
{
  "benign": {
    "mean": 95.3,
    "std": 42.1,
    "min": 12,
    "max": 245,
    "percentile_25": 65,
    "percentile_75": 128
  },
  "malignant": {
    "mean": 98.7,
    "std": 44.2,
    "min": 15,
    "max": 248,
    "percentile_25": 68,
    "percentile_75": 132
  },
  "normal": {
    "mean": 92.1,
    "std": 38.9,
    "min": 8,
    "max": 235,
    "percentile_25": 62,
    "percentile_75": 120
  }
}
```

---

## 1.4 - Artifact & Quality Inspection
**Purpose**: Identify problematic images, understand data quality

**Task**: Manually open and inspect 20-30 images per class
- Look for: Text overlays, watermarks, depth markers
- Look for: Black borders, padding regions
- Look for: Multiple lesion regions in one image
- Note: Any severely corrupted or unusual images

**Output Location**: `eda/artifact_report.txt`
**Note**: Results saved to existing `eda/` folder at root
```
ARTIFACT FINDINGS:
================

Text Overlays/Watermarks:
- Found in ~45% of images
- Location: Top-right, bottom corners
- Severity: Low (affects <10% of image area)

Black Border Padding:
- Found in ~60% of images
- Thickness: 5-20 pixels per side
- Recommendation: Crop 5-10% from border

Multiple Lesions:
- Found in ~8% of benign images
- Found in ~12% of malignant images
- Note: Ground truth masks show single lesion region (verified)

Severely Problematic Images: NONE

RECOMMENDATION: Implement 5% border crop in preprocessing
```

---

## 1.5 - Ground Truth Mask Inspection
**Purpose**: Verify mask quality and correctness

**Task**: Load 10-15 mask images per class
- Verify all pixels are binary (0 or 255)
- Overlay on original images
- Check anatomical correctness
- For normal images: verify masks are completely black (empty)

**Output Location**: `eda/mask_validation_report.txt`
**Note**: Results saved to existing `eda/` folder at root
```
MASK VALIDATION:
================

Binary Check:
- All masks properly binary: YES ✓
- No intermediate grey values: YES ✓

Normal Class Masks:
- Total normal images inspected: 15
- Masks with non-zero pixels (should be 0): 0 ✓
- All correctly labeled as background: YES ✓

Benign/Malignant Masks:
- Mask coverage range: 5% - 35% of image
- Boundaries anatomically reasonable: YES ✓
- Alignment with original image: YES ✓

Conclusion: Masks are valid and properly labeled
```

---

## 1.6 - Create Sample Visualization
**Purpose**: Visual confirmation of data quality

**Output Location**: `eda/sample_images_grid.png`
**Note**: Results saved to existing `eda/` folder at root
```
Grid with 12 samples (4 per class):
- Left column: Original image
- Right column: Image + mask overlay (red)
```

---

# PHASE 2: ANALYSIS & CLASS WEIGHTS COMPUTATION
**Duration**: 1 day  
**Deliverables**: Class weights + detailed statistics

## 2.1 - Class Distribution Analysis
**Purpose**: Understand imbalance, compute loss function weights

**Task**: Count images per class, compute weights

**Calculation**:
```
Total images = 780
benign: 436 images (55.9%)
malignant: 210 images (26.9%)
normal: 134 images (17.2%)

Class weights formula:
weight_c = total_images / (num_classes * count_c)

weight_benign = 780 / (3 * 436) = 0.597
weight_malignant = 780 / (3 * 210) = 1.238
weight_normal = 780 / (3 * 134) = 1.940
```

**Output Location**: `eda/class_distribution.json`
**Note**: Results saved to existing `eda/` folder at root
```json
{
  "class_distribution": {
    "benign": {
      "count": 436,
      "percentage": 55.9,
      "weight": 0.597
    },
    "malignant": {
      "count": 210,
      "percentage": 26.9,
      "weight": 1.238
    },
    "normal": {
      "count": 134,
      "percentage": 17.2,
      "weight": 1.940
    }
  },
  "total_images": 780,
  "imbalance_ratio": "3.26:1.57:1"
}
```

---

## 2.2 - Lesion Coverage Analysis (Optional but useful)
**Purpose**: Understand lesion size distribution across classes

**Task**: Calculate % of image covered by mask for each class

**Output**: Statistics showing lesion size distribution

---

# PHASE 3: PREPROCESSING INFRASTRUCTURE
**Duration**: 2 days  
**Deliverables**: Python scripts for each preprocessing step

## 3.1 - Create Preprocessing Module Structure
```
preprocessing/
├── __init__.py
├── dataloader.py          → Load and iterate images
├── format_standardizer.py → Convert RGB→greyscale, handle formats
├── denoise.py             → Anisotropic diffusion
├── contrast.py            → CLAHE enhancement
├── normalization.py       → Normalize to ImageNet specs
├── resize.py              → Bilinear resize to 224×224
├── augmentation.py        → Training data augmentation
├── preprocessor.py        → Main pipeline orchestrator
└── utils.py               → Helper functions
```

---

## 3.2 - Implement Each Preprocessing Component

### Component A: Format Standardizer
**Input**: Raw PNG image file  
**Output**: Single-channel uint8 greyscale (H, W)
```python
# If 3-channel and identical across channels → keep 1 channel
# If 1-channel greyscale → keep as is
# After: all images are (H, W), dtype=uint8, values 0-255
```

---

### Component B: Denoise (Anisotropic Diffusion)
**Parameters**:
- Iterations: 10
- Conductance: 30
- Time step: 0.1
**Input**: Greyscale uint8 (H, W)
**Output**: Denoised float32 (H, W), rescaled to 0-255 uint8

---

### Component C: CLAHE (Contrast Enhancement)
**Parameters**:
- Clip limit: 2.0
- Tile grid: 8×8
**Input**: Denoised uint8 (H, W)
**Output**: Enhanced contrast uint8 (H, W)

---

### Component D: Normalization
**Parameters**:
- ImageNet mean: 0.485
- ImageNet std: 0.229
**Input**: uint8 (H, W) [0-255]
**Output**: float32 (H, W) [-2.2 to 2.2] (ImageNet normalized)

---

### Component E: Resize
**Parameters**:
- Target size: 224×224
- Interpolation: bilinear
**Input**: float32 (H, W)
**Output**: float32 (224, 224)

---

### Component F: Tensor Construction
**Input**: Normalized image float32 (224, 224)
**Output**: 3-channel tensor float32 (224, 224, 3) [replicate greyscale]

---

## 3.3 - Implement Data Pipeline Classes
**Create classes for**:
- `ImageProcessor`: Single image transformation
- `BatchProcessor`: Process batches of images
- `DatasetPreprocessor`: Orchestrate full pipeline

---

# PHASE 4: PREPROCESSING EXECUTION
**Duration**: 2-3 days  
**Deliverables**: Preprocessed images stored on disk

## 4.1 - Create Processed Data Directory Structure
```
datasets/processed/
├── images/            # Preprocessed images (224×224)
│   ├── benign/
│   ├── malignant/
│   └── normal/
├── masks/             # Resized masks (224×224, binary)
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── metadata/
    └── preprocessing_log.json
```

---

## 4.2 - Process All Images
**Task**: Run preprocessing pipeline on all 780 images
- Input: `datasets/raw/Dataset_BUSI_with_GT/[class]/`
- Output: `datasets/processed/images/[class]/` + `datasets/processed/masks/[class]/`

**Logging**:
- Save processing time per image
- Track any errors or warnings
- Generate summary statistics

**Verification**:
- Count output files match input count
- Spot check image shapes: all (224, 224, 3)?
- Spot check value ranges: [-2.2, 2.2]?

**Output**: `results/preprocessing/preprocessing_log.json`
```json
{
  "total_images_processed": 780,
  "total_images_successful": 780,
  "total_images_failed": 0,
  "processing_time_minutes": 45,
  "average_time_per_image_ms": 3500,
  "output_directory": "datasets/processed/images/",
  "image_shapes_verified": true,
  "value_ranges_verified": true
}
```

---

# PHASE 5: DATASET SPLITTING
**Duration**: 1 day  
**Deliverables**: Train/Val/Test splits with manifest files

## 5.1 - Stratified Split (70/15/15)
**Requirements**:
- Stratified: Each split has same class proportions as original
- 70% training: ~546 images (~244 benign, ~147 malignant, ~94 normal)
- 15% validation: ~117 images (~61 benign, ~31 malignant, ~20 normal)
- 15% test: ~117 images (~61 benign, ~32 malignant, ~20 normal)

**Algorithm**:
1. Create list of all image filenames with labels
2. Use sklearn.model_selection.StratifiedShuffleSplit
3. Assign each image to one split (no overlap)

---

## 5.2 - Create Split Manifest Files
**Output files**:
- `datasets/processed/manifests/train_manifest.txt`
- `datasets/processed/manifests/val_manifest.txt`
- `datasets/processed/manifests/test_manifest.txt`

**File format** (one line per image):
```
benign_001.png benign
benign_002.png benign
malignant_045.png malignant
normal_012.png normal
...
```

---

## 5.3 - Create Split Statistics
**Output**: `results/dataset_splits/split_statistics.json`
```json
{
  "train": {
    "total": 546,
    "benign": 306,
    "malignant": 147,
    "normal": 93,
    "distribution": "benign:55.7%, malignant:26.9%, normal:17.0%"
  },
  "validation": {
    "total": 117,
    "benign": 65,
    "malignant": 32,
    "normal": 20,
    "distribution": "benign:55.6%, malignant:27.4%, normal:17.1%"
  },
  "test": {
    "total": 117,
    "benign": 65,
    "malignant": 31,
    "normal": 21,
    "distribution": "benign:55.6%, malignant:26.5%, normal:17.9%"
  },
  "stratification_verified": true
}
```

---

# PHASE 6: VALIDATION & TESTING
**Duration**: 1 day  
**Deliverables**: Verified, ready-to-train dataset

## 6.1 - Verify Split Integrity
**Checks**:
- [ ] All 780 images assigned to exactly one split
- [ ] No image appears twice
- [ ] Class proportions maintained across splits
- [ ] Manifest files load without errors
- [ ] All referenced files exist on disk

---

## 6.2 - Load and Visualize Test Batches
**Task**: Create data loaders, load sample batches
- Load batch of 16 images from each split
- Visualize: 4 images from each class per split
- Verify: Tensor shapes, value ranges, image quality

**Output**: `results/dataset_splits/sample_batches_visualization.png`
```
Grid showing:
- Train batch (12 images, 4 per class)
- Val batch (12 images, 4 per class)
- Test batch (12 images, 4 per class)
```

---

## 6.3 - Create PyTorch/TF Data Loaders
**Output files**:
- `preprocessing/data_loaders.py` - Reusable DataLoader classes
- Tests: Verify loaders work with augmentation pipeline

---

## 6.4 - Final Statistics Report
**Output**: `results/final_data_report.json`
```json
{
  "dataset_ready": true,
  "total_images": 780,
  "image_shape": [224, 224, 3],
  "value_range": [-2.2, 2.2],
  "train_split": 546,
  "val_split": 117,
  "test_split": 117,
  "class_weights": {
    "benign": 0.597,
    "malignant": 1.238,
    "normal": 1.940
  },
  "augmentation_enabled": true,
  "augmentation_types": [
    "horizontal_flip (p=0.5)",
    "rotation±15° (p=0.5)",
    "brightness_contrast (p=0.3)",
    "translation (p=0.3)"
  ],
  "preprocessing_log": "results/preprocessing/preprocessing_log.json",
  "ready_for_training": true
}
```

---

# SUMMARY OF OUTPUTS & FOLDER STRUCTURE

## Existing Folders (Already at root)
```
eda/                                    ← EXISTING FOLDER
├── eda.ipynb                          (EDA notebook)
└── (Phase 1 & 2 outputs will be saved here)

preprocessing/                          ← EXISTING FOLDER
├── __init__.py
├── dataloader.py
├── contrast.py
├── denoise.py
├── documentation.md
├── normalization.py
├── resize.py
└── (Phase 3 new modules will be added here)
```

## Folders to CREATE
```
datasets/
├── raw/                               ← (Already exists with BUSI data)
│   └── Dataset_BUSI_with_GT/
│       ├── benign/
│       ├── malignant/
│       └── normal/
│
└── processed/                         ← CREATE (Phase 4)
    ├── images/                        (Preprocessed images)
    │   ├── benign/
    │   ├── malignant/
    │   └── normal/
    ├── masks/                         (Resized masks)
    │   ├── benign/
    │   ├── malignant/
    │   └── normal/
    └── manifests/                     (CREATE - Phase 5)
        ├── train_manifest.txt
        ├── val_manifest.txt
        └── test_manifest.txt

results/                                ← CREATE (if doesn't exist)
├── preprocessing/                     (Phase 4 logs)
│   └── preprocessing_log.json
└── dataset_splits/                    (Phase 5 & 6)
    ├── split_statistics.json
    └── sample_batches_visualization.png
```

## Phase-by-Phase Output Locations (CORRECTED!)

**PHASE 1 Outputs** → `eda/phase1_eda_analysis.ipynb` + `eda/outputs/`
- Notebook: `eda/phase1_eda_analysis.ipynb`
- Data Files:
  - `eda/outputs/file_count_report.txt`
  - `eda/outputs/dimension_analysis.json`
  - `eda/outputs/intensity_stats.json`
  - `eda/outputs/intensity_histograms.png`
  - `eda/outputs/artifact_report.txt`
  - `eda/outputs/mask_validation_report.txt`
  - `eda/outputs/sample_images_grid.png`

**PHASE 2 Code & Outputs** → `preprocessing/phase2_class_weights.py` + `preprocessing/outputs/`
- Script: `preprocessing/phase2_class_weights.py`
- Output:
  - `preprocessing/outputs/phase2_class_distribution.json`

**PHASE 3 Code** → `preprocessing/` (Core modules)
- New modules: `preprocessing/format_standardizer.py`
- New modules: `preprocessing/augmentation.py`
- New modules: `preprocessing/preprocessor.py`
- New modules: `preprocessing/utils.py`
- Implementations: Update existing empty modules

**PHASE 4 Code & Outputs** → `preprocessing/phase4_execute_preprocessing.py` + `datasets/processed/` + `preprocessing/outputs/`
- Script: `preprocessing/phase4_execute_preprocessing.py`
- Processed Images: `datasets/processed/images/[benign|malignant|normal]/*.png` (780 files)
- Processed Masks: `datasets/processed/masks/[benign|malignant|normal]/*.png` (780 files)
- Log: `preprocessing/outputs/phase4_preprocessing_log.json`

**PHASE 5 Code & Outputs** → `preprocessing/phase5_dataset_splitting.py` + `datasets/processed/manifests/` + `preprocessing/outputs/`
- Script: `preprocessing/phase5_dataset_splitting.py`
- Manifests:
  - `datasets/processed/manifests/train_manifest.txt`
  - `datasets/processed/manifests/val_manifest.txt`
  - `datasets/processed/manifests/test_manifest.txt`
- Stats: `preprocessing/outputs/phase5_split_statistics.json`

**PHASE 6 Code & Outputs** → `preprocessing/phase6_validate_dataset.py` + `results/`
- Script: `preprocessing/phase6_validate_dataset.py`
- Outputs:
  - `preprocessing/outputs/phase6_validation_report.json`
  - `results/final_data_report.json` (comprehensive summary)
  - `results/sample_batches_visualization.png` (sample grid)
```

---

# PHASE 7: CLASS IMBALANCE HANDLING & TRAINING PREPARATION
**Duration**: 1-2 days  
**Deliverables**: Sampler + Augmentation strategy for training

## Problem Statement
**Dataset Imbalance**:
- Benign: 437 images (56.0%) - MAJORITY class
- Malignant: 210 images (26.9%) - MINORITY class  
- Normal: 133 images (17.1%) - MINORITY class
- **Imbalance Ratio**: 3.29:1.58:1.00

**Clinical Impact**: 
Missing a malignant tumor (false negative) is far more costly than misclassifying a benign mass. The loss function must reflect this clinical reality.

---

## 7.1 - METHOD 1: CLASS WEIGHTS IN LOSS FUNCTION (REQUIRED)
**Why**: Directly tells the model that misclassifying minority classes is more expensive

**Implementation**:
```python
# Already computed in Phase 2:
class_weights = {
    'benign': 0.595,
    'malignant': 1.238,
    'normal': 1.955
}

# Use in training:
import torch
import torch.nn as nn
from preprocessing import BUSIDataLoader

loader = BUSIDataLoader(processed_dir='datasets/processed')
train_dataset = loader.create_dataset('train', 'datasets/processed/manifests/train_manifest.txt')

# Get class weights as tensor
weights_tensor = torch.tensor([0.595, 1.238, 1.955])

# Apply to loss function
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# In training loop:
for batch in train_loader:
    images = batch['image']
    labels = batch['label']
    
    outputs = model(images)
    loss = criterion(outputs, labels)  # Weighted loss
    loss.backward()
```

**Benefit**: Simple, non-negotiable for medical imaging. Aligns with clinical cost of errors.

---

## 7.2 - METHOD 2: AUGMENTATION-BASED OVERSAMPLING (RECOMMENDED)
**Why**: Increases diversity of minority class samples without pure duplication. Standard practice for small medical datasets.

**Strategy**:
1. Use `ClassAwareSampler` to oversample minority classes during batching
2. Apply heavier augmentation to minority classes (malignant & normal)
3. Apply lighter augmentation to majority class (benign)

### File: `preprocessing/sampler.py` (NEW - Phase 7.1)
**Code to create**:
```python
"""Class-aware sampler for augmentation-based oversampling"""
import numpy as np
from torch.utils.data import Sampler
from typing import Iterator

class ClassAwareSampler(Sampler):
    """
    Sampler that applies class-aware sampling probabilities
    Minority classes (malignant, normal) sampled more frequently
    Each epoch sees full dataset but distributed by class frequency inverse
    """
    
    def __init__(self, dataset, num_samples=None):
        """
        Args:
            dataset: BUSIDataset instance
            num_samples: Total samples per epoch (None = len(dataset))
        """
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        
        # Calculate inverse class frequencies (minority = higher probability)
        unique_labels = np.unique(dataset.labels)
        class_counts = np.bincount(dataset.labels)
        
        # Inverse weighting: higher count → lower probability
        weights = 1.0 / class_counts
        weights = weights / weights.sum()  # Normalize
        
        # Assign weight to each sample based on its class
        self.weights = np.array([weights[label] for label in dataset.labels])
        self.weights = self.weights / self.weights.sum()
    
    def __iter__(self) -> Iterator:
        """Yields indices sampled according to class weights"""
        indices = np.random.choice(
            len(self.dataset),
            size=self.num_samples,
            replace=True,  # Allow repeats for upsampling
            p=self.weights
        )
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.num_samples
```

**Output Location**: `preprocessing/sampler.py`

---

### Update: `preprocessing/dataloader.py` (Phase 7.2)
**Modify `__getitem__` to include class-specific augmentation**:

```python
def __getitem__(self, idx: int) -> Dict:
    image_path = self.image_paths[idx]
    label = self.labels[idx]
    
    # Load image
    image = np.load(image_path)
    
    # Add augmentation flag for class-aware augmentation
    if self._augmentation_enabled:
        if label in [1, 2]:  # Malignant (1) or Normal (2) - MINORITY
            # Heavier augmentation for minority classes
            image = self._augment_aggressive(image)
        else:  # Benign (0) - MAJORITY
            # Lighter augmentation for majority class
            image = self._augment_light(image)
    
    # Convert to torch tensor
    image_tensor = torch.from_numpy(image).float()
    
    return {
        'image': image_tensor,
        'label': torch.tensor(label, dtype=torch.long)
    }

def _augment_aggressive(self, image: np.ndarray) -> np.ndarray:
    """Heavier augmentation for minority classes (malignant, normal)"""
    augmentation_config = {
        'horizontal_flip': {'probability': 0.7},
        'rotate': {'angle_range': (-20, 20), 'probability': 0.7},
        'brightness_contrast': {'factor_range': (0.7, 1.3), 'probability': 0.6},
        'translate': {'max_shift': 0.15, 'probability': 0.6}
    }
    return self.augmenter.augment(image, augmentation_config)

def _augment_light(self, image: np.ndarray) -> np.ndarray:
    """Lighter augmentation for majority class (benign)"""
    augmentation_config = {
        'horizontal_flip': {'probability': 0.3},
        'rotate': {'angle_range': (-10, 10), 'probability': 0.3},
        'brightness_contrast': {'factor_range': (0.85, 1.15), 'probability': 0.2},
        'translate': {'max_shift': 0.05, 'probability': 0.2}
    }
    return self.augmenter.augment(image, augmentation_config)
```

---

## 7.3 - TRAINING SETUP WITH BOTH METHODS
**Complete training data loader setup**:

```python
from preprocessing import BUSIDataLoader
from preprocessing.sampler import ClassAwareSampler
import torch
from torch.utils.data import DataLoader

# Initialize loader and dataset
loader = BUSIDataLoader(processed_dir='datasets/processed', batch_size=32)
train_dataset = loader.create_dataset('train', 'datasets/processed/manifests/train_manifest.txt')

# Enable augmentation in dataset
train_dataset._augmentation_enabled = True

# Method 1: Class weights for loss function
class_weights = torch.tensor([0.595, 1.238, 1.955])
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Method 2: Class-aware sampler for oversampling
sampler = ClassAwareSampler(train_dataset)

# Create data loader with sampler
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,         # ← Class-aware sampling
    num_workers=4,
    pin_memory=True
)

# Validation/Test loaders (NO augmentation, NO special sampling)
val_loader = loader.create_dataloader('val', 'datasets/processed/manifests/val_manifest.txt', shuffle=False)
test_loader = loader.create_dataloader('test', 'datasets/processed/manifests/test_manifest.txt', shuffle=False)

# Use in training loop:
for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch['image']
        labels = batch['label']
        
        outputs = model(images)
        loss = criterion(outputs, labels)  # Weighted loss + augmented minority classes
        
        loss.backward()
        optimizer.step()
```

---

## 7.4 - EXPECTED BENEFITS
**Without imbalance handling:**
- Model may ignore minority classes (especially 'normal')
- High overall accuracy but poor minority class recall
- Clinical failure: Miss malignant tumors (false negatives)

**With both methods:**
1. **Class weights**: Tell loss function minority misclassification is expensive
2. **Oversampling + augmentation**: Expose model to more minority class variations
3. **Result**: Balanced precision/recall across all classes, clinically appropriate

**Expected Performance**:
- Binign (majority): ~88-92% accuracy
- Malignant (minority): ~85-90% accuracy  ← Most important
- Normal (minority): ~80-87% accuracy     ← Second most important

---

## 7.5 - FILES TO CREATE/MODIFY

**CREATE**:
- `preprocessing/sampler.py` - ClassAwareSampler implementation
- `training/imbalance_config.json` - Configuration for augmentation strategies

**MODIFY**:
- `preprocessing/dataloader.py` - Add augmentation-aware methods
- `preprocessing/__init__.py` - Export ClassAwareSampler

**Reference** (Already exists):
- Class weights: `preprocessing/outputs/phase2_class_distribution.json`
- Augmentation methods: `preprocessing/augmentation.py`

---

# EXECUTION CHECKLIST

## Phase 1: EDA (Day 1) → Notebook in `eda/`, Outputs in `eda/outputs/`
- [x] 1.1: Count all files → `eda/outputs/file_count_report.txt` [DONE]
- [x] 1.2: Analyze dimensions → `eda/outputs/dimension_analysis.json` [DONE]
- [x] 1.3: Compute intensity distributions → `eda/outputs/intensity_stats.json` [DONE]
- [x] 1.4: Inspect artifacts (20-30 samples) → `eda/outputs/artifact_report.txt` [DONE]
- [x] 1.5: Validate masks (10-15 samples) → `eda/outputs/mask_validation_report.txt` [DONE]
- [x] 1.6: Create sample visualization → `eda/outputs/sample_images_grid.png` [DONE]

## Phase 2: Class Weights Analysis (Day 2) → Script in `preprocessing/`, Output in `preprocessing/outputs/`
- [x] 2.1: Compute class distribution + weights → `preprocessing/phase2_class_weights.py` [DONE]
- [x] 2.1: Output JSON → `preprocessing/outputs/phase2_class_distribution.json` [DONE]
- [x] 2.2: (Optional) Lesion coverage analysis [DONE]

## Phase 3: Preprocessing Infrastructure (Days 2-3) → All modules in `preprocessing/`
- [x] 3.1: Create/verify core module structure [DONE]
- [x] 3.2: Implement format_standardizer.py [DONE]
- [x] 3.2: Implement denoise.py (anisotropic diffusion) [DONE]
- [x] 3.2: Implement contrast.py (CLAHE - may update existing) [DONE]
- [x] 3.2: Implement normalization.py (ImageNet specs - may update existing) [DONE]
- [x] 3.2: Implement resize.py (224×224 - may update existing) [DONE]
- [x] 3.2: Implement augmentation.py [DONE]
- [x] 3.3: Create preprocessor.py (main orchestrator) [DONE]
- [x] 3.3: Create utils.py (helper functions) [DONE]

## Phase 4: Preprocessing Execution (Days 3-4) → Script in `preprocessing/`, Outputs in `datasets/processed/` & `preprocessing/outputs/`
- [x] 4.1: Create directory structure [DONE]
  - [x] `datasets/processed/images/[benign|malignant|normal]/` [DONE]
  - [x] `datasets/processed/masks/[benign|malignant|normal]/` [DONE]
  - [x] `preprocessing/outputs/` (for logs) [DONE]
- [x] 4.2: Run preprocessing script → `preprocessing/phase4_execute_preprocessing.py` [DONE - 780/780 success]
- [x] 4.2: Verify outputs → `preprocessing/outputs/phase4_preprocessing_log.json` [DONE]
- [x] 4.3: Verify all 780 images processed successfully [DONE]

## Phase 5: Dataset Splitting (Days 4-5) → Script in `preprocessing/`, Manifests in `datasets/processed/manifests/`
- [x] 5.1: Create splitting script → `preprocessing/phase5_dataset_splitting.py` [DONE]
- [x] 5.1: Perform stratified 70/15/15 split [DONE - 545/115/120]
- [x] 5.2: Create manifest files [DONE]
  - [x] `datasets/processed/manifests/train_manifest.txt` [DONE]
  - [x] `datasets/processed/manifests/val_manifest.txt` [DONE]
  - [x] `datasets/processed/manifests/test_manifest.txt` [DONE]
- [x] 5.3: Generate statistics → `preprocessing/outputs/phase5_split_statistics.json` [DONE]

## Phase 6: Final Validation & Reports (Days 5-6) → Script in `preprocessing/`, Reports in `results/` & `preprocessing/outputs/`
- [x] 6.1: Create validation script → `preprocessing/phase6_validate_dataset.py` [DONE]
- [x] 6.1: Verify split integrity (no leakage, all 780 assigned) [DONE - 100% verified]
- [x] 6.2: Load and visualize test batches → `results/sample_batches_visualization.png` [DONE]
- [x] 6.3: Finalize data loaders in `preprocessing/dataloader.py` (update if needed) [DONE]
- [x] 6.4: Generate final reports [DONE]:
  - [x] `preprocessing/outputs/phase6_validation_report.json` [DONE]
  - [x] `results/final_data_report.json` (comprehensive summary) [DONE]

## Phase 7: Class Imbalance Handling (Days 6-7) → Files in `preprocessing/` & config in `training/`
- [x] 7.1: Implement Method 1 - Class Weights (use Phase 2 weights in loss function) [DONE - config created]
- [x] 7.2: Create ClassAwareSampler → `preprocessing/sampler.py` [DONE - ClassAwareSampler + WeightedRandomSampler]
- [x] 7.2: Update DataLoader augmentation methods → `preprocessing/dataloader.py` [DONE - _augment_aggressive() + _augment_light()]
- [x] 7.3: Create training configuration template → `training/imbalance_config.json` [DONE - comprehensive config]
- [x] 7.4: Document metrics to track (precision, recall, F1 per class) [DONE - documented in config]
- [x] 7.5: Update `preprocessing/__init__.py` to export sampler [DONE - exported ClassAwareSampler]

---

**Total Timeline**: 7 days completed ✅  
**Ready for Training**: Yes, all phases complete - dataset fully prepared with class imbalance handling

---

# PHASE 7 COMPLETION SUMMARY

## Files Created (Phase 7)

### 1. `preprocessing/sampler.py`
- **ClassAwareSampler**: Inverse frequency weighting for minority class oversampling
- **WeightedRandomSampler**: Alternative sampler with custom weight specification
- Enables sampling with replacement to expose model to minority variations

### 2. `training/imbalance_config.json`
- Complete configuration template for class imbalance handling
- Method 1 (Class Weights): Pre-computed in Phase 2, to be used in CrossEntropyLoss
- Method 2 (Augmentation-Based Oversampling): Configuration for aggressive vs. light augmentation
- Training setup examples and expected outcomes documented

## Files Modified (Phase 7)

### 1. `preprocessing/dataloader.py`
- Added `DataAugmenter` import from augmentation module
- Added `_augmentation_enabled` flag to BUSIDataset (default=False, set to True for training)
- Added `self.augmenter = DataAugmenter()` attribute
- Modified `__getitem__()` to apply class-aware augmentation:
  - Minority classes (malignant=1, normal=2): `_augment_aggressive()` (70% flip, 70% rotate±20°, 60% brightness±30%, 60% translate±15%)
  - Majority class (benign=0): `_augment_light()` (30% flip, 30% rotate±10°, 20% brightness±15%, 20% translate±5%)
- Added `_augment_aggressive()` method with aggressive augmentation config
- Added `_augment_light()` method with light augmentation config

### 2. `preprocessing/__init__.py`
- Added import: `from .sampler import ClassAwareSampler, WeightedRandomSampler`
- Added exports: `ClassAwareSampler`, `WeightedRandomSampler` to `__all__`
- Both samplers now accessible via: `from preprocessing import ClassAwareSampler`

## How to Use Phase 7 in Training

```python
from preprocessing import BUSIDataLoader, ClassAwareSampler
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# Initialize loader
loader = BUSIDataLoader(processed_dir='datasets/processed', batch_size=32)

# Create training dataset with augmentation ENABLED
train_dataset = loader.create_dataset('train', 'datasets/processed/manifests/train_manifest.txt')
train_dataset._augmentation_enabled = True  # ← Enable class-aware augmentation

# Method 1: Class Weights (from Phase 2)
class_weights = torch.tensor([0.595, 1.238, 1.955])
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Method 2: Class-Aware Sampler (from Phase 7)
sampler = ClassAwareSampler(train_dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,           # ← Uses inverse frequency weighting
    num_workers=4,
    pin_memory=True
)

# Validation/Test loaders (no augmentation, no special sampling)
val_loader = loader.create_dataloader('val', 'datasets/processed/manifests/val_manifest.txt', shuffle=False)
test_loader = loader.create_dataloader('test', 'datasets/processed/manifests/test_manifest.txt', shuffle=False)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch['image']
        labels = batch['label']
        
        outputs = model(images)
        loss = criterion(outputs, labels)  # Weighted loss + augmented minority classes
        
        loss.backward()
        optimizer.step()
```

## Expected Benefits

**Without Imbalance Handling:**
- Model ignores minority classes
- High overall accuracy but poor minority recall
- Clinical failure: Miss malignant tumors

**With Both Methods (Phase 7):**
- Benign (majority): ~88-92% accuracy
- Malignant (minority): ~85-90% accuracy ← **CRITICAL**
- Normal (minority): ~80-87% accuracy
- Balanced F1 scores across all classes
- Clinically appropriate: Catches malignant cases

## Key Metrics to Track During Training

- Per-class precision, recall, F1 score
- Overall accuracy
- Class-weighted accuracy
- Confusion matrix showing false negatives (most critical for medical imaging)
- ROC-AUC per class

## Files Unchanged but Used in Phase 7

- `preprocessing/augmentation.py` - DataAugmenter methods used by class-aware strategies
- `preprocessing/outputs/phase2_class_distribution.json` - Class weights source
- `datasets/processed/images/` - Pre-processed images (unchanged)
- `datasets/processed/manifests/` - Train/val/test splits (unchanged)

---

**STATUS**: ✅ All data preparation complete. Dataset ready for training with clinically-appropriate class imbalance handling methods implemented.
