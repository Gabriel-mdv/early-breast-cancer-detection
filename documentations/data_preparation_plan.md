# Data Preparation Plan - Early Breast Cancer Detection
## MobileViTv3 + BUSI Dataset

**Status**: Phase 1 COMPLETE - Ready for Phase 2 onwards  
**Target**: Split dataset ready for training model  
**Timeline**: 4-6 days remaining (Phases 2-6)  
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

# EXECUTION CHECKLIST

## Phase 1: EDA (Day 1) → Notebook in `eda/`, Outputs in `eda/outputs/`
- [x] 1.1: Count all files → `eda/outputs/file_count_report.txt` [DONE]
- [x] 1.2: Analyze dimensions → `eda/outputs/dimension_analysis.json` [DONE]
- [x] 1.3: Compute intensity distributions → `eda/outputs/intensity_stats.json` [DONE]
- [x] 1.4: Inspect artifacts (20-30 samples) → `eda/outputs/artifact_report.txt` [DONE]
- [x] 1.5: Validate masks (10-15 samples) → `eda/outputs/mask_validation_report.txt` [DONE]
- [x] 1.6: Create sample visualization → `eda/outputs/sample_images_grid.png` [DONE]

## Phase 2: Class Weights Analysis (Day 2) → Script in `preprocessing/`, Output in `preprocessing/outputs/`
- [ ] 2.1: Compute class distribution + weights → `preprocessing/phase2_class_weights.py`
- [ ] 2.1: Output JSON → `preprocessing/outputs/phase2_class_distribution.json`
- [ ] 2.2: (Optional) Lesion coverage analysis

## Phase 3: Preprocessing Infrastructure (Days 2-3) → All modules in `preprocessing/`
- [ ] 3.1: Create/verify core module structure
- [ ] 3.2: Implement format_standardizer.py
- [ ] 3.2: Implement denoise.py (anisotropic diffusion)
- [ ] 3.2: Implement contrast.py (CLAHE - may update existing)
- [ ] 3.2: Implement normalization.py (ImageNet specs - may update existing)
- [ ] 3.2: Implement resize.py (224×224 - may update existing)
- [ ] 3.2: Implement augmentation.py
- [ ] 3.3: Create preprocessor.py (main orchestrator)
- [ ] 3.3: Create utils.py (helper functions)

## Phase 4: Preprocessing Execution (Days 3-4) → Script in `preprocessing/`, Outputs in `datasets/processed/` & `preprocessing/outputs/`
- [ ] 4.1: Create directory structure
  - [ ] `datasets/processed/images/[benign|malignant|normal]/`
  - [ ] `datasets/processed/masks/[benign|malignant|normal]/`
  - [ ] `preprocessing/outputs/` (for logs)
- [ ] 4.2: Run preprocessing script → `preprocessing/phase4_execute_preprocessing.py`
- [ ] 4.2: Verify outputs → `preprocessing/outputs/phase4_preprocessing_log.json`
- [ ] 4.3: Verify all 780 images processed successfully

## Phase 5: Dataset Splitting (Days 4-5) → Script in `preprocessing/`, Manifests in `datasets/processed/manifests/`
- [ ] 5.1: Create splitting script → `preprocessing/phase5_dataset_splitting.py`
- [ ] 5.1: Perform stratified 70/15/15 split
- [ ] 5.2: Create manifest files
  - [ ] `datasets/processed/manifests/train_manifest.txt`
  - [ ] `datasets/processed/manifests/val_manifest.txt`
  - [ ] `datasets/processed/manifests/test_manifest.txt`
- [ ] 5.3: Generate statistics → `preprocessing/outputs/phase5_split_statistics.json`

## Phase 6: Final Validation & Reports (Days 5-6) → Script in `preprocessing/`, Reports in `results/` & `preprocessing/outputs/`
- [ ] 6.1: Create validation script → `preprocessing/phase6_validate_dataset.py`
- [ ] 6.1: Verify split integrity (no leakage, all 780 assigned)
- [ ] 6.2: Load and visualize test batches → `results/sample_batches_visualization.png`
- [ ] 6.3: Finalize data loaders in `preprocessing/dataloader.py` (update if needed)
- [ ] 6.4: Generate final reports:
  - [ ] `preprocessing/outputs/phase6_validation_report.json`
  - [ ] `results/final_data_report.json` (comprehensive summary)

---

**Total Timeline**: 5-7 days  
**Ready for Training**: Yes, after Phase 6 complete
