# Breast Ultrasound Images Dataset (BUSI) - Documentation

## Dataset Overview

**Dataset Name**: Breast Ultrasound Images Dataset (BUSI)  
**Source**: Kaggle  
**Citation**: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863  
**License**: CC0 - Public Domain  
**Use Case**: Classification, Detection & Segmentation

---

## Dataset Characteristics

### Image Specifications
- **Total Images**: 780 ultrasound images
- **Image Format**: PNG
- **Image Size**: 500 × 500 pixels (average)
- **Modality**: B-mode Ultrasound
- **Quality**: Clinical-grade medical images

### Patient Demographics
- **Total Patients**: 600 female patients
- **Age Range**: 25-75 years old
- **Collection Date**: 2018
- **Collection Frequency**: Baseline only (single snapshot per patient)

### Class Distribution (Images)
| Class | Count | Percentage |
|-------|-------|-----------|
| Benign | 436 | 55.9% |
| Malignant | 210 | 26.9% |
| Normal | 134 | 17.2% |
| **TOTAL** | **780** | **100%** |

### File Structure
```
Dataset_BUSI_with_GT/
├── benign/           (436 image files + 436 GT masks = 872 files)
├── malignant/        (210 image files + 210 GT masks = 420 files)
└── normal/           (134 image files + 134 GT masks = 268 files)
```

**Note**: Each image has a corresponding Ground Truth (GT) segmentation mask. GT images show the lesion boundaries for benign/malignant cases.

---

## Data Characteristics

### Key Features
- **Real Clinical Data**: Collected from actual patient scans
- **Balanced Dataset**: Reasonable class distribution (not heavily imbalanced)
- **Segmentation Masks**: GT images provided for localization/detection tasks
- **Standardized Size**: Consistent 500×500 pixel format
- **No Preprocessing Required**: Images are already cleaned and ready to use

### Dataset Quality
- **Usability Score**: 10.0/10 (Kaggle)
- **Research Verified**: Published in peer-reviewed journal (Data in Brief, 2020)
- **Reliable Labels**: Clinical annotations from medical professionals

---

## Task Applicability

### ✅ Classification (Primary Task)
- 3-class classification: Normal vs. Benign vs. Malignant
- Sufficient data size for deep learning
- Good class distribution for supervised learning

### ✅ Detection (Secondary Task)
- Ground truth masks enable detection/localization studies
- Identify lesion location and boundaries
- Can be repurposed as object detection dataset

### ✅ Segmentation (Optional)
- GT masks available for semantic segmentation
- Lesion boundary delineation
- Could improve classification confidence

---

## Recommendations for Project Implementation

### Dataset Split Strategy
**Suggested Train/Validation/Test Split**: 70% / 15% / 15%
- Train: ~546 images (306 benign, 147 malignant, 93 normal)
- Validation: ~117 images (65 benign, 32 malignant, 20 normal)
- Test: ~117 images (65 benign, 31 malignant, 21 normal)

### Preprocessing Pipeline
1. **Normalization**: Scale pixel values to [0, 1] or [-1, 1]
2. **Resizing**: Standardize to MobileViT input size (suggested: 256×256 or 224×224)
3. **Data Augmentation**:
   - Random rotation (±15 degrees)
   - Random horizontal flip
   - Slight brightness/contrast adjustment
   - Elastic deformations (medical imaging best practice)

### Class Imbalance Handling
- **Minor imbalance present** (Normal: 17%, Benign: 56%, Malignant: 27%)
- Recommendations:
  - Use `class_weights` in loss function
  - Apply stratified sampling during train/val/test split
  - Consider weighted sampling for data augmentation

### Expected Performance
- **Dataset Size**: 780 images - Medium dataset
- **Model Complexity**: MobileViT is well-suited
- **Expected Accuracy**: 85-95% with proper training and augmentation

---

## Directory Organization

```
datasets/
├── documentation.md          (this file)
├── raw/
│   └── Dataset_BUSI_with_GT/
│       ├── benign/
│       ├── malignant/
│       └── normal/
├── processed/
│   ├── train/
│   │   ├── benign/
│   │   ├── malignant/
│   │   └── normal/
│   ├── val/
│   │   ├── benign/
│   │   ├── malignant/
│   │   └── normal/
│   └── test/
│       ├── benign/
│       ├── malignant/
│       └── normal/
├── metadata/
│   ├── class_distribution.json
│   ├── train_split.txt
│   ├── val_split.txt
│   └── test_split.txt
└── augmented/               (optional: for augmented training data)
    ├── train/
    │   ├── benign/
    │   ├── malignant/
    │   └── normal/
    └── metadata/
```

---

## Data Preparation Checklist

- [ ] Download and extract dataset to `datasets/raw/`
- [ ] Verify file counts match documentation
- [ ] Create `datasets/processed/` directory structure
- [ ] Implement train/val/test splitting logic
- [ ] Create metadata files tracking splits
- [ ] Implement data loading pipeline
- [ ] Set up preprocessing functions
- [ ] Test augmentation pipeline
- [ ] Verify image loading and dimensions
- [ ] Document any anomalies or missing images

---

## Important Notes

1. **GT Mask Handling**: Ground truth masks can be used for:
   - Validation of detection models
   - Auxiliary supervision for segmentation
   - Should be excluded from main classification input (use only RGB images for training)

2. **Image Boundaries**: Some images may have dark borders (ultrasound artifacts). Consider:
   - Cropping to ROI
   - Masking borders during preprocessing
   - Testing both approaches

3. **Data Leakage Prevention**: Ensure strict separation of train/val/test sets
   - Use stratified splitting by patient if possible
   - Some patients may appear in multiple images

---

## References & Further Reading

- **Original Paper**: Al-Dhabyani et al., 2020 - Data in Brief
- **Kaggle Dataset**: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- **Related Work**: Breast cancer detection using deep learning
- **Ultrasound Imaging**: B-mode ultrasound basics and artifacts

---

**Last Updated**: March 15, 2026  
**Dataset Status**: Ready for preprocessing and training pipeline development
