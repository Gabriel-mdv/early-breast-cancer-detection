---
# MobileFCMViTv3: In-Depth Architecture & System Overview

## 1. Model Architecture
MobileFCMViTv3 is a modular deep learning pipeline for breast cancer detection from ultrasound images. It combines classical fuzzy clustering (FCM) with a modern MobileViTv3 backbone, enabling robust feature fusion and mobile deployment.

### Main Modules & Code Structure
- **benchmark/**: Scripts for latency, memory, and model size benchmarking.
- **config/**: YAML files for dataset, model, and training configuration.
- **datasets/**: Raw and processed data, including benign, malignant, normal, and masks.
- **documentations/**: Markdown files for architecture, experiment setup, training, evaluation, and data preparation.
- **eda/**: Exploratory Data Analysis scripts and reports.
- **evaluation/**: Metrics, confusion matrix, ROC, Grad-CAM, and evaluation pipeline.
- **experiments/**: Ablation studies, baseline models, experiment tracking.
- **export/**: Model export scripts (ONNX, TFLite, quantization).
- **fcm/**: Fuzzy C-Means clustering, membership map generation, cluster map generator.
- **models/**: Model architectures (MobileViTv3, EfficientNet, DenseNet, fusion blocks, layers).
- **preprocessing/**: Denoising, normalization, resizing, augmentation, dataloader, format standardizer.
- **scripts/**: CLI scripts for training, evaluation, export, precompute FCM features.
- **training/**: Training infrastructure, trainer, training loop, callbacks.
- **wandb/**: Experiment tracking logs.

## 2. MobileViTv3 Block
MobileViTv3 is a lightweight vision transformer block designed for mobile and edge devices. It combines:
- Local feature extraction (ConvBNAct)
- Patch unfolding/folding for transformer input
- Transformer encoder for global context
- Fusion of local and global features

**Mathematics:**
- Patch Unfolding: $x \in \mathbb{R}^{B \times C \times H \times W} \rightarrow \text{patches} \in \mathbb{R}^{B \times N \times (C \cdot p^2)}$
- Transformer: $\text{patches} \rightarrow \text{global features}$
- Patch Folding: $\text{global features} \rightarrow \mathbb{R}^{B \times (C \cdot p^2) \times H \times W}$
- Fusion: $\text{cat}(\text{local}, \text{folded}) \rightarrow \text{ConvBNAct}$

## 3. Fusion Mechanism
Fusion is performed by concatenating local features and FCM-encoded cluster maps, followed by an attention block:
- $\text{Fusion} = \text{cat}(x, fcm)$
- $\text{AttnMask} = \sigma(\text{Conv}(\text{Fusion}))$
- $\text{Output} = \text{Fusion} \times \text{AttnMask}$

## 4. Fuzzy C-Means (FCM) Mathematics
FCM is a clustering algorithm that assigns membership values to each pixel:
- Objective: $J = \sum_{i=1}^N \sum_{j=1}^C u_{ij}^m \|x_i - c_j\|^2$
- $u_{ij}$: membership of pixel $i$ to cluster $j$
- $c_j$: cluster center
- $m$: fuzziness parameter
- Cluster maps and membership maps are generated and fed to the model.

## 5. Metrics
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Confusion Matrix**: Visualizes TP, TN, FP, FN
- **ROC Curve & AUC**: Measures discrimination ability
- **Grad-CAM**: Visualizes model attention
- **Precision, Recall, F1**: Standard classification metrics

## 6. Dataset Description
- **BUSI Dataset**: Breast ultrasound images, labeled as benign, malignant, normal
- **Structure**: `datasets/benign/`, `datasets/malignant/`, `datasets/normal/`, `datasets/processed/`
- **Masks**: Segmentation masks for region annotation
- **Splits**: Train, validation, test splits defined in `config/dataset_config.yaml`
- **Preprocessing**: Denoising, normalization, resizing, augmentation

## 7. Training, Evaluation, Testing
- **Training**: Config-driven, supports experiment tracking, checkpointing, custom workflows
- **Evaluation**: Metrics, confusion matrix, ROC, Grad-CAM, export for deployment
- **Testing**: Final model tested on held-out set, results visualized and logged

## 8. Export & Deployment
- Models exported to ONNX, TFLite for mobile app integration
- Quantization for efficient inference
- Flutter app for real-time inference and reporting

---
## MobileFCMViTv3 Architecture

### Overview
MobileFCMViTv3 is a modular pipeline for breast cancer detection from ultrasound images, designed for extensibility, reproducibility, and mobile deployment.

### Key Components
- **Dataset Loader:** Handles ingestion, splitting, and transformation of ultrasound images. Supports DICOM, PNG, JPEG formats.
- **Preprocessing Pipeline:** Denoising, normalization, resizing, enhancement, and advanced methods (wavelet, histogram, thresholding).
- **Clustering Module:** Fuzzy C-Means clustering for region segmentation and feature extraction.
- **Model Architectures:** MobileViTv3 backbone, FCM feature encoder, fusion blocks, attention layers. Supports additional models (EfficientNet, DenseNet).
- **Training Infrastructure:** Config-driven training, optimizer, scheduler, callbacks, experiment tracking.
- **Evaluation:** Metrics, confusion matrix, ROC, Grad-CAM visualization.
- **Export & Quantization:** PyTorch → ONNX → TensorFlow → TFLite, with quantization for mobile.
- **Benchmarking:** Latency, memory, model size tests.
- **Mobile App:** Flutter-based app for inference, visualization, and report upload.
- **Integration:** Adapters for hospital/cloud systems, external APIs, experiment tracking.

### Data Flow
1. **Input:** Ultrasound images loaded via dataset loader
2. **Preprocessing:** Images processed through pipeline
3. **Clustering:** FCM generates cluster/membership maps
4. **Model:** Processed images and cluster maps fed to MobileFCMViTv3 model
5. **Training:** Model trained with config-driven workflow
6. **Evaluation:** Metrics and visualizations generated
7. **Export:** Model exported for deployment
8. **Mobile App:** TFLite model used for inference on device

### Modular Structure
- Each module is a separate Python package or script
- Config files control parameters and workflow
- CLI scripts automate training, evaluation, export, benchmarking
- Mobile app integrates exported model for real-time inference
