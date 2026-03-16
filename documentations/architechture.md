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
