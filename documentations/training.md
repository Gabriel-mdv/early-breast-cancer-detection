## Training Guide

### 1. Configuration
- Edit `config/training_config.yaml` for training parameters:
	- epochs, batch size, learning rate, optimizer, scheduler, experiment tracking
- Edit `config/model_config.yaml` for model architecture
- Edit `config/dataset_config.yaml` for dataset paths and splits

### 2. Preprocessing
- Configure preprocessing steps in `preprocessing/preprocessing_pipeline.py`
- Enable/disable denoising, normalization, resizing, enhancement

### 3. Starting Training
```bash
python scripts/train.py --config_dir config/
```
- Supports resume, checkpointing, experiment tracking
- Logs metrics, losses, checkpoints to `results/`

### 4. Experiment Tracking
- Supported platforms: Weights & Biases, MLflow, Neptune
- Enable in `training_config.yaml` and set API keys

### 5. Custom Workflows
- Edit `training/custom_workflows.py` for advanced experiments
- Supports ablation studies, baseline comparisons

### 6. Monitoring & Visualization
- Metrics, losses, predictions, confusion matrix, ROC, Grad-CAM
- Visualize in experiment tracking dashboards

### 7. Troubleshooting
- Check logs in `results/`
- Verify config parameters and dataset paths
- For CUDA errors, check GPU setup and PyTorch installation
