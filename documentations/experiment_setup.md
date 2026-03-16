## Experiment Setup Guide

### 1. Dataset Preparation
- Organize ultrasound images in folders as specified in `config/dataset_config.yaml`
- Supported formats: DICOM, PNG, JPEG
- Optionally, use `datasets/external_api_adapter.py` to fetch data from external sources

### 2. Configuration
- Edit YAML files in `config/`:
	- `dataset_config.yaml`: dataset paths, splits, augmentation
	- `training_config.yaml`: epochs, batch size, optimizer, scheduler, experiment tracking
	- `model_config.yaml`: model architecture, parameters

### 3. Preprocessing Pipeline
- Configure preprocessing steps in `preprocessing/preprocessing_pipeline.py`
- Enable/disable denoising, normalization, resizing, enhancement, advanced methods

### 4. Experiment Tracking
- Supported: Weights & Biases, MLflow, Neptune
- Set API keys in environment variables or config
- Enable tracking in `training_config.yaml`

### 5. Custom Workflows
- Edit or add scripts in `training/custom_workflows.py` for ablation studies, baseline comparisons, or advanced experiments

### 6. Running Experiments
- Use CLI scripts:
	```bash
	python scripts/train.py --config_dir config/
	python scripts/evaluate.py --config_dir config/
	```
- Results saved in `results/`

### 7. Logging & Visualization
- Metrics, losses, predictions, confusion matrix, ROC, Grad-CAM visualizations
- Use experiment tracking dashboards for monitoring

### 8. Troubleshooting
- Check logs in `results/` and experiment tracking platform
- Verify dataset paths and config parameters
