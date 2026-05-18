# Model Training and Evaluation

## Overview
This experiment contains the training, inference, and evaluation pipeline for fine-tuning Qwen3-VL models of different sizes (2B, 4B, 8B) on the CoMa dataset.

## Model Directory Structure
Each model directory (`2B/`, `4B/`, `8B/`) contains:
- `checkpoints/` - Training checkpoints
- `logs/` - Training loss logs
- `submit_images/` - Generated massings visualized in urban context
- `config.json` - Main training configuration
- `inference_config.json` - Inference configuration (produces `submit.json`)
- `save_model.json` - Configuration for pushing to HuggingFace Hub
- `submit_visualization_config.json` - Visualization configuration (produces images and `visualized_submit.json`)

## Additional Directories
- `visualization/` - Generates loss curves from training logs

## Root Files
- `validation_config.json` - Metrics computation configuration (produces `result.json`)
- `result.json` - Final evaluation metrics for all models

## Execution Commands

### Training
```bash
torchrun --nproc_per_node=8 run.py --config_path path/to/model/config.json
```

### Inference
```bash
python run.py --config_path path/to/model/inference_config.json
```

### Visualization
```bash
python run.py --config_path path/to/model/submit_visualization_config.json
```

### Metrics Validation
```bash
python run.py --config_path validation_config.json
```


## Notes
- All models are from Qwen3-VL series
- Training uses distributed processing with 8 GPUs
- Pipeline produces both quantitative metrics and qualitative visualizations