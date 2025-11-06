# Quick Reference Guide

## Command Line Usage

### Run Full Ablation Study
```bash
python ablation_study.py
```

### With Custom Dataset Path
```bash
python ablation_study.py --data-root /path/to/egogesture
```

### With Custom Output Directory
```bash
python ablation_study.py --output-root /path/to/results
```

### Resume Interrupted Study
```bash
python ablation_study.py --resume
```

### CPU Mode (for testing)
```bash
python ablation_study.py --device cpu
```

## Quick Demo

### Show All Configurations
```bash
python example_ablation_demo.py --show-configs
```

### Run Quick Demo (2 epochs)
```bash
python example_ablation_demo.py --run-demo
```

## Verify Setup

```bash
python test_ablation_setup.py
```

## Python API

### Basic Usage
```python
from ablation_study import AblationStudy

study = AblationStudy(
    output_root='./results',
    data_root='/path/to/data'
)
study.run()
```

### Create Model
```python
from ablation_study_project.models import create_model

model = create_model({
    'num_classes': 83,
    'n_segment': 8,
    'use_rgb': True,
    'use_depth': True,
    'use_transformer': True,
    'pooling_type': 'multi_scale'
})
```

### Create DataLoader
```python
from ablation_study_project.datasets import create_dataloader

train_loader = create_dataloader(
    data_root='/path/to/data',
    split='train',
    batch_size=16,
    n_segment=8,
    use_rgb=True,
    use_depth=True,
    use_augmentation=True
)
```

### Train Model
```python
from ablation_study_project.utils import train_main

history = train_main(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config={
        'base_lr': 0.001,
        'weight_decay': 1e-4,
        'epochs': 40,
        'use_augmentation': True,
        'use_progressive_unfreezing': False
    },
    save_dir='./checkpoints'
)
```

### Evaluate Model
```python
from ablation_study_project.utils import compute_metrics

metrics = compute_metrics(
    model=model,
    dataloader=test_loader,
    device=device,
    num_classes=83
)

print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
print(f"FPS: {metrics['inference_fps']:.2f}")
```

## Configuration

### Modify Hyperparameters
Edit `ablation_study_project/configs/ablation_config.py`:

```python
BASE_LR = 0.001          # Learning rate
WEIGHT_DECAY = 1e-4      # L2 regularization
BATCH_SIZE = 16          # Batch size
EPOCHS = 40              # Number of epochs
NUM_WORKERS = 4          # Data loader workers
```

### Add New Variant
```python
ABLATION_CONFIGS['my_variant'] = {
    'name': 'My Custom Variant',
    'description': 'Description here',
    'use_rgb': True,
    'use_depth': True,
    'use_transformer': True,
    'pooling_type': 'multi_scale',
    'use_augmentation': True,
    'use_progressive_unfreezing': False,
    'expected_acc': 92.0
}
```

## Output Files

### Individual Variant Results
- `checkpoints/best_model.pth` - Best model weights
- `checkpoints/training_history.json` - Training metrics
- `logs/training_log.txt` - Training log
- `plots/learning_curves.png` - Learning curves plot
- `metrics.json` - Evaluation metrics

### Comparison Results
- `ablation_comparison_table.csv` - CSV table
- `ablation_comparison_table.json` - JSON data
- `ablation_comparison_report.txt` - Text report
- `component_contribution_analysis.txt` - Component analysis
- `accuracy_comparison_plot.png` - Bar chart
- `component_ablation_visualization.png` - Component plot
- `learning_curves_all_variants.png` - All curves
- `ablation_summary_report.pdf` - PDF summary

## Key Metrics

### Accuracy Metrics
- `top1_accuracy` - Top-1 classification accuracy
- `top5_accuracy` - Top-5 classification accuracy
- `per_class_accuracy` - Accuracy per gesture class

### Efficiency Metrics
- `total_parameters` - Total model parameters
- `trainable_parameters` - Trainable parameters
- `gflops` - Computational complexity
- `inference_fps` - Frames per second
- `inference_latency_ms` - Latency in milliseconds

## Troubleshooting

### Import Error
```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/ML"
```

### Out of Memory
```python
# Reduce batch size in config
BATCH_SIZE = 8  # or smaller
```

### Slow Data Loading
```python
# Increase number of workers
NUM_WORKERS = 8  # or higher
```

## Tips

1. **Always test first**: Run `test_ablation_setup.py` before full training
2. **Use resume**: Save time with `--resume` flag
3. **Monitor progress**: Check `ablation_study.log` file
4. **Clear GPU cache**: Between experiments if needed
5. **Use demo first**: Test with `example_ablation_demo.py`

## File Locations

- Main script: `ablation_study.py`
- Demo script: `example_ablation_demo.py`
- Test script: `test_ablation_setup.py`
- Models: `ablation_study_project/models/`
- Datasets: `ablation_study_project/datasets/`
- Utils: `ablation_study_project/utils/`
- Configs: `ablation_study_project/configs/`
- Docs: `ABLATION_STUDY_README.md`
- Summary: `PROJECT_SUMMARY.md`

## Support

For detailed documentation, see:
- `ABLATION_STUDY_README.md` - Complete usage guide
- `PROJECT_SUMMARY.md` - Project overview and features

For help:
```bash
python ablation_study.py --help
python example_ablation_demo.py --help
```
