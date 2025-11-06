# Ablation Study for TSM Two-Stream Fusion Network

This directory contains a comprehensive ablation study framework for evaluating different components of the TSM (Temporal Shift Module) Two-Stream Fusion Network for RGB-D gesture recognition.

## Overview

The ablation study systematically evaluates the contribution of each component by training and testing 6 different model variants:

1. **Baseline-1**: RGB-only TSM (no depth, no transformer, simple avg pooling)
2. **Baseline-2**: RGB-D dual-stream TSM (no transformer, simple avg pooling)
3. **Exp-A**: RGB-D + Transformer fusion (no multi-scale pooling)
4. **Exp-B**: RGB-D + Transformer + Multi-scale pooling
5. **Exp-C**: Full model without MixUp/CutMix augmentation
6. **Exp-D**: Full TSMTFN model with all components and progressive unfreezing

## Project Structure

```
ablation_study_project/
├── models/
│   ├── __init__.py
│   └── tsm_network.py          # TSM Two-Stream Fusion Network architecture
├── datasets/
│   ├── __init__.py
│   └── egogesture_dataset.py   # EgoGesture dataset loader
├── utils/
│   ├── __init__.py
│   ├── training.py             # Training utilities
│   ├── metrics.py              # Evaluation metrics
│   └── visualization.py        # Plotting and visualization
├── configs/
│   ├── __init__.py
│   └── ablation_config.py      # Experiment configurations
└── ablation_study.py           # Main ablation study script
```

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install tqdm scikit-learn Pillow
```

### Python Version

- Python 3.7 or higher

## Usage

### Basic Usage

Run the complete ablation study with default settings:

```bash
python ablation_study.py
```

### Custom Dataset Path

Specify custom dataset path:

```bash
python ablation_study.py --data-root /path/to/egogesture/dataset
```

### Custom Output Directory

Specify custom output directory:

```bash
python ablation_study.py --output-root /path/to/results
```

### Resume Training

Resume from existing checkpoints:

```bash
python ablation_study.py --resume
```

### CPU-Only Mode

Run on CPU (useful for testing):

```bash
python ablation_study.py --device cpu
```

## Configuration

### Hyperparameters

Default hyperparameters are defined in `configs/ablation_config.py`:

- **Learning Rate**: 0.001
- **Weight Decay**: 1e-4
- **Batch Size**: 16
- **Epochs**: 40 (with early stopping)
- **Number of Segments**: 8 frames per video

### Ablation Configurations

Each variant has specific configuration:

```python
{
    'use_rgb': True/False,
    'use_depth': True/False,
    'use_transformer': True/False,
    'pooling_type': 'simple_avg' or 'multi_scale',
    'use_augmentation': True/False,
    'use_progressive_unfreezing': True/False
}
```

## Output Structure

Results are saved in the following structure:

```
/workspace/Abrar/Results/ablation_results/
├── baseline_1_rgb_tsm_only/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── last_checkpoint.pth
│   ├── logs/
│   │   └── training_log.txt
│   ├── plots/
│   │   └── learning_curves.png
│   ├── metrics.json
│   └── training_history.json
├── baseline_2_rgb_d_dual_stream/
├── exp_a_add_transformer/
├── exp_b_add_multiscale_pooling/
├── exp_c_full_no_augmentation/
├── exp_d_full_model_progressive_unfreezing/
├── ablation_comparison_table.csv
├── ablation_comparison_table.json
├── ablation_comparison_report.txt
├── component_contribution_analysis.txt
├── accuracy_comparison_plot.png
├── component_ablation_visualization.png
├── learning_curves_all_variants.png
└── ablation_summary_report.pdf
```

## Metrics Tracked

For each variant, the following metrics are computed:

### Training Metrics
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch
- Top-5 validation accuracy
- Learning rate schedule
- Training time per epoch

### Evaluation Metrics
- **Accuracy**: Top-1 and Top-5 test accuracy
- **Per-class accuracy**: Accuracy for each gesture class
- **Model complexity**: Total and trainable parameters
- **Computational cost**: Estimated GFLOPs
- **Inference speed**: FPS and latency (ms)

## Expected Results

Based on the ablation study, expected Top-1 accuracies are:

| Variant | Expected Accuracy |
|---------|------------------|
| Baseline-1 | ~85-87% |
| Baseline-2 | ~90-92% |
| Exp-A | ~93-94% |
| Exp-B | ~94-95% |
| Exp-C | ~94.5-95% |
| Exp-D | ~94.78% |

## Visualizations Generated

1. **Learning Curves**: Training/validation loss and accuracy for each variant
2. **Comparison Bar Charts**: Side-by-side comparison of key metrics
3. **Component Contribution**: Analysis of each component's impact
4. **Comprehensive PDF Report**: Multi-page summary with all metrics

## Component Analysis

The ablation study automatically generates a component contribution analysis that:

- Identifies which components are critical for performance
- Ranks components by their contribution
- Provides recommendations for model design
- Shows diminishing returns of adding more components

## Customization

### Adding New Variants

To add new variants, edit `configs/ablation_config.py`:

```python
ABLATION_CONFIGS['my_new_variant'] = {
    'name': 'My New Variant',
    'description': 'Description of the variant',
    'use_rgb': True,
    'use_depth': True,
    'use_transformer': False,
    'pooling_type': 'simple_avg',
    'use_augmentation': True,
    'use_progressive_unfreezing': False,
    'expected_acc': 90.0
}
```

### Modifying Model Architecture

Edit `models/tsm_network.py` to modify the TSM network architecture.

### Custom Data Augmentation

Edit `datasets/egogesture_dataset.py` to add custom augmentation strategies.

## Memory Management

The ablation study includes automatic memory management:

- GPU cache is cleared between experiments
- Checkpoints are saved periodically
- Early stopping prevents unnecessary training

## Logging

All training progress is logged to:
- Console output
- `ablation_study.log` file
- Individual training logs per variant

## Error Handling

The study includes robust error handling:

- Individual variant failures don't stop the entire study
- Checkpoint resumption for interrupted training
- Detailed error logging with stack traces

## Citation

If you use this ablation study framework in your research, please cite:

```
@misc{tsm_ablation_study,
  title={Ablation Study for TSM Two-Stream Fusion Network},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/Abrar-123-del/ML}}
}
```

## License

This project follows the same license as the parent repository.

## Contact

For questions or issues, please open an issue on GitHub.
