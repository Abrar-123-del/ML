"""
Configuration for ablation study experiments
"""

# Dataset configuration
DATASET_CONFIG = {
    'data_root': '/workspace/Abrar/Data/EgoGesture',
    'num_classes': 83,
    'n_segment': 8,
}

# Training hyperparameters
BASE_LR = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 40

# Ablation study configurations
ABLATION_CONFIGS = {
    'baseline_1_rgb_tsm_only': {
        'name': 'Baseline-1: RGB-only TSM',
        'description': 'RGB-only TSM without depth, transformer, or multi-scale pooling',
        'use_rgb': True,
        'use_depth': False,
        'use_transformer': False,
        'pooling_type': 'simple_avg',
        'use_augmentation': True,
        'use_progressive_unfreezing': False,
        'expected_acc': 86.0
    },
    'baseline_2_rgb_d_dual_stream': {
        'name': 'Baseline-2: RGB-D Dual-Stream TSM',
        'description': 'RGB-D dual-stream TSM without transformer or multi-scale pooling',
        'use_rgb': True,
        'use_depth': True,
        'use_transformer': False,
        'pooling_type': 'simple_avg',
        'use_augmentation': True,
        'use_progressive_unfreezing': False,
        'expected_acc': 91.0
    },
    'exp_a_add_transformer': {
        'name': 'Exp-A: Add Transformer Fusion',
        'description': 'RGB-D with transformer fusion, without multi-scale pooling',
        'use_rgb': True,
        'use_depth': True,
        'use_transformer': True,
        'pooling_type': 'simple_avg',
        'use_augmentation': True,
        'use_progressive_unfreezing': False,
        'expected_acc': 93.5
    },
    'exp_b_add_multiscale_pooling': {
        'name': 'Exp-B: Add Multi-scale Pooling',
        'description': 'RGB-D with transformer fusion and multi-scale pooling',
        'use_rgb': True,
        'use_depth': True,
        'use_transformer': True,
        'pooling_type': 'multi_scale',
        'use_augmentation': True,
        'use_progressive_unfreezing': False,
        'expected_acc': 94.5
    },
    'exp_c_full_no_augmentation': {
        'name': 'Exp-C: Full Model without Augmentation',
        'description': 'Full model without MixUp/CutMix augmentation',
        'use_rgb': True,
        'use_depth': True,
        'use_transformer': True,
        'pooling_type': 'multi_scale',
        'use_augmentation': False,
        'use_progressive_unfreezing': False,
        'expected_acc': 94.75
    },
    'exp_d_full_model_progressive_unfreezing': {
        'name': 'Exp-D: Full TSMTFN Model',
        'description': 'Full model with all components and progressive unfreezing',
        'use_rgb': True,
        'use_depth': True,
        'use_transformer': True,
        'pooling_type': 'multi_scale',
        'use_augmentation': True,
        'use_progressive_unfreezing': True,
        'expected_acc': 94.78
    }
}

# Output directories
OUTPUT_ROOT = '/workspace/Abrar/Results/ablation_results'
