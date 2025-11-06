"""
Configs package
"""
from .ablation_config import (
    DATASET_CONFIG,
    BASE_LR,
    WEIGHT_DECAY,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    ABLATION_CONFIGS,
    OUTPUT_ROOT
)

__all__ = [
    'DATASET_CONFIG',
    'BASE_LR',
    'WEIGHT_DECAY',
    'BATCH_SIZE',
    'NUM_WORKERS',
    'EPOCHS',
    'ABLATION_CONFIGS',
    'OUTPUT_ROOT'
]
