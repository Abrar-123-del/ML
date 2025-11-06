"""
Datasets package for EgoGesture RGB-D dataset
"""
from .egogesture_dataset import EgoGestureFusionDataset, create_dataloader

__all__ = ['EgoGestureFusionDataset', 'create_dataloader']
