"""
EgoGesture Dataset for RGB-D Gesture Recognition

This module provides a PyTorch Dataset class for loading and preprocessing
EgoGesture dataset with RGB and depth modalities.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random


class EgoGestureFusionDataset(Dataset):
    """
    EgoGesture RGB-D Fusion Dataset
    
    Supports loading RGB and depth frames with various augmentation strategies.
    """
    
    def __init__(self, data_root, split='train', n_segment=8, 
                 use_rgb=True, use_depth=True, 
                 use_augmentation=True, transform=None):
        """
        Args:
            data_root: Root directory of the dataset
            split: 'train', 'val', or 'test'
            n_segment: Number of frames to sample per video
            use_rgb: Whether to load RGB frames
            use_depth: Whether to load depth frames
            use_augmentation: Whether to apply data augmentation
            transform: Optional custom transform
        """
        self.data_root = data_root
        self.split = split
        self.n_segment = n_segment
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_augmentation = use_augmentation
        
        # Load dataset metadata
        self.samples = self._load_samples()
        
        # Define transforms
        if transform is None:
            if split == 'train' and use_augmentation:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                          saturation=0.4, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        # Depth transform (single channel)
        if split == 'train' and use_augmentation:
            self.depth_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
            ])
        else:
            self.depth_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
    
    def _load_samples(self):
        """
        Load dataset samples
        
        Returns a list of dictionaries with:
        - video_path: Path to video/frames
        - label: Gesture class label
        - num_frames: Total number of frames
        """
        # Mock implementation - in real scenario, this would load from annotation files
        samples = []
        
        # Generate synthetic sample data for demonstration
        # In real implementation, this would read from dataset files
        num_samples = 1000 if self.split == 'train' else 200
        num_classes = 83  # EgoGesture has 83 classes
        
        for i in range(num_samples):
            sample = {
                'video_id': f'{self.split}_video_{i:04d}',
                'label': i % num_classes,
                'num_frames': random.randint(20, 100)
            }
            samples.append(sample)
        
        return samples
    
    def _load_frame(self, video_id, frame_idx, modality='rgb'):
        """
        Load a single frame
        
        Args:
            video_id: Video identifier
            frame_idx: Frame index
            modality: 'rgb' or 'depth'
        
        Returns:
            frame: PIL Image
        """
        # Mock implementation - generates synthetic frames
        # In real scenario, this would load from disk
        
        if modality == 'rgb':
            # Generate random RGB image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            # Generate random depth image (grayscale)
            img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            img = np.stack([img] * 3, axis=-1)  # Convert to 3-channel for transforms
        
        return Image.fromarray(img)
    
    def _sample_indices(self, num_frames):
        """
        Sample frame indices from video
        
        Args:
            num_frames: Total number of frames in video
        
        Returns:
            indices: List of frame indices to sample
        """
        if num_frames <= self.n_segment:
            # If video is shorter than n_segment, repeat frames
            indices = list(range(num_frames))
            while len(indices) < self.n_segment:
                indices.append(indices[-1])
        else:
            # Uniform sampling
            tick = num_frames / float(self.n_segment)
            indices = [int(tick * i) for i in range(self.n_segment)]
        
        return indices
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            dict with keys:
                - 'rgb': RGB frames (n_segment, 3, H, W) if use_rgb
                - 'depth': Depth frames (n_segment, 1, H, W) if use_depth
                - 'label': Class label
        """
        sample = self.samples[idx]
        video_id = sample['video_id']
        label = sample['label']
        num_frames = sample['num_frames']
        
        # Sample frame indices
        frame_indices = self._sample_indices(num_frames)
        
        result = {'label': label}
        
        # Load RGB frames
        if self.use_rgb:
            rgb_frames = []
            for frame_idx in frame_indices:
                frame = self._load_frame(video_id, frame_idx, modality='rgb')
                frame = self.transform(frame)
                rgb_frames.append(frame)
            result['rgb'] = torch.stack(rgb_frames, dim=0)
        
        # Load depth frames
        if self.use_depth:
            depth_frames = []
            for frame_idx in frame_indices:
                frame = self._load_frame(video_id, frame_idx, modality='depth')
                frame = self.depth_transform(frame)
                # Take only first channel for depth
                frame = frame[0:1, :, :]
                depth_frames.append(frame)
            result['depth'] = torch.stack(depth_frames, dim=0)
        
        return result


def random_mix(data1, data2, labels1, labels2, alpha=1.0):
    """
    Mix two samples using MixUp or CutMix
    
    Args:
        data1, data2: Input data tensors
        labels1, labels2: Labels for the samples
        alpha: Beta distribution parameter
    
    Returns:
        mixed_data: Mixed data
        mixed_labels: Mixed labels (one-hot encoded)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    # Randomly choose between MixUp and CutMix
    if random.random() > 0.5:
        # MixUp
        mixed_data = lam * data1 + (1 - lam) * data2
    else:
        # CutMix
        mixed_data = data1.clone()
        _, _, h, w = data1.size()
        
        # Generate random box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        mixed_data[:, :, bby1:bby2, bbx1:bbx2] = data2[:, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    return mixed_data, lam


def create_dataloader(data_root, split='train', batch_size=16, n_segment=8,
                     use_rgb=True, use_depth=True, use_augmentation=True,
                     num_workers=4, shuffle=None):
    """
    Create a DataLoader for EgoGesture dataset
    
    Args:
        data_root: Root directory of the dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        n_segment: Number of frames per video
        use_rgb: Whether to use RGB stream
        use_depth: Whether to use depth stream
        use_augmentation: Whether to apply augmentation
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data (None = auto based on split)
    
    Returns:
        dataloader: PyTorch DataLoader instance
    """
    dataset = EgoGestureFusionDataset(
        data_root=data_root,
        split=split,
        n_segment=n_segment,
        use_rgb=use_rgb,
        use_depth=use_depth,
        use_augmentation=use_augmentation
    )
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
