"""
Training utilities for TSM network
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import numpy as np
from tqdm import tqdm
import json


def train_epoch(model, dataloader, criterion, optimizer, device, 
                use_augmentation=False, epoch=0):
    """
    Train for one epoch
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        use_augmentation: Whether to use MixUp/CutMix
        epoch: Current epoch number
    
    Returns:
        dict with 'loss' and 'accuracy'
    """
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Get data
        label = batch['label'].to(device)
        
        # Prepare inputs based on what model expects
        rgb_input = None
        depth_input = None
        
        if 'rgb' in batch:
            rgb_frames = batch['rgb'].to(device)
            # Reshape: (batch, n_segment, C, H, W) -> (batch*n_segment, C, H, W)
            b, t, c, h, w = rgb_frames.size()
            rgb_input = rgb_frames.view(b * t, c, h, w)
        
        if 'depth' in batch:
            depth_frames = batch['depth'].to(device)
            b, t, c, h, w = depth_frames.size()
            depth_input = depth_frames.view(b * t, c, h, w)
        
        # Apply MixUp/CutMix if enabled
        if use_augmentation and np.random.random() > 0.5:
            # Simple mixup implementation
            indices = torch.randperm(label.size(0))
            label2 = label[indices]
            lam = np.random.beta(1.0, 1.0)
            
            if rgb_input is not None:
                rgb_input = lam * rgb_input + (1 - lam) * rgb_input[indices.repeat_interleave(t)]
            if depth_input is not None:
                depth_input = lam * depth_input + (1 - lam) * depth_input[indices.repeat_interleave(t)]
            
            # Forward pass
            outputs = model(rgb_input, depth_input)
            
            # Mixed loss
            loss = lam * criterion(outputs, label) + (1 - lam) * criterion(outputs, label2)
        else:
            # Forward pass
            outputs = model(rgb_input, depth_input)
            loss = criterion(outputs, label)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def validate(model, dataloader, criterion, device):
    """
    Validate model
    
    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        dict with 'loss', 'accuracy', 'top5_accuracy'
    """
    model.eval()
    
    total_loss = 0
    correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            label = batch['label'].to(device)
            
            # Prepare inputs
            rgb_input = None
            depth_input = None
            
            if 'rgb' in batch:
                rgb_frames = batch['rgb'].to(device)
                b, t, c, h, w = rgb_frames.size()
                rgb_input = rgb_frames.view(b * t, c, h, w)
            
            if 'depth' in batch:
                depth_frames = batch['depth'].to(device)
                b, t, c, h, w = depth_frames.size()
                depth_input = depth_frames.view(b * t, c, h, w)
            
            # Forward pass
            outputs = model(rgb_input, depth_input)
            loss = criterion(outputs, label)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            correct_k = top5_pred.eq(label.view(1, -1).expand_as(top5_pred))
            top5_correct += correct_k[:5].sum().item()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total,
        'top5_accuracy': 100. * top5_correct / total
    }


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'max'
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop


def train_main(model, train_loader, val_loader, config, save_dir):
    """
    Main training function
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
        save_dir: Directory to save checkpoints and logs
    
    Returns:
        dict with training history
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Base hyperparameters
    base_lr = config.get('base_lr', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    epochs = config.get('epochs', 40)
    use_augmentation = config.get('use_augmentation', True)
    use_progressive_unfreezing = config.get('use_progressive_unfreezing', False)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, mode='max')
    
    # Progressive unfreezing
    if use_progressive_unfreezing:
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Only train classifier initially
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_top5_acc': [],
        'lr': [],
        'epoch_time': []
    }
    
    best_val_acc = 0
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Progressive unfreezing logic
        if use_progressive_unfreezing:
            if epoch == 5:  # Unfreeze more layers after 5 epochs
                print("Unfreezing backbone layers...")
                for param in model.parameters():
                    param.requires_grad = True
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_augmentation=use_augmentation, epoch=epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record metrics
        epoch_time = time.time() - epoch_start_time
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_top5_acc'].append(val_metrics['top5_accuracy'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(epoch_time)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
              f"Top-5 Acc: {val_metrics['top5_accuracy']:.2f}%")
        print(f"  Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  Saved best model with Val Acc: {best_val_acc:.2f}%")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'config': config
        }, os.path.join(save_dir, 'last_checkpoint.pth'))
        
        # Early stopping check
        if early_stopping(val_metrics['accuracy']):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save final history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    return history
