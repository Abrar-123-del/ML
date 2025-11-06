"""
Metrics computation utilities
"""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time


def calculate_accuracy(outputs, labels, topk=(1, 5)):
    """
    Calculate top-k accuracy
    
    Args:
        outputs: Model predictions (batch_size, num_classes)
        labels: Ground truth labels (batch_size,)
        topk: Tuple of k values for top-k accuracy
    
    Returns:
        List of top-k accuracies
    """
    maxk = max(topk)
    batch_size = labels.size(0)
    
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    
    return res


def compute_per_class_accuracy(model, dataloader, device, num_classes=83):
    """
    Compute per-class accuracy
    
    Args:
        model: Neural network model
        dataloader: Data loader
        device: Device to compute on
        num_classes: Number of classes
    
    Returns:
        dict with per-class accuracy
    """
    model.eval()
    
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    with torch.no_grad():
        for batch in dataloader:
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
            _, predicted = outputs.max(1)
            
            # Update per-class statistics
            for i in range(label.size(0)):
                label_idx = label[i].item()
                class_total[label_idx] += 1
                if predicted[i] == label[i]:
                    class_correct[label_idx] += 1
    
    # Compute per-class accuracy
    per_class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[f'class_{i}'] = 100.0 * class_correct[i] / class_total[i]
        else:
            per_class_acc[f'class_{i}'] = 0.0
    
    return per_class_acc


def calculate_inference_speed(model, input_shape, device, num_runs=100):
    """
    Calculate inference speed (FPS)
    
    Args:
        model: Neural network model
        input_shape: Tuple of input shape
        device: Device to run on
        num_runs: Number of inference runs for averaging
    
    Returns:
        dict with 'fps' and 'latency_ms'
    """
    model.eval()
    
    # Create dummy input
    rgb_input = torch.randn(*input_shape).to(device)
    depth_input = torch.randn(*input_shape).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(rgb_input, depth_input)
    
    # Measure inference time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(rgb_input, depth_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_runs) * 1000  # Convert to ms
    fps = num_runs / total_time
    
    return {
        'fps': fps,
        'latency_ms': avg_latency
    }


def calculate_flops(model, input_shape=(1, 3, 224, 224)):
    """
    Estimate FLOPs for the model
    
    Args:
        model: Neural network model
        input_shape: Input shape
    
    Returns:
        Estimated GFLOPs
    """
    # Simple estimation based on parameter count
    # For accurate measurement, use libraries like thop or fvcore
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # Rough estimation: 2 FLOPs per MAC (multiply-accumulate)
    # Assuming each parameter is used once per forward pass
    h, w = input_shape[-2:]
    spatial_size = h * w
    
    # Very rough estimation
    estimated_flops = total_params * spatial_size * 2
    gflops = estimated_flops / 1e9
    
    return gflops


def compute_metrics(model, dataloader, device, num_classes=83, input_shape=(8, 3, 224, 224)):
    """
    Compute comprehensive evaluation metrics
    
    Args:
        model: Neural network model
        dataloader: Data loader
        device: Device to compute on
        num_classes: Number of classes
        input_shape: Input shape for speed calculation
    
    Returns:
        dict with various metrics
    """
    # Top-1 and Top-5 accuracy
    model.eval()
    total = 0
    top1_correct = 0
    top5_correct = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
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
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += label.size(0)
            top1_correct += predicted.eq(label).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            correct_k = top5_pred.eq(label.view(1, -1).expand_as(top5_pred))
            top5_correct += correct_k[:5].sum().item()
            
            # Store for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    # Calculate accuracies
    top1_acc = 100. * top1_correct / total
    top5_acc = 100. * top5_correct / total
    
    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(model, dataloader, device, num_classes)
    
    # Inference speed
    speed_metrics = calculate_inference_speed(model, input_shape, device)
    
    # Parameter count
    total_params = model.get_parameter_count() if hasattr(model, 'get_parameter_count') else \
                   sum(p.numel() for p in model.parameters())
    trainable_params = model.get_trainable_parameter_count() if hasattr(model, 'get_trainable_parameter_count') else \
                       sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # FLOPs
    gflops = calculate_flops(model, input_shape)
    
    metrics = {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'per_class_accuracy': per_class_acc,
        'inference_fps': speed_metrics['fps'],
        'inference_latency_ms': speed_metrics['latency_ms'],
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'gflops': gflops
    }
    
    return metrics
