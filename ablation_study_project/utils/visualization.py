"""
Visualization utilities for ablation study results
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def plot_learning_curves(history, save_path, title='Training History'):
    """
    Plot training and validation learning curves
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    if 'val_top5_acc' in history:
        axes[0, 1].plot(epochs, history['val_top5_acc'], 'g-', label='Val Top-5 Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 0].plot(epochs, history['lr'], 'g-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Epoch time
    if 'epoch_time' in history:
        axes[1, 1].plot(epochs, history['epoch_time'], 'm-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_learning_curves(all_histories, variant_names, save_path):
    """
    Plot learning curves for all variants on the same plot
    
    Args:
        all_histories: List of history dictionaries
        variant_names: List of variant names
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_names)))
    
    # Validation loss
    for i, (history, name) in enumerate(zip(all_histories, variant_names)):
        epochs = range(1, len(history['val_loss']) + 1)
        axes[0].plot(epochs, history['val_loss'], 
                    color=colors[i], label=name, linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Validation Loss Comparison', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Validation accuracy
    for i, (history, name) in enumerate(zip(all_histories, variant_names)):
        epochs = range(1, len(history['val_acc']) + 1)
        axes[1].plot(epochs, history['val_acc'], 
                    color=colors[i], label=name, linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1].set_title('Validation Accuracy Comparison', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_bar_chart(results, metric_name, save_path, ylabel=None):
    """
    Create bar chart comparing metrics across variants
    
    Args:
        results: Dictionary with variant names as keys and metric values
        metric_name: Name of the metric to plot
        save_path: Path to save the plot
        ylabel: Y-axis label (defaults to metric_name)
    """
    plt.figure(figsize=(12, 6))
    
    variants = list(results.keys())
    values = list(results.values())
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(variants)))
    bars = plt.bar(range(len(variants)), values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Model Variant', fontsize=12)
    plt.ylabel(ylabel or metric_name, fontsize=12)
    plt.title(f'{metric_name} Comparison Across Variants', fontsize=14)
    plt.xticks(range(len(variants)), variants, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_ablation_summary(all_results, variant_names, save_path):
    """
    Create comprehensive ablation summary visualization
    
    Args:
        all_results: List of result dictionaries for each variant
        variant_names: List of variant names
        save_path: Path to save the PDF
    """
    with PdfPages(save_path) as pdf:
        # Page 1: Top-1 and Top-5 Accuracy Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        top1_accs = [r['top1_accuracy'] for r in all_results]
        top5_accs = [r['top5_accuracy'] for r in all_results]
        
        x = np.arange(len(variant_names))
        width = 0.35
        
        axes[0].bar(x, top1_accs, width, label='Top-1', alpha=0.8, color='steelblue')
        axes[0].set_xlabel('Model Variant')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Top-1 Accuracy Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(variant_names, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(top1_accs):
            axes[0].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=9)
        
        axes[1].bar(x, top5_accs, width, label='Top-5', alpha=0.8, color='coral')
        axes[1].set_xlabel('Model Variant')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Top-5 Accuracy Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(variant_names, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(top5_accs):
            axes[1].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Model Complexity Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        params = [r['total_parameters'] / 1e6 for r in all_results]  # In millions
        gflops = [r['gflops'] for r in all_results]
        fps = [r['inference_fps'] for r in all_results]
        latency = [r['inference_latency_ms'] for r in all_results]
        
        # Parameters
        axes[0, 0].bar(x, params, alpha=0.8, color='lightgreen')
        axes[0, 0].set_xlabel('Model Variant')
        axes[0, 0].set_ylabel('Parameters (M)')
        axes[0, 0].set_title('Model Parameters')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(variant_names, rotation=45, ha='right')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # GFLOPs
        axes[0, 1].bar(x, gflops, alpha=0.8, color='lightcoral')
        axes[0, 1].set_xlabel('Model Variant')
        axes[0, 1].set_ylabel('GFLOPs')
        axes[0, 1].set_title('Computational Complexity')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(variant_names, rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # FPS
        axes[1, 0].bar(x, fps, alpha=0.8, color='skyblue')
        axes[1, 0].set_xlabel('Model Variant')
        axes[1, 0].set_ylabel('FPS')
        axes[1, 0].set_title('Inference Speed')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(variant_names, rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Latency
        axes[1, 1].bar(x, latency, alpha=0.8, color='plum')
        axes[1, 1].set_xlabel('Model Variant')
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].set_title('Inference Latency')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(variant_names, rotation=45, ha='right')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Component Contribution Analysis
        if len(all_results) >= 2:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Calculate improvements relative to baseline
            baseline_acc = all_results[0]['top1_accuracy']
            improvements = [r['top1_accuracy'] - baseline_acc for r in all_results]
            
            bars = ax.barh(variant_names, improvements, alpha=0.8, color='teal')
            
            # Color negative improvements differently
            for i, (bar, imp) in enumerate(zip(bars, improvements)):
                if imp < 0:
                    bar.set_color('salmon')
                ax.text(imp + 0.1, i, f'{imp:+.2f}%', va='center', fontsize=10)
            
            ax.set_xlabel('Accuracy Improvement over Baseline (%)')
            ax.set_title('Component Contribution to Performance')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def plot_component_ablation(component_contributions, save_path):
    """
    Visualize the contribution of each component
    
    Args:
        component_contributions: Dictionary with component names and their contributions
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    components = list(component_contributions.keys())
    contributions = list(component_contributions.values())
    
    colors = ['green' if c > 0 else 'red' for c in contributions]
    
    bars = plt.barh(components, contributions, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, contributions):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{value:+.2f}%',
                ha='left' if value > 0 else 'right',
                va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Accuracy Contribution (%)', fontsize=12)
    plt.title('Component Ablation Analysis', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
