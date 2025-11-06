#!/usr/bin/env python3
"""
TSMTFN Network Architecture Flowchart Generator

This script generates a comprehensive visualization of the Two-Stream Multi-scale
Temporal Fusion Network (TSMTFN) architecture using matplotlib and graphviz.

The visualization includes:
- Two-stream RGB-D architecture
- Tensor dimensions at each stage
- GFLOPs breakdown for each component
- Color-coded modules for visual clarity
- Professional styling suitable for academic publication
- High-resolution output (300 DPI) in multiple formats

Author: Generated for ML Repository
Date: November 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Output settings
OUTPUT_DIR = Path("Results")
OUTPUT_FILENAME = "TSMTFN_Architecture"
DPI = 300
FORMATS = ['png', 'pdf', 'svg']

# Figure settings
FIGURE_WIDTH = 20
FIGURE_HEIGHT = 14
BACKGROUND_COLOR = 'white'

# Color scheme for different components
COLORS = {
    'input': '#E8F4F8',          # Light blue
    'rgb_stream': '#FFE5E5',     # Light red
    'depth_stream': '#E5F5E5',   # Light green
    'fusion': '#FFF4E5',         # Light orange
    'transformer': '#F0E5FF',    # Light purple
    'pooling': '#FFFFE5',        # Light yellow
    'classifier': '#FFE5F5',     # Light pink
    'output': '#E5E5E5',         # Light gray
    'arrow': '#333333',          # Dark gray
}

# Font settings
FONT_SIZES = {
    'title': 18,
    'component': 11,
    'dimension': 9,
    'gflops': 9,
    'annotation': 8,
    'legend': 10,
}

# Network architecture specifications
NETWORK_SPECS = {
    'input': {
        'batch_size': 16,
        'temporal': 16,
        'channels': 3,
        'height': 224,
        'width': 224,
    },
    'output': {
        'batch_size': 16,
        'classes': 83,
    },
    'gflops': {
        'rgb_resnet34': 3.6,
        'depth_resnet34': 3.6,
        'tsm': 0.0,
        'temporal_fusion': 1.2,
        'transformer': 4.8,
        'pooling': 0.5,
        'classifier': 0.3,
        'total': 14.0,
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_component_box(ax, x, y, width, height, label, color, 
                         dimensions='', gflops='', fontsize_label=11, 
                         fontsize_info=9, alpha=0.8):
    """
    Create a colored box representing a network component.
    
    Args:
        ax: Matplotlib axis
        x, y: Position coordinates
        width, height: Box dimensions
        label: Main label text
        color: Box color
        dimensions: Tensor dimensions text
        gflops: GFLOPs information text
        fontsize_label: Font size for main label
        fontsize_info: Font size for dimensions and GFLOPs
        alpha: Box transparency
    
    Returns:
        FancyBboxPatch object
    """
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='black',
        linewidth=2,
        alpha=alpha
    )
    ax.add_patch(box)
    
    # Add label
    ax.text(x + width/2, y + height/2 + 0.15, label,
            ha='center', va='center', fontsize=fontsize_label,
            weight='bold', wrap=True)
    
    # Add dimensions if provided
    if dimensions:
        ax.text(x + width/2, y + height/2 - 0.1, dimensions,
                ha='center', va='center', fontsize=fontsize_info,
                style='italic', color='#333333')
    
    # Add GFLOPs if provided
    if gflops:
        ax.text(x + width/2, y + height/2 - 0.3, gflops,
                ha='center', va='center', fontsize=fontsize_info,
                weight='bold', color='#CC0000')
    
    return box

def create_arrow(ax, x1, y1, x2, y2, label='', color='#333333', 
                 linewidth=2, arrowstyle='->', fontsize=8):
    """
    Create an arrow connecting two components.
    
    Args:
        ax: Matplotlib axis
        x1, y1: Start coordinates
        x2, y2: End coordinates
        label: Optional label for the arrow
        color: Arrow color
        linewidth: Arrow line width
        arrowstyle: Arrow style
        fontsize: Font size for label
    
    Returns:
        FancyArrowPatch object
    """
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=arrowstyle,
        color=color,
        linewidth=linewidth,
        mutation_scale=20,
        zorder=1
    )
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label,
                ha='center', va='bottom', fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='none', alpha=0.8))
    
    return arrow

def format_tensor_shape(b, t=None, c=None, h=None, w=None):
    """
    Format tensor shape for display.
    
    Args:
        b: Batch size (required)
        t: Temporal dimension (optional)
        c: Channels (optional)
        h: Height (optional)
        w: Width (optional)
    
    Returns:
        str: Formatted tensor shape string, e.g., "(16, 16, 3, 224, 224)"
             or "(16, 512)" for 2D tensors
    """
    # Build the shape string with only non-None values
    dims = [str(b)]
    for dim in [t, c, h, w]:
        if dim is not None and dim != '':
            dims.append(str(dim))
    
    return f"({', '.join(dims)})"

def format_gflops(value):
    """Format GFLOPs value for display."""
    return f"{value:.1f} GFLOPs"

# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def generate_tsmtfn_flowchart():
    """
    Generate the complete TSMTFN architecture flowchart.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), 
                           facecolor=BACKGROUND_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    # Title
    ax.text(5, 12.5, 'TSMTFN: Two-Stream Multi-scale Temporal Fusion Network',
            ha='center', va='center', fontsize=FONT_SIZES['title'],
            weight='bold')
    
    # Subtitle with total GFLOPs
    total_gflops = NETWORK_SPECS['gflops']['total']
    ax.text(5, 12.0, f'Total Computational Cost: {format_gflops(total_gflops)}',
            ha='center', va='center', fontsize=FONT_SIZES['annotation'],
            style='italic', color='#CC0000', weight='bold')
    
    # ========================================================================
    # INPUT STAGE
    # ========================================================================
    
    # RGB Input
    specs = NETWORK_SPECS['input']
    rgb_input_dim = format_tensor_shape(specs['batch_size'], specs['temporal'],
                                        specs['channels'], specs['height'], 
                                        specs['width'])
    create_component_box(ax, 1.5, 10.5, 2.5, 0.8,
                        'RGB Input', COLORS['input'],
                        dimensions=rgb_input_dim,
                        fontsize_label=FONT_SIZES['component'],
                        fontsize_info=FONT_SIZES['dimension'])
    
    # Depth Input
    depth_input_dim = format_tensor_shape(specs['batch_size'], specs['temporal'],
                                          specs['channels'], specs['height'], 
                                          specs['width'])
    create_component_box(ax, 6, 10.5, 2.5, 0.8,
                        'Depth Input', COLORS['input'],
                        dimensions=depth_input_dim,
                        fontsize_label=FONT_SIZES['component'],
                        fontsize_info=FONT_SIZES['dimension'])
    
    # ========================================================================
    # RGB STREAM
    # ========================================================================
    
    # RGB ResNet-34 + TSM
    rgb_resnet_dim = format_tensor_shape(16, 16, 512, 7, 7)
    rgb_gflops = format_gflops(NETWORK_SPECS['gflops']['rgb_resnet34'])
    create_component_box(ax, 1.2, 9.0, 3.1, 1.0,
                        'ResNet-34 + TSM\n(RGB Stream)', 
                        COLORS['rgb_stream'],
                        dimensions=rgb_resnet_dim,
                        gflops=rgb_gflops,
                        fontsize_label=FONT_SIZES['component'],
                        fontsize_info=FONT_SIZES['gflops'])
    
    # Arrow from RGB Input to RGB ResNet
    create_arrow(ax, 2.75, 10.5, 2.75, 10.0)
    
    # ========================================================================
    # DEPTH STREAM
    # ========================================================================
    
    # Depth ResNet-34 + TSM
    depth_resnet_dim = format_tensor_shape(16, 16, 512, 7, 7)
    depth_gflops = format_gflops(NETWORK_SPECS['gflops']['depth_resnet34'])
    create_component_box(ax, 5.7, 9.0, 3.1, 1.0,
                        'ResNet-34 + TSM\n(Depth Stream)', 
                        COLORS['depth_stream'],
                        dimensions=depth_resnet_dim,
                        gflops=depth_gflops,
                        fontsize_label=FONT_SIZES['component'],
                        fontsize_info=FONT_SIZES['gflops'])
    
    # Arrow from Depth Input to Depth ResNet
    create_arrow(ax, 7.25, 10.5, 7.25, 10.0)
    
    # ========================================================================
    # TEMPORAL FUSION
    # ========================================================================
    
    # TemporalDW1D Fusion
    fusion_dim = format_tensor_shape(16, 16, 512, 7, 7)
    fusion_gflops = format_gflops(NETWORK_SPECS['gflops']['temporal_fusion'])
    create_component_box(ax, 3.5, 7.3, 3, 1.0,
                        'TemporalDW1D Fusion', 
                        COLORS['fusion'],
                        dimensions=fusion_dim,
                        gflops=fusion_gflops,
                        fontsize_label=FONT_SIZES['component'],
                        fontsize_info=FONT_SIZES['gflops'])
    
    # Arrows from both streams to fusion
    create_arrow(ax, 2.75, 9.0, 4.2, 8.3, label='Concatenate')
    create_arrow(ax, 7.25, 9.0, 5.8, 8.3, label='Concatenate')
    
    # ========================================================================
    # TRANSFORMER BLOCKS
    # ========================================================================
    
    # Transformer Block 1
    trans1_dim = format_tensor_shape(16, 16, 512, 7, 7)
    trans1_gflops = format_gflops(NETWORK_SPECS['gflops']['transformer'] / 2)
    create_component_box(ax, 3.5, 5.8, 3, 0.9,
                        'Transformer Block 1\n(Multi-Head Attention)', 
                        COLORS['transformer'],
                        dimensions=trans1_dim,
                        gflops=trans1_gflops,
                        fontsize_label=FONT_SIZES['component'],
                        fontsize_info=FONT_SIZES['gflops'])
    
    # Arrow from fusion to transformer 1
    create_arrow(ax, 5, 7.3, 5, 6.7)
    
    # Transformer Block 2
    trans2_dim = format_tensor_shape(16, 16, 512, 7, 7)
    trans2_gflops = format_gflops(NETWORK_SPECS['gflops']['transformer'] / 2)
    create_component_box(ax, 3.5, 4.4, 3, 0.9,
                        'Transformer Block 2\n(Multi-Head Attention)', 
                        COLORS['transformer'],
                        dimensions=trans2_dim,
                        gflops=trans2_gflops,
                        fontsize_label=FONT_SIZES['component'],
                        fontsize_info=FONT_SIZES['gflops'])
    
    # Arrow from transformer 1 to transformer 2
    create_arrow(ax, 5, 5.8, 5, 5.33)
    
    # ========================================================================
    # MULTI-SCALE TEMPORAL POOLING
    # ========================================================================
    
    pooling_dim = format_tensor_shape(16, 1, 512, 1, 1)
    pooling_gflops = format_gflops(NETWORK_SPECS['gflops']['pooling'])
    create_component_box(ax, 3.5, 3.0, 3, 0.9,
                        'Multi-Scale Temporal\nPooling', 
                        COLORS['pooling'],
                        dimensions=pooling_dim,
                        gflops=pooling_gflops,
                        fontsize_label=FONT_SIZES['component'],
                        fontsize_info=FONT_SIZES['gflops'])
    
    # Arrow from transformer 2 to pooling
    create_arrow(ax, 5, 4.4, 5, 3.9)
    
    # Annotation for pooling scales
    ax.text(7.0, 3.45, 'Scales: [1, 2, 4]\nAdaptive Pooling',
            ha='left', va='center', fontsize=FONT_SIZES['annotation'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFCC', 
                     edgecolor='black', linewidth=1))
    
    # ========================================================================
    # CLASSIFICATION HEAD
    # ========================================================================
    
    classifier_dim = format_tensor_shape(16, 512)
    classifier_gflops = format_gflops(NETWORK_SPECS['gflops']['classifier'])
    create_component_box(ax, 3.5, 1.5, 3, 0.9,
                        'Classification Head\n(FC + Dropout)', 
                        COLORS['classifier'],
                        dimensions=classifier_dim,
                        gflops=classifier_gflops,
                        fontsize_label=FONT_SIZES['component'],
                        fontsize_info=FONT_SIZES['gflops'])
    
    # Arrow from pooling to classifier
    create_arrow(ax, 5, 3.0, 5, 2.4)
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    output_specs = NETWORK_SPECS['output']
    output_dim = f"({output_specs['batch_size']}, {output_specs['classes']})"
    create_component_box(ax, 3.75, 0.2, 2.5, 0.8,
                        'Output\n(83 Gesture Classes)', 
                        COLORS['output'],
                        dimensions=output_dim,
                        fontsize_label=FONT_SIZES['component'],
                        fontsize_info=FONT_SIZES['dimension'])
    
    # Arrow from classifier to output
    create_arrow(ax, 5, 1.5, 5, 1.0)
    
    # ========================================================================
    # ANNOTATIONS AND LEGEND
    # ========================================================================
    
    # Add annotation for TSM
    ax.text(0.3, 9.5, 'TSM: Temporal Shift Module\n(0 additional params)',
            ha='left', va='center', fontsize=FONT_SIZES['annotation'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4F8', 
                     edgecolor='black', linewidth=1))
    
    # Add feature dimension change annotations
    ax.text(0.3, 7.8, 'Feature Evolution:\n224×224 → 7×7\nTemporal: 16 frames',
            ha='left', va='center', fontsize=FONT_SIZES['annotation'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF4E5', 
                     edgecolor='black', linewidth=1))
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['input'], edgecolor='black', 
                      label='Input Layer'),
        mpatches.Patch(facecolor=COLORS['rgb_stream'], edgecolor='black', 
                      label='RGB Stream'),
        mpatches.Patch(facecolor=COLORS['depth_stream'], edgecolor='black', 
                      label='Depth Stream'),
        mpatches.Patch(facecolor=COLORS['fusion'], edgecolor='black', 
                      label='Fusion Module'),
        mpatches.Patch(facecolor=COLORS['transformer'], edgecolor='black', 
                      label='Transformer'),
        mpatches.Patch(facecolor=COLORS['pooling'], edgecolor='black', 
                      label='Pooling'),
        mpatches.Patch(facecolor=COLORS['classifier'], edgecolor='black', 
                      label='Classifier'),
        mpatches.Patch(facecolor=COLORS['output'], edgecolor='black', 
                      label='Output'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', 
             fontsize=FONT_SIZES['legend'], framealpha=0.9,
             title='Component Types', title_fontsize=FONT_SIZES['legend'])
    
    # Add computation summary box
    summary_text = (
        'Computational Breakdown:\n'
        f'RGB ResNet-34: {NETWORK_SPECS["gflops"]["rgb_resnet34"]:.1f} GFLOPs\n'
        f'Depth ResNet-34: {NETWORK_SPECS["gflops"]["depth_resnet34"]:.1f} GFLOPs\n'
        f'Temporal Fusion: {NETWORK_SPECS["gflops"]["temporal_fusion"]:.1f} GFLOPs\n'
        f'Transformers (×2): {NETWORK_SPECS["gflops"]["transformer"]:.1f} GFLOPs\n'
        f'Multi-Scale Pooling: {NETWORK_SPECS["gflops"]["pooling"]:.1f} GFLOPs\n'
        f'Classification: {NETWORK_SPECS["gflops"]["classifier"]:.1f} GFLOPs\n'
        f'─────────────────────\n'
        f'Total: {NETWORK_SPECS["gflops"]["total"]:.1f} GFLOPs'
    )
    
    ax.text(9.7, 1.5, summary_text,
            ha='right', va='bottom', fontsize=FONT_SIZES['annotation'],
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', 
                     edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    
    return fig, ax

# ============================================================================
# SAVE FUNCTIONS
# ============================================================================

def save_flowchart(fig, output_dir, filename, formats, dpi):
    """
    Save the flowchart in multiple formats.
    
    Args:
        fig: Matplotlib figure
        output_dir: Output directory path
        filename: Base filename (without extension)
        formats: List of format extensions
        dpi: Resolution in dots per inch
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for fmt in formats:
        output_path = output_dir / f"{filename}.{fmt}"
        
        # Save with appropriate settings for each format
        if fmt == 'png':
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        elif fmt == 'pdf':
            fig.savefig(output_path, format='pdf', bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        elif fmt == 'svg':
            fig.savefig(output_path, format='svg', bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        saved_files.append(output_path)
        print(f"✓ Saved: {output_path}")
    
    return saved_files

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("TSMTFN Network Architecture Flowchart Generator")
    print("=" * 70)
    print()
    
    print("Configuration:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Base filename: {OUTPUT_FILENAME}")
    print(f"  DPI: {DPI}")
    print(f"  Formats: {', '.join(FORMATS)}")
    print(f"  Figure size: {FIGURE_WIDTH}×{FIGURE_HEIGHT} inches")
    print()
    
    print("Generating flowchart...")
    fig, ax = generate_tsmtfn_flowchart()
    print("✓ Flowchart generated successfully")
    print()
    
    print("Saving flowchart in multiple formats...")
    saved_files = save_flowchart(fig, OUTPUT_DIR, OUTPUT_FILENAME, 
                                 FORMATS, DPI)
    print()
    
    print("=" * 70)
    print(f"Successfully generated {len(saved_files)} file(s):")
    for file_path in saved_files:
        print(f"  • {file_path}")
    print("=" * 70)
    
    # Close the figure to free memory
    plt.close(fig)

if __name__ == "__main__":
    main()
