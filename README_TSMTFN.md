# TSMTFN Network Architecture Flowchart Generator

## Overview

This script generates a comprehensive visualization of the **Two-Stream Multi-scale Temporal Fusion Network (TSMTFN)** architecture, designed for RGB-D gesture recognition tasks.

## Features

- **Two-Stream Architecture**: Clear visualization of RGB and Depth processing streams
- **Tensor Dimensions**: Shows tensor shapes (B, T, C, H, W) at each processing stage
- **GFLOPs Breakdown**: Computational cost for each major component
- **Key Modules Labeled**:
  - ResNet-34 + TSM (Temporal Shift Module) for both streams
  - TemporalDW1D Fusion
  - Transformer blocks (×2)
  - Multi-Scale Temporal Pooling
  - Classification Head
- **Professional Styling**: Publication-ready graphics with color-coded components
- **High Resolution**: 300 DPI output suitable for academic papers
- **Multiple Formats**: PNG, PDF, and SVG outputs

## Requirements

```bash
pip install matplotlib numpy
```

## Usage

### Basic Usage

```bash
python3 generate_tsmtfn_flowchart.py
```

This will generate three files in the `Results/` directory:
- `TSMTFN_Architecture.png` (high-resolution raster image)
- `TSMTFN_Architecture.pdf` (vector format for LaTeX documents)
- `TSMTFN_Architecture.svg` (scalable vector graphics)

### Customization

The script includes customizable parameters at the top of the file:

```python
# Output settings
OUTPUT_DIR = Path("Results")
OUTPUT_FILENAME = "TSMTFN_Architecture"
DPI = 300
FORMATS = ['png', 'pdf', 'svg']

# Figure settings
FIGURE_WIDTH = 20
FIGURE_HEIGHT = 14

# Color scheme
COLORS = {
    'input': '#E8F4F8',
    'rgb_stream': '#FFE5E5',
    'depth_stream': '#E5F5E5',
    # ... customize colors as needed
}

# Font settings
FONT_SIZES = {
    'title': 18,
    'component': 11,
    'dimension': 9,
    # ... customize font sizes as needed
}
```

## Network Architecture Details

### Input Specifications
- **RGB Input**: (B=16, T=16, 3, 224, 224)
- **Depth Input**: (B=16, T=16, 3, 224, 224)
- B: Batch size
- T: Temporal frames
- C: Channels
- H×W: Spatial dimensions

### Computational Cost (GFLOPs)
| Component | GFLOPs |
|-----------|--------|
| RGB ResNet-34 + TSM | 3.6 |
| Depth ResNet-34 + TSM | 3.6 |
| TemporalDW1D Fusion | 1.2 |
| Transformer Blocks (×2) | 4.8 |
| Multi-Scale Temporal Pooling | 0.5 |
| Classification Head | 0.3 |
| **Total** | **~14.0** |

### Output
- **Shape**: (B=16, 83)
- **Classes**: 83 gesture classes

## Architecture Flow

1. **Input Stage**: RGB and Depth streams receive video input (16 frames)
2. **Feature Extraction**: ResNet-34 with TSM extracts spatial-temporal features
3. **Fusion**: TemporalDW1D combines RGB and Depth features
4. **Attention**: Two transformer blocks capture long-range dependencies
5. **Pooling**: Multi-scale temporal pooling aggregates information
6. **Classification**: Fully-connected head predicts gesture class

## File Structure

```
.
├── generate_tsmtfn_flowchart.py    # Main script
├── README_TSMTFN.md                # This file
└── Results/                        # Output directory
    ├── TSMTFN_Architecture.png
    ├── TSMTFN_Architecture.pdf
    └── TSMTFN_Architecture.svg
```

## Color Coding

- **Light Blue**: Input layers
- **Light Red**: RGB stream components
- **Light Green**: Depth stream components
- **Light Orange**: Fusion modules
- **Light Purple**: Transformer blocks
- **Light Yellow**: Pooling layers
- **Light Pink**: Classification layers
- **Gray**: Output layer

## Notes

- The script is modular and easy to update with new components
- All dimensions and GFLOPs values can be customized in the `NETWORK_SPECS` dictionary
- The visualization is optimized for both digital display and print publication
- TSM (Temporal Shift Module) adds 0 additional parameters - it's a parameter-free operation

## Citation

If you use this visualization in your research, please cite appropriately and ensure compliance with your institution's guidelines for generated figures.

## License

This script is part of the ML repository and follows the same license terms.
