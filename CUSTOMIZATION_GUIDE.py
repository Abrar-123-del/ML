#!/usr/bin/env python3
"""
Example: Customizing TSMTFN Flowchart Generation

This script demonstrates how to customize various aspects of the
flowchart generation by modifying the parameters.
"""

# Example 1: Changing output directory
# Open generate_tsmtfn_flowchart.py and modify:
# OUTPUT_DIR = Path("Results")
# to:
# OUTPUT_DIR = Path("custom_output")

# Example 2: Changing DPI for higher/lower resolution
# DPI = 300  # Default
# DPI = 150  # Lower resolution, smaller file size
# DPI = 600  # Higher resolution for print

# Example 3: Changing output formats
# FORMATS = ['png', 'pdf', 'svg']  # Default
# FORMATS = ['png']  # Only PNG
# FORMATS = ['pdf', 'svg']  # Only vector formats

# Example 4: Customizing colors
# COLORS = {
#     'input': '#YOUR_COLOR_HERE',
#     'rgb_stream': '#YOUR_COLOR_HERE',
#     # ... etc
# }

# Example 5: Adjusting figure size
# FIGURE_WIDTH = 20  # Default width in inches
# FIGURE_HEIGHT = 14  # Default height in inches
# 
# FIGURE_WIDTH = 16  # Smaller width
# FIGURE_HEIGHT = 12  # Smaller height

# Example 6: Modifying network specifications
# Update the NETWORK_SPECS dictionary to match your architecture:
# NETWORK_SPECS = {
#     'input': {
#         'batch_size': 32,  # Changed from 16
#         'temporal': 8,     # Changed from 16
#         ...
#     },
#     'gflops': {
#         'rgb_resnet34': 3.6,
#         ...
#     }
# }

print("""
TSMTFN Flowchart Customization Guide
=====================================

The generate_tsmtfn_flowchart.py script can be customized by modifying
the following sections:

1. OUTPUT SETTINGS (Lines ~30-35)
   - Output directory
   - Filename
   - DPI (resolution)
   - Output formats

2. FIGURE SETTINGS (Lines ~37-39)
   - Figure width and height
   - Background color

3. COLOR SCHEME (Lines ~41-52)
   - Colors for each component type
   - Arrow colors

4. FONT SETTINGS (Lines ~54-61)
   - Font sizes for different elements
   - Title, labels, annotations

5. NETWORK SPECIFICATIONS (Lines ~63-91)
   - Input/output dimensions
   - GFLOPs for each component
   - Batch size, temporal frames, etc.

To customize:
1. Open generate_tsmtfn_flowchart.py in a text editor
2. Locate the configuration section
3. Modify the values as needed
4. Save and run: python3 generate_tsmtfn_flowchart.py

For more details, see README_TSMTFN.md
""")
