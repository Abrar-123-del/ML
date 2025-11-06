#!/usr/bin/env python3
"""
Example: Quick Demo of Ablation Study (CPU-only, small scale)

This script demonstrates the ablation study framework with a minimal example
that runs quickly on CPU for testing purposes.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_study import AblationStudy
from ablation_study_project.configs import ABLATION_CONFIGS


def run_quick_demo():
    """Run a quick demo with reduced epochs"""
    print("="*80)
    print("QUICK ABLATION STUDY DEMO")
    print("="*80)
    print("\nThis is a minimal demo that:")
    print("- Uses CPU (no GPU required)")
    print("- Runs with reduced epochs (2 epochs per variant)")
    print("- Uses synthetic data (no real dataset required)")
    print("- Demonstrates all 6 model configurations")
    print("\n" + "="*80 + "\n")
    
    # Temporarily reduce epochs for quick demo
    import ablation_study_project.configs.ablation_config as config_module
    original_epochs = config_module.EPOCHS
    config_module.EPOCHS = 2  # Just 2 epochs for demo
    
    # Create ablation study with demo output directory
    study = AblationStudy(
        output_root='./ablation_demo_results',
        data_root='/tmp/demo_data',  # Will use synthetic data
        resume=False
    )
    
    try:
        # Run the study
        study.run()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nResults saved to: ./ablation_demo_results/")
        print("\nGenerated files:")
        print("- ablation_comparison_table.csv")
        print("- ablation_comparison_table.json")
        print("- ablation_comparison_report.txt")
        print("- component_contribution_analysis.txt")
        print("- accuracy_comparison_plot.png")
        print("- component_ablation_visualization.png")
        print("- learning_curves_all_variants.png")
        print("- ablation_summary_report.pdf")
        print("\nPlus individual results for each of the 6 variants.")
        
    except Exception as e:
        print(f"\nDemo failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original epochs
        config_module.EPOCHS = original_epochs


def show_configurations():
    """Display all ablation configurations"""
    print("="*80)
    print("ABLATION STUDY CONFIGURATIONS")
    print("="*80)
    
    for i, (variant_id, config) in enumerate(ABLATION_CONFIGS.items(), 1):
        print(f"\n{i}. {config['name']}")
        print(f"   ID: {variant_id}")
        print(f"   Description: {config['description']}")
        print(f"   Configuration:")
        print(f"     - RGB: {config['use_rgb']}")
        print(f"     - Depth: {config['use_depth']}")
        print(f"     - Transformer: {config['use_transformer']}")
        print(f"     - Pooling: {config['pooling_type']}")
        print(f"     - Augmentation: {config['use_augmentation']}")
        print(f"     - Progressive Unfreezing: {config['use_progressive_unfreezing']}")
        print(f"   Expected Accuracy: {config['expected_acc']:.2f}%")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Quick demo of ablation study framework'
    )
    parser.add_argument(
        '--show-configs',
        action='store_true',
        help='Show all configurations and exit'
    )
    parser.add_argument(
        '--run-demo',
        action='store_true',
        help='Run quick demo (2 epochs per variant)'
    )
    
    args = parser.parse_args()
    
    if args.show_configs:
        show_configurations()
    elif args.run_demo:
        run_quick_demo()
    else:
        print("Ablation Study Demo")
        print("\nUsage:")
        print("  python example_ablation_demo.py --show-configs    # Show all configurations")
        print("  python example_ablation_demo.py --run-demo        # Run quick demo")
        print("\nFor full ablation study, use:")
        print("  python ablation_study.py")
