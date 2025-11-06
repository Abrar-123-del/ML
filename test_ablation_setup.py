#!/usr/bin/env python3
"""
Quick test script to verify ablation study setup

This script performs basic sanity checks on the ablation study components
without running the full training pipeline.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from ablation_study_project.models import TSMTwoStreamFusionNetwork, create_model
from ablation_study_project.datasets import EgoGestureFusionDataset
from ablation_study_project.configs import ABLATION_CONFIGS, DATASET_CONFIG


def test_model_creation():
    """Test creating models with different configurations"""
    print("="*80)
    print("Testing Model Creation")
    print("="*80)
    
    for variant_id, config in ABLATION_CONFIGS.items():
        print(f"\nTesting: {config['name']}")
        
        model_config = {
            'num_classes': DATASET_CONFIG['num_classes'],
            'n_segment': DATASET_CONFIG['n_segment'],
            'use_rgb': config['use_rgb'],
            'use_depth': config['use_depth'],
            'use_transformer': config['use_transformer'],
            'pooling_type': config['pooling_type'],
            'dropout_rate': 0.5
        }
        
        try:
            model = create_model(model_config)
            params = model.get_parameter_count()
            print(f"  ✓ Model created successfully")
            print(f"  Parameters: {params:,}")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            return False
    
    print("\n✓ All models created successfully!")
    return True


def test_forward_pass():
    """Test forward pass through different model configurations"""
    print("\n" + "="*80)
    print("Testing Forward Pass")
    print("="*80)
    
    batch_size = 2
    n_segment = 8
    
    # Create dummy inputs
    rgb_input = torch.randn(batch_size * n_segment, 3, 224, 224)
    depth_input = torch.randn(batch_size * n_segment, 1, 224, 224)
    
    # Test RGB-only model
    print("\nTesting RGB-only model...")
    model = create_model({
        'num_classes': 83,
        'n_segment': 8,
        'use_rgb': True,
        'use_depth': False,
        'use_transformer': False,
        'pooling_type': 'simple_avg'
    })
    
    try:
        output = model(rgb_input, None)
        assert output.shape == (batch_size, 83), f"Expected shape (2, 83), got {output.shape}"
        print(f"  ✓ RGB-only forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        return False
    
    # Test RGB-D model with transformer
    print("\nTesting RGB-D model with transformer...")
    model = create_model({
        'num_classes': 83,
        'n_segment': 8,
        'use_rgb': True,
        'use_depth': True,
        'use_transformer': True,
        'pooling_type': 'multi_scale'
    })
    
    try:
        output = model(rgb_input, depth_input)
        assert output.shape == (batch_size, 83), f"Expected shape (2, 83), got {output.shape}"
        print(f"  ✓ RGB-D transformer forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        return False
    
    print("\n✓ All forward passes successful!")
    return True


def test_dataset_creation():
    """Test dataset creation and data loading"""
    print("\n" + "="*80)
    print("Testing Dataset Creation")
    print("="*80)
    
    try:
        # Test RGB-only dataset
        print("\nTesting RGB-only dataset...")
        dataset = EgoGestureFusionDataset(
            data_root='/tmp/dummy_data',
            split='train',
            n_segment=8,
            use_rgb=True,
            use_depth=False,
            use_augmentation=True
        )
        print(f"  ✓ Dataset created. Size: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"  Sample keys: {sample.keys()}")
        if 'rgb' in sample:
            print(f"  RGB shape: {sample['rgb'].shape}")
        print(f"  Label: {sample['label']}")
        
        # Test RGB-D dataset
        print("\nTesting RGB-D dataset...")
        dataset = EgoGestureFusionDataset(
            data_root='/tmp/dummy_data',
            split='train',
            n_segment=8,
            use_rgb=True,
            use_depth=True,
            use_augmentation=True
        )
        print(f"  ✓ RGB-D dataset created. Size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"  Sample keys: {sample.keys()}")
        if 'rgb' in sample:
            print(f"  RGB shape: {sample['rgb'].shape}")
        if 'depth' in sample:
            print(f"  Depth shape: {sample['depth'].shape}")
        
        print("\n✓ Dataset creation successful!")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """Validate ablation study configurations"""
    print("\n" + "="*80)
    print("Testing Configuration Validation")
    print("="*80)
    
    required_keys = ['name', 'description', 'use_rgb', 'use_depth', 
                     'use_transformer', 'pooling_type', 'use_augmentation',
                     'use_progressive_unfreezing', 'expected_acc']
    
    for variant_id, config in ABLATION_CONFIGS.items():
        print(f"\nValidating: {variant_id}")
        
        # Check all required keys present
        for key in required_keys:
            if key not in config:
                print(f"  ✗ Missing key: {key}")
                return False
        
        # Validate values
        if not (config['use_rgb'] or config['use_depth']):
            print(f"  ✗ Must use at least RGB or depth!")
            return False
        
        if config['pooling_type'] not in ['simple_avg', 'multi_scale']:
            print(f"  ✗ Invalid pooling_type: {config['pooling_type']}")
            return False
        
        print(f"  ✓ Configuration valid")
    
    print(f"\n✓ All {len(ABLATION_CONFIGS)} configurations are valid!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("ABLATION STUDY SETUP VERIFICATION")
    print("="*80 + "\n")
    
    tests = [
        ("Configuration Validation", test_config_validation),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Dataset Creation", test_dataset_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED! The ablation study setup is ready.")
    else:
        print("✗ SOME TESTS FAILED. Please review the errors above.")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
