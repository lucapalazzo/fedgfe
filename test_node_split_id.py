#!/usr/bin/env python3
"""
Test script to verify node_split_id functionality in VEGASDataset
"""

import sys
sys.path.insert(0, '/home/lpala/fedgfe/system')

from datautils.dataset_vegas import VEGASDataset

def test_node_split_id():
    """Test that different node_split_id values give different samples"""

    print("=" * 80)
    print("Testing VEGASDataset with node_split_id parameter")
    print("=" * 80)

    # Test configuration similar to the config file
    test_config = {
        'selected_classes': ['dog'],
        'samples_per_node': 200,
        'node_split_seed': 42,
        'train_ratio': 0.9,
        'val_ratio': 0.1,
        'test_ratio': 0.0,
        'split': 'train',
        'enable_ast_cache': False,
        'load_audio': False,
        'load_image': False,
        'load_video': False
    }

    # Create dataset with node_split_id=0
    print("\n--- Creating dataset with node_split_id=0 ---")
    dataset_0 = VEGASDataset(
        **test_config,
        node_split_id=0
    )

    # Create dataset with node_split_id=1
    print("\n--- Creating dataset with node_split_id=1 ---")
    dataset_1 = VEGASDataset(
        **test_config,
        node_split_id=1
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nDataset 0 (node_split_id=0):")
    print(f"  Total samples: {len(dataset_0)}")
    print(f"  Samples per class: {dataset_0.get_samples_per_class()}")

    print(f"\nDataset 1 (node_split_id=1):")
    print(f"  Total samples: {len(dataset_1)}")
    print(f"  Samples per class: {dataset_1.get_samples_per_class()}")

    # Get sample file IDs to verify they are different
    if len(dataset_0) > 0 and len(dataset_1) > 0:
        file_ids_0 = set([s['file_id'] for s in dataset_0.samples[:10]])
        file_ids_1 = set([s['file_id'] for s in dataset_1.samples[:10]])

        print(f"\nFirst 10 file IDs from dataset 0: {sorted(list(file_ids_0))}")
        print(f"First 10 file IDs from dataset 1: {sorted(list(file_ids_1))}")

        overlap = file_ids_0.intersection(file_ids_1)
        print(f"\nOverlapping file IDs (should be 0): {len(overlap)}")

        if len(overlap) == 0:
            print("\n✓ SUCCESS: The two datasets have different samples!")
        else:
            print("\n✗ FAILURE: The datasets share samples!")
            print(f"  Overlapping files: {overlap}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_node_split_id()
