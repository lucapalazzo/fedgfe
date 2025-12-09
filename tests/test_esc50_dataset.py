#!/usr/bin/env python3
"""
Quick test script for ESC-50 dataset implementation.
"""

import sys
sys.path.insert(0, '/home/lpala/fedgfe/system')

from datautils.dataset_esc50 import ESC50Dataset
import json

print("=" * 70)
print("ESC-50 Dataset Test")
print("=" * 70)

# Test 1: Load class labels
print("\n1. Testing class labels loading...")
dataset_path = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
class_labels_path = f"{dataset_path}/class_labels.json"

with open(class_labels_path, 'r') as f:
    class_labels = json.load(f)

print(f"   Total classes: {len(class_labels)}")
print(f"   First 5 classes: {list(class_labels.keys())[:5]}")

# Test 2: Create dataset with specific classes
print("\n2. Creating dataset with 3 animal classes...")
try:
    dataset = ESC50Dataset(
        root_dir=dataset_path,
        selected_classes=['dog', 'cat', 'rooster'],
        split='train',
        use_folds=True,
        train_folds=[0, 1, 2],
        test_folds=[3, 4],
        enable_cache=False
    )

    print(f"   ✓ Dataset created successfully")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Active classes: {dataset.get_class_names()}")
    print(f"   Samples per class: {dataset.get_samples_per_class()}")

except Exception as e:
    print(f"   ✗ Error creating dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load a single sample
print("\n3. Testing sample loading...")
try:
    sample = dataset[0]
    print(f"   ✓ Sample loaded successfully")
    print(f"   Sample keys: {list(sample.keys())}")
    print(f"   Audio shape: {sample['audio'].shape}")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Label: {sample['label']} ({sample['class_name']})")
    print(f"   File ID: {sample['file_id']}")
    print(f"   Fold: {sample['fold']}")

except Exception as e:
    print(f"   ✗ Error loading sample: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test with all 50 classes
print("\n4. Testing with all 50 classes...")
try:
    full_dataset = ESC50Dataset(
        root_dir=dataset_path,
        split='all',
        use_folds=False,
        enable_cache=False
    )

    print(f"   ✓ Full dataset created")
    print(f"   Total samples: {len(full_dataset)}")
    print(f"   Total classes: {full_dataset.get_num_classes()}")

    samples_per_class = full_dataset.get_samples_per_class()
    print(f"   Min samples per class: {min(samples_per_class.values())}")
    print(f"   Max samples per class: {max(samples_per_class.values())}")

except Exception as e:
    print(f"   ✗ Error with full dataset: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test fold splitting
print("\n5. Testing fold-based splitting...")
try:
    train_dataset = ESC50Dataset(
        root_dir=dataset_path,
        selected_classes=['dog', 'cat'],
        split='train',
        use_folds=True,
        train_folds=[0, 1, 2, 3],
        test_folds=[4],
        enable_cache=False
    )

    test_dataset = ESC50Dataset(
        root_dir=dataset_path,
        selected_classes=['dog', 'cat'],
        split='test',
        use_folds=True,
        train_folds=[0, 1, 2, 3],
        test_folds=[4],
        enable_cache=False
    )

    print(f"   ✓ Fold splitting working")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Total: {len(train_dataset) + len(test_dataset)}")

except Exception as e:
    print(f"   ✗ Error with fold splitting: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test class exclusion
print("\n6. Testing class exclusion...")
try:
    excluded_dataset = ESC50Dataset(
        root_dir=dataset_path,
        selected_classes=['dog', 'cat', 'rooster', 'pig', 'cow'],
        excluded_classes=['pig', 'cow'],
        split='train',
        use_folds=True,
        train_folds=[0, 1, 2],
        enable_cache=False
    )

    print(f"   ✓ Class exclusion working")
    print(f"   Selected: ['dog', 'cat', 'rooster', 'pig', 'cow']")
    print(f"   Excluded: ['pig', 'cow']")
    print(f"   Active classes: {excluded_dataset.get_class_names()}")

except Exception as e:
    print(f"   ✗ Error with class exclusion: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("ESC-50 Dataset Test Completed!")
print("=" * 70)
