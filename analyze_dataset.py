#!/usr/bin/env python
import sys
sys.path.append('/home/lpala/fedgfe/system')
import numpy as np
import os

def analyze_dataset_structure(dataset_path):
    """Analyze the structure of a split dataset."""
    print(f"\n=== Analyzing {dataset_path} ===\n")

    # Check directory structure
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return

    # List directories
    dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

    print(f"Directories: {dirs}")
    print(f"Files: {files}")

    # Check train and test directories
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            npz_files = [f for f in os.listdir(split_path) if f.endswith('.npz')]
            print(f"\n{split} directory: {len(npz_files)} npz files")

            if npz_files:
                # Analyze first file
                first_file = os.path.join(split_path, npz_files[0])
                data = np.load(first_file, allow_pickle=True)

                print(f"  Sample file: {npz_files[0]}")
                print(f"  Keys in npz: {list(data.keys())}")

                for key in data.keys():
                    item = data[key]
                    if hasattr(item, 'shape'):
                        print(f"    {key}: shape={item.shape}, dtype={item.dtype}")
                    else:
                        print(f"    {key}: type={type(item)}")

                # If data is wrapped in 'data' key with object dtype, unwrap it
                if 'data' in data and data['data'].dtype == object:
                    actual_data = data['data'].item()
                    print(f"  Unwrapped data keys: {list(actual_data.keys()) if isinstance(actual_data, dict) else 'Not a dict'}")

                    if isinstance(actual_data, dict):
                        for k, v in actual_data.items():
                            if hasattr(v, 'shape'):
                                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                            else:
                                print(f"    {k}: type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")

                        # Use the unwrapped data for further analysis
                        data_to_analyze = actual_data
                    else:
                        data_to_analyze = data
                else:
                    data_to_analyze = data

                # Check if it has standard keys
                if 'samples' in data_to_analyze and 'labels' in data_to_analyze:
                    print(f"\n  Standard format detected (samples + labels)")
                    samples = data_to_analyze['samples']
                    labels = data_to_analyze['labels']

                    print(f"  Number of samples: {len(samples)}")
                    if len(samples) > 0:
                        print(f"  First sample shape: {samples[0].shape if hasattr(samples[0], 'shape') else 'N/A'}")

                    # Check label structure
                    if hasattr(labels, 'shape'):
                        print(f"  Labels shape: {labels.shape}")
                        if len(labels.shape) > 1:
                            print(f"  Multi-task detected: {labels.shape[1]} tasks")

                    # Check for segmentation masks
                    if 'masks' in data_to_analyze:
                        masks = data_to_analyze['masks']
                        print(f"  Masks shape: {masks.shape}")
                        print(f"  Segmentation dataset detected")
                    if 'semantic_masks' in data_to_analyze:
                        masks = data_to_analyze['semantic_masks']
                        print(f"  Semantic masks shape: {masks.shape}")
                        print(f"  Segmentation dataset detected")

                data.close()

# Analyze multiple datasets
datasets = [
    "/home/lpala/fedgfe/dataset/JSRT-8C-ClaSSeg",
    "/home/lpala/fedgfe/dataset/JSRT-4C-ClaSSeg",
    "/home/lpala/fedgfe/dataset/JSRT-1C-ClaSSeg",
    "/home/lpala/fedgfe/dataset/CIFAR10",
]

for dataset_path in datasets:
    if os.path.exists(dataset_path):
        analyze_dataset_structure(dataset_path)
    else:
        print(f"\nDataset {dataset_path} not found")

print("\n=== Analysis Complete ===")