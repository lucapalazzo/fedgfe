#!/usr/bin/env python
import sys
sys.path.append('/home/lpala/fedgfe/system')

import os
import torch
from datautils.fl_splitted_dataset import FLSplittedDataset
from datautils.node_dataset import NodeData

class MockArgs:
    def __init__(self):
        self.dataset = "JSRT-8C-ClaSSeg"
        self.dataset_dir_prefix = "/home/lpala/fedgfe"
        self.device = "cpu"
        self.num_classes = 8

def test_fl_splitted_dataset():
    """Test FLSplittedDataset directly."""
    print("=== Testing FLSplittedDataset directly ===")

    dataset_path = "/home/lpala/fedgfe/dataset/JSRT-8C-ClaSSeg"
    node_id = 0

    try:
        # Test train dataset
        train_dataset = FLSplittedDataset(
            dataset_path=dataset_path,
            node_id=node_id,
            is_train=True
        )

        print(f"Train dataset loaded successfully!")
        print(f"Dataset info: {train_dataset.get_data_info()}")

        # Test sample access
        if len(train_dataset) > 0:
            sample, label = train_dataset[0]
            print(f"First sample shape: {sample.shape}")
            print(f"Label keys: {list(label.keys()) if isinstance(label, dict) else type(label)}")

        # Test test dataset
        test_dataset = FLSplittedDataset(
            dataset_path=dataset_path,
            node_id=node_id,
            is_train=False
        )

        print(f"Test dataset loaded successfully!")
        print(f"Test dataset length: {len(test_dataset)}")

        return True

    except Exception as e:
        print(f"Error testing FLSplittedDataset: {e}")
        return False

def test_node_data_integration():
    """Test NodeData integration with FLSplittedDataset."""
    print("\n=== Testing NodeData integration ===")

    args = MockArgs()

    try:
        # Create NodeData instance
        node_data = NodeData(
            args=args,
            node_id=0,
            dataset_split_id=0
        )

        print(f"NodeData created: {node_data}")
        print(f"Using FL splitted: {node_data.use_fl_splitted}")

        # Test loading train data
        train_loader = node_data.load_train_data(batch_size=2)
        if train_loader is not None:
            print(f"Train dataloader created successfully!")
            print(f"Train samples: {node_data.train_samples}")

            # Test one batch
            for batch_idx, (samples, labels) in enumerate(train_loader):
                print(f"Batch {batch_idx}: samples shape {samples.shape}")
                print(f"Labels type: {type(labels)}")
                if isinstance(labels, dict):
                    for key, value in labels.items():
                        print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
                break

        # Test loading test data
        test_loader = node_data.load_test_data(batch_size=2)
        if test_loader is not None:
            print(f"Test dataloader created successfully!")
            print(f"Test samples: {node_data.test_samples}")

        return True

    except Exception as e:
        print(f"Error testing NodeData integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_other_datasets():
    """Test with other dataset formats."""
    print("\n=== Testing other dataset formats ===")

    datasets_to_test = [
        "/home/lpala/fedgfe/dataset/CIFAR10",
        "/home/lpala/fedgfe/dataset/JSRT-4C-ClaSSeg",
        "/home/lpala/fedgfe/dataset/JSRT-1C-ClaSSeg"
    ]

    for dataset_path in datasets_to_test:
        if os.path.exists(dataset_path):
            try:
                print(f"\nTesting {os.path.basename(dataset_path)}:")
                train_dataset = FLSplittedDataset(
                    dataset_path=dataset_path,
                    node_id=0,
                    is_train=True
                )

                info = train_dataset.get_data_info()
                print(f"  Length: {info['length']}")
                print(f"  Format: {info['format']}")
                print(f"  Keys: {info['data_keys']}")
                if info['length'] > 0:
                    sample, label = train_dataset[0]
                    print(f"  Sample shape: {sample.shape}")

            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"\nDataset {dataset_path} not found")

if __name__ == "__main__":
    print("Testing FLSplittedDataset integration...")

    # Test 1: Direct FLSplittedDataset usage
    success1 = test_fl_splitted_dataset()

    # Test 2: NodeData integration
    success2 = test_node_data_integration()

    # Test 3: Other datasets
    test_other_datasets()

    print(f"\n=== Test Results ===")
    print(f"FLSplittedDataset direct: {'✓' if success1 else '✗'}")
    print(f"NodeData integration: {'✓' if success2 else '✗'}")

    if success1 and success2:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. Check errors above.")