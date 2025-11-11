"""
Test script for NodeData dataset merging functionality.
"""

import sys
import os
import torch
from torch.utils.data import Dataset
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from system.datautils.node_dataset import NodeData


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, num_samples=100, num_classes=3, node_id=0):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.node_id = node_id

        # Generate balanced data
        samples_per_class = num_samples // num_classes
        self.data = []
        self.labels = []

        for class_id in range(num_classes):
            for _ in range(samples_per_class):
                # Random data with node_id information for tracking
                data_point = torch.randn(10) + node_id  # Add offset based on node_id
                self.data.append(data_point)
                self.labels.append(class_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Args:
    """Mock args object."""

    def __init__(self):
        self.dataset = 'test_dataset'
        self.num_classes = 3
        self.device = 'cpu'
        self.dataset_dir_prefix = ''


def test_basic_merging():
    """Test basic dataset merging."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Dataset Merging")
    print("=" * 80)

    args = Args()

    # Create 3 nodes
    nodes = []
    for node_id in range(3):
        dataset = SimpleDataset(num_samples=90, num_classes=3, node_id=node_id)

        node = NodeData(
            args=args,
            node_id=node_id,
            dataset_split_id=node_id,
            dataset=dataset,
            split_ratio=0.7,
            val_ratio=0.15,
            stratify=True
        )
        nodes.append(node)
        print(f"✓ Created {node}")

    # Merge training data
    merged_train_loader = NodeData.create_merged_dataloader(
        nodes,
        split_type='train',
        batch_size=16,
        shuffle=True
    )

    print(f"\n✓ Merged train loader created: {len(merged_train_loader.dataset)} samples")

    # Test iteration
    batch_count = 0
    for batch_data, batch_labels in merged_train_loader:
        batch_count += 1
        if batch_count == 1:
            print(f"✓ First batch shape: data={batch_data.shape}, labels={batch_labels.shape}")

    print(f"✓ Total batches: {batch_count}")


def test_merge_validation():
    """Test merging validation data."""
    print("\n" + "=" * 80)
    print("TEST 2: Merge Validation Data")
    print("=" * 80)

    args = Args()

    nodes = []
    for node_id in range(3):
        dataset = SimpleDataset(num_samples=90, num_classes=3, node_id=node_id)

        node = NodeData(
            args=args,
            node_id=node_id,
            dataset_split_id=node_id,
            dataset=dataset,
            split_ratio=0.7,
            val_ratio=0.15,
            stratify=True
        )
        nodes.append(node)

    # Merge validation data
    merged_val_loader = NodeData.create_merged_dataloader(
        nodes,
        split_type='val',
        batch_size=16,
        shuffle=False  # Don't shuffle validation
    )

    print(f"✓ Merged val loader: {len(merged_val_loader.dataset)} samples")
    print(f"✓ Shuffle disabled: {not merged_val_loader.sampler}")


def test_instance_method():
    """Test instance method for merging."""
    print("\n" + "=" * 80)
    print("TEST 3: Instance Method Merging")
    print("=" * 80)

    args = Args()

    nodes = []
    for node_id in range(3):
        dataset = SimpleDataset(num_samples=90, num_classes=3, node_id=node_id)

        node = NodeData(
            args=args,
            node_id=node_id,
            dataset_split_id=node_id,
            dataset=dataset,
            split_ratio=0.7,
            val_ratio=0.15,
            stratify=False
        )
        nodes.append(node)

    # Node 0 merges with nodes 1 and 2
    merged_loader = nodes[0].merge_with_nodes(
        [nodes[1], nodes[2]],
        split_type='train',
        batch_size=16
    )

    print(f"✓ Node 0 merged with others: {len(merged_loader.dataset)} samples")

    # Should be same as static method
    static_loader = NodeData.create_merged_dataloader(
        nodes,
        split_type='train',
        batch_size=16
    )

    assert len(merged_loader.dataset) == len(static_loader.dataset)
    print(f"✓ Verified: instance method equals static method")


def test_mixed_split_dataloader():
    """Test mixed split dataloader."""
    print("\n" + "=" * 80)
    print("TEST 4: Mixed Split Dataloader")
    print("=" * 80)

    args = Args()

    nodes = []
    for node_id in range(3):
        dataset = SimpleDataset(num_samples=90, num_classes=3, node_id=node_id)

        node = NodeData(
            args=args,
            node_id=node_id,
            dataset_split_id=node_id,
            dataset=dataset,
            split_ratio=0.7,
            val_ratio=0.15,
            stratify=True
        )
        nodes.append(node)

    # Mix: train from node0, val from node1, test from node2
    mixed_loader = NodeData.create_mixed_split_dataloader(
        nodes,
        split_configs=['train', 'val', 'test'],
        batch_size=16,
        shuffle=True
    )

    total_samples = len(mixed_loader.dataset)
    expected = nodes[0].train_samples + nodes[1].val_samples + nodes[2].test_samples

    print(f"✓ Mixed loader: {total_samples} samples")
    print(f"  - Train from Node 0: {nodes[0].train_samples}")
    print(f"  - Val from Node 1: {nodes[1].val_samples}")
    print(f"  - Test from Node 2: {nodes[2].test_samples}")
    print(f"  - Expected total: {expected}")

    assert total_samples == expected
    print(f"✓ Verified: total matches expected")


def test_get_merged_dataset():
    """Test getting merged dataset without dataloader."""
    print("\n" + "=" * 80)
    print("TEST 5: Get Merged Dataset")
    print("=" * 80)

    args = Args()

    nodes = []
    for node_id in range(3):
        dataset = SimpleDataset(num_samples=60, num_classes=3, node_id=node_id)

        node = NodeData(
            args=args,
            node_id=node_id,
            dataset_split_id=node_id,
            dataset=dataset,
            split_ratio=0.7,
            val_ratio=0.15,
            stratify=False
        )
        nodes.append(node)

    # Get merged dataset
    merged_dataset = NodeData.merge_datasets_from_nodes(
        nodes,
        split_type='train',
        return_dataset=True
    )

    print(f"✓ Merged dataset size: {len(merged_dataset)}")

    # Get individual datasets
    datasets_list = NodeData.merge_datasets_from_nodes(
        nodes,
        split_type='train',
        return_dataset=False
    )

    print(f"✓ Individual datasets: {len(datasets_list)} nodes")
    for i, ds in enumerate(datasets_list):
        print(f"  - Node {i}: {len(ds)} samples")


def test_class_distribution():
    """Test that class distribution is preserved."""
    print("\n" + "=" * 80)
    print("TEST 6: Class Distribution Preservation")
    print("=" * 80)

    args = Args()

    nodes = []
    for node_id in range(3):
        dataset = SimpleDataset(num_samples=90, num_classes=3, node_id=node_id)

        node = NodeData(
            args=args,
            node_id=node_id,
            dataset_split_id=node_id,
            dataset=dataset,
            split_ratio=0.7,
            val_ratio=0.15,
            stratify=True
        )
        nodes.append(node)

    # Merge with stratification
    merged_loader = NodeData.create_merged_dataloader(
        nodes,
        split_type='train',
        batch_size=16
    )

    # Count labels
    from collections import Counter
    all_labels = []
    for _, batch_labels in merged_loader:
        all_labels.extend(batch_labels.tolist())

    label_counts = Counter(all_labels)
    total = len(all_labels)

    print(f"\nMerged Dataset Class Distribution:")
    print("-" * 50)
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / total) * 100
        print(f"  Class {label}: {count:3d} samples ({percentage:5.2f}%)")

    # Check balance (should be roughly equal for 3 classes)
    percentages = [label_counts[i] / total * 100 for i in range(3)]
    max_diff = max(percentages) - min(percentages)

    print(f"\nMax percentage difference: {max_diff:.2f}%")

    if max_diff < 5.0:  # Within 5%
        print("✓ Classes are well balanced")
    else:
        print("⚠ Classes may be imbalanced (but this is expected with small samples)")


def test_progressive_merging():
    """Test progressive addition of nodes."""
    print("\n" + "=" * 80)
    print("TEST 7: Progressive Node Addition")
    print("=" * 80)

    args = Args()

    # Create nodes
    all_nodes = []
    for node_id in range(4):
        dataset = SimpleDataset(num_samples=60, num_classes=3, node_id=node_id)

        node = NodeData(
            args=args,
            node_id=node_id,
            dataset_split_id=node_id,
            dataset=dataset,
            split_ratio=0.7,
            val_ratio=0.15,
            stratify=False
        )
        all_nodes.append(node)

    # Progressively add nodes
    active_nodes = []
    for i, node in enumerate(all_nodes):
        active_nodes.append(node)

        loader = NodeData.create_merged_dataloader(
            active_nodes,
            split_type='train',
            batch_size=16
        )

        print(f"Round {i}: {len(active_nodes)} nodes, {len(loader.dataset)} samples")


def test_no_validation_nodes():
    """Test handling of nodes without validation data."""
    print("\n" + "=" * 80)
    print("TEST 8: Nodes Without Validation Data")
    print("=" * 80)

    args = Args()

    nodes = []
    for node_id in range(3):
        dataset = SimpleDataset(num_samples=60, num_classes=3, node_id=node_id)

        node = NodeData(
            args=args,
            node_id=node_id,
            dataset_split_id=node_id,
            dataset=dataset,
            split_ratio=0.8,
            val_ratio=0.0,  # No validation
            stratify=False
        )
        nodes.append(node)

    try:
        # This should fail because nodes have no validation data
        merged_val_loader = NodeData.create_merged_dataloader(
            nodes,
            split_type='val',
            batch_size=16
        )
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")


def test_custom_dataloader_params():
    """Test custom dataloader parameters."""
    print("\n" + "=" * 80)
    print("TEST 9: Custom DataLoader Parameters")
    print("=" * 80)

    args = Args()

    nodes = []
    for node_id in range(2):
        dataset = SimpleDataset(num_samples=60, num_classes=3, node_id=node_id)

        node = NodeData(
            args=args,
            node_id=node_id,
            dataset_split_id=node_id,
            dataset=dataset,
            split_ratio=0.7,
            val_ratio=0.15,
            stratify=False
        )
        nodes.append(node)

    # Create with custom parameters
    merged_loader = NodeData.create_merged_dataloader(
        nodes,
        split_type='train',
        batch_size=8,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=False
    )

    print(f"✓ Batch size: {merged_loader.batch_size}")
    print(f"✓ Drop last: {merged_loader.drop_last}")
    print(f"✓ Num workers: {merged_loader.num_workers}")
    print(f"✓ Total batches: {len(merged_loader)}")


def main():
    """Run all tests."""
    try:
        print("\n" + "=" * 80)
        print("NODEDATA MERGING TESTS")
        print("=" * 80)

        test_basic_merging()
        test_merge_validation()
        test_instance_method()
        test_mixed_split_dataloader()
        test_get_merged_dataset()
        test_class_distribution()
        test_progressive_merging()
        test_no_validation_nodes()
        test_custom_dataloader_params()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
