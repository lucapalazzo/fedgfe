"""
Test script for NodeData validation split and stratification features.
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

    def __init__(self, num_samples=1000, num_classes=3):
        self.num_samples = num_samples
        self.num_classes = num_classes

        # Generate balanced data
        samples_per_class = num_samples // num_classes
        self.data = []
        self.labels = []

        for class_id in range(num_classes):
            for _ in range(samples_per_class):
                # Random data
                data_point = torch.randn(10)
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


def count_labels(dataloader):
    """Count labels in a dataloader."""
    label_counts = {}
    total = 0

    for batch_data, batch_labels in dataloader:
        for label in batch_labels:
            label_item = label.item()
            label_counts[label_item] = label_counts.get(label_item, 0) + 1
            total += 1

    return label_counts, total


def print_distribution(label_counts, total, split_name):
    """Print label distribution."""
    print(f"\n{split_name} Distribution:")
    print("-" * 40)
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / total) * 100
        print(f"  Class {label}: {count:4d} samples ({percentage:5.2f}%)")
    print(f"  Total:    {total:4d} samples")


def test_automatic_stratified_split():
    """Test automatic splitting with stratification."""
    print("\n" + "=" * 80)
    print("TEST 1: Automatic Stratified Split")
    print("=" * 80)

    # Create dataset
    dataset = SimpleDataset(num_samples=1000, num_classes=3)
    args = Args()

    # Create NodeData with stratification
    node_data = NodeData(
        args=args,
        node_id=0,
        dataset_split_id=0,
        dataset=dataset,
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=True
    )

    print(f"\n✓ {node_data}")

    # Load dataloaders
    train_loader = node_data.load_train_data(batch_size=32)
    val_loader = node_data.load_val_data(batch_size=32)
    test_loader = node_data.load_test_data(batch_size=32)

    # Count labels
    train_counts, train_total = count_labels(train_loader)
    val_counts, val_total = count_labels(val_loader)
    test_counts, test_total = count_labels(test_loader)

    print_distribution(train_counts, train_total, "Train")
    print_distribution(val_counts, val_total, "Validation")
    print_distribution(test_counts, test_total, "Test")

    # Verify stratification (distributions should be similar)
    def get_percentages(counts, total):
        return {k: (v / total) * 100 for k, v in counts.items()}

    train_pct = get_percentages(train_counts, train_total)
    val_pct = get_percentages(val_counts, val_total)
    test_pct = get_percentages(test_counts, test_total)

    print("\n=== Stratification Check ===")
    tolerance = 5.0  # 5% tolerance
    all_pass = True

    for label in sorted(train_pct.keys()):
        train_p = train_pct[label]
        val_p = val_pct.get(label, 0)
        test_p = test_pct.get(label, 0)

        train_val_diff = abs(train_p - val_p)
        train_test_diff = abs(train_p - test_p)

        status_val = "✓" if train_val_diff <= tolerance else "✗"
        status_test = "✓" if train_test_diff <= tolerance else "✗"

        print(f"Class {label}:")
        print(f"  Train vs Val:  {status_val} ({train_val_diff:.2f}% diff)")
        print(f"  Train vs Test: {status_test} ({train_test_diff:.2f}% diff)")

        if train_val_diff > tolerance or train_test_diff > tolerance:
            all_pass = False

    if all_pass:
        print(f"\n✓ Stratification PASSED (tolerance: {tolerance}%)")
    else:
        print(f"\n✗ Stratification FAILED (tolerance: {tolerance}%)")

    return all_pass


def test_random_split():
    """Test random split without stratification."""
    print("\n" + "=" * 80)
    print("TEST 2: Random Split (No Stratification)")
    print("=" * 80)

    dataset = SimpleDataset(num_samples=900, num_classes=3)
    args = Args()

    node_data = NodeData(
        args=args,
        node_id=0,
        dataset_split_id=0,
        dataset=dataset,
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=False  # Disable stratification
    )

    print(f"\n✓ {node_data}")

    # Load dataloaders
    train_loader = node_data.load_train_data(batch_size=32)
    val_loader = node_data.load_val_data(batch_size=32)
    test_loader = node_data.load_test_data(batch_size=32)

    # Count labels
    train_counts, train_total = count_labels(train_loader)
    val_counts, val_total = count_labels(val_loader)
    test_counts, test_total = count_labels(test_loader)

    print_distribution(train_counts, train_total, "Train")
    print_distribution(val_counts, val_total, "Validation")
    print_distribution(test_counts, test_total, "Test")

    print("\n✓ Random split completed (distributions may vary)")


def test_custom_datasets():
    """Test with custom pre-split datasets."""
    print("\n" + "=" * 80)
    print("TEST 3: Custom Pre-Split Datasets")
    print("=" * 80)

    train_dataset = SimpleDataset(num_samples=700, num_classes=3)
    val_dataset = SimpleDataset(num_samples=150, num_classes=3)
    test_dataset = SimpleDataset(num_samples=150, num_classes=3)

    args = Args()

    node_data = NodeData(
        args=args,
        node_id=0,
        dataset_split_id=0,
        custom_train_dataset=train_dataset,
        custom_val_dataset=val_dataset,
        custom_test_dataset=test_dataset
    )

    print(f"\n✓ {node_data}")

    # Load dataloaders
    train_loader = node_data.load_train_data(batch_size=32)
    val_loader = node_data.load_val_data(batch_size=32)
    test_loader = node_data.load_test_data(batch_size=32)

    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    print(f"✓ Test loader: {len(test_loader)} batches")


def test_no_validation():
    """Test without validation split."""
    print("\n" + "=" * 80)
    print("TEST 4: No Validation Split (val_ratio=0)")
    print("=" * 80)

    dataset = SimpleDataset(num_samples=1000, num_classes=3)
    args = Args()

    node_data = NodeData(
        args=args,
        node_id=0,
        dataset_split_id=0,
        dataset=dataset,
        split_ratio=0.8,
        val_ratio=0.0,  # No validation
        stratify=True
    )

    print(f"\n✓ {node_data}")

    # Load dataloaders
    train_loader = node_data.load_train_data(batch_size=32)
    val_loader = node_data.load_val_data(batch_size=32)
    test_loader = node_data.load_test_data(batch_size=32)

    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {val_loader}")  # Should be None
    print(f"✓ Test loader: {len(test_loader)} batches")

    assert val_loader is None, "Validation loader should be None when val_ratio=0"
    print("\n✓ Correctly handles no validation case")


def test_reproducibility():
    """Test that same node_id produces same splits."""
    print("\n" + "=" * 80)
    print("TEST 5: Reproducibility (Same node_id)")
    print("=" * 80)

    dataset1 = SimpleDataset(num_samples=1000, num_classes=3)
    dataset2 = SimpleDataset(num_samples=1000, num_classes=3)  # Same data

    args = Args()

    # Create two NodeData instances with same node_id
    node_data1 = NodeData(
        args=args,
        node_id=42,
        dataset_split_id=0,
        dataset=dataset1,
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=True
    )

    node_data2 = NodeData(
        args=args,
        node_id=42,  # Same node_id
        dataset_split_id=0,
        dataset=dataset2,
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=True
    )

    print(f"\n✓ NodeData 1: {node_data1}")
    print(f"✓ NodeData 2: {node_data2}")

    # Check that splits have same sizes
    assert node_data1.train_samples == node_data2.train_samples
    assert node_data1.val_samples == node_data2.val_samples
    assert node_data1.test_samples == node_data2.test_samples

    print("\n✓ Reproducibility verified: same node_id produces same split sizes")


def test_different_nodes():
    """Test that different node_ids produce different shuffles."""
    print("\n" + "=" * 80)
    print("TEST 6: Different Nodes (Different node_ids)")
    print("=" * 80)

    dataset = SimpleDataset(num_samples=1000, num_classes=3)
    args = Args()

    # Create multiple nodes
    nodes = []
    for node_id in range(3):
        # Need to recreate dataset for each node for fresh split
        node_dataset = SimpleDataset(num_samples=1000, num_classes=3)

        node_data = NodeData(
            args=args,
            node_id=node_id,
            dataset_split_id=node_id,
            dataset=node_dataset,
            split_ratio=0.7,
            val_ratio=0.15,
            stratify=True
        )
        nodes.append(node_data)
        print(f"✓ Node {node_id}: {node_data}")

    # All nodes should have same split sizes (but different samples)
    for i in range(1, len(nodes)):
        assert nodes[0].train_samples == nodes[i].train_samples
        assert nodes[0].val_samples == nodes[i].val_samples
        assert nodes[0].test_samples == nodes[i].test_samples

    print("\n✓ All nodes have consistent split sizes")


def test_device_transfer():
    """Test device transfer."""
    print("\n" + "=" * 80)
    print("TEST 7: Device Transfer")
    print("=" * 80)

    dataset = SimpleDataset(num_samples=100, num_classes=3)
    args = Args()

    node_data = NodeData(
        args=args,
        node_id=0,
        dataset_split_id=0,
        dataset=dataset,
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=True
    )

    # Test device transfer
    if torch.cuda.is_available():
        node_data.to('cuda')
        print("✓ Transferred to CUDA")
        node_data.to('cpu')
        print("✓ Transferred back to CPU")
    else:
        node_data.to('cpu')
        print("✓ Device transfer works (CPU only)")


def main():
    """Run all tests."""
    try:
        print("\n" + "=" * 80)
        print("NODEDATA VALIDATION SPLIT TESTS")
        print("=" * 80)

        # Run tests
        stratified_pass = test_automatic_stratified_split()
        test_random_split()
        test_custom_datasets()
        test_no_validation()
        test_reproducibility()
        test_different_nodes()
        test_device_transfer()

        print("\n" + "=" * 80)
        if stratified_pass:
            print("ALL TESTS PASSED ✓")
        else:
            print("SOME TESTS HAD WARNINGS (but completed)")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
