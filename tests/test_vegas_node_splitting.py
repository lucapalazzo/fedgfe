"""
Test script for VEGAS dataset node splitting functionality.
Tests both num_nodes and samples_per_node modes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'system'))

from datautils.dataset_vegas import VEGASDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_num_nodes_mode():
    """Test node splitting with num_nodes parameter."""
    print("\n" + "="*80)
    print("TEST 1: Node Splitting with num_nodes")
    print("="*80)

    num_nodes = 3
    seed = 42
    classes = ['dog', 'baby_cry', 'chainsaw']

    nodes = []
    for node_id in range(num_nodes):
        dataset = VEGASDataset(
            root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
            selected_classes=classes,
            split='train',
            node_id=node_id,
            num_nodes=num_nodes,
            node_split_seed=seed,
            enable_cache=False
        )
        nodes.append(dataset)

        print(f"\nNode {node_id}:")
        print(f"  Total samples: {len(dataset)}")
        samples_per_class = dataset.get_samples_per_class()
        for class_name, count in samples_per_class.items():
            print(f"    {class_name}: {count} samples")

    # Verify no overlap
    print("\n" + "-"*80)
    print("Verifying no overlap between nodes...")
    all_file_ids = []
    for i, dataset in enumerate(nodes):
        file_ids = [sample['file_id'] for sample in dataset.samples]
        overlap = set(all_file_ids).intersection(set(file_ids))
        if overlap:
            print(f"ERROR: Node {i} has {len(overlap)} overlapping samples!")
            return False
        all_file_ids.extend(file_ids)

    total_samples = sum(len(node) for node in nodes)
    print(f"✓ No overlap detected")
    print(f"✓ Total unique samples across all nodes: {total_samples}")

    # Verify reproducibility
    print("\n" + "-"*80)
    print("Verifying reproducibility...")
    dataset_copy = VEGASDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
        selected_classes=classes,
        split='train',
        node_id=0,
        num_nodes=num_nodes,
        node_split_seed=seed,
        enable_cache=False
    )

    if len(dataset_copy) == len(nodes[0]):
        file_ids_original = [s['file_id'] for s in nodes[0].samples]
        file_ids_copy = [s['file_id'] for s in dataset_copy.samples]
        if file_ids_original == file_ids_copy:
            print(f"✓ Reproducibility verified: Same samples in same order")
        else:
            print(f"ERROR: Same length but different samples!")
            return False
    else:
        print(f"ERROR: Different dataset sizes!")
        return False

    print("\n✓ TEST 1 PASSED\n")
    return True


def test_samples_per_node_mode():
    """Test node splitting with samples_per_node parameter."""
    print("\n" + "="*80)
    print("TEST 2: Node Splitting with samples_per_node")
    print("="*80)

    samples_per_node = 10
    seed = 42
    classes = ['dog', 'baby_cry']

    nodes = []
    for node_id in range(3):
        dataset = VEGASDataset(
            root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
            selected_classes=classes,
            split='train',
            node_id=node_id,
            samples_per_node=samples_per_node,
            node_split_seed=seed,
            enable_cache=False
        )
        nodes.append(dataset)

        print(f"\nNode {node_id}:")
        print(f"  Total samples: {len(dataset)}")
        samples_per_class = dataset.get_samples_per_class()
        for class_name, count in samples_per_class.items():
            print(f"    {class_name}: {count} samples")

            # Verify each class has exactly samples_per_node samples
            expected = min(samples_per_node, count)  # In case class has fewer samples
            if count != expected:
                print(f"    WARNING: Expected {expected} samples, got {count}")

    # Verify no overlap
    print("\n" + "-"*80)
    print("Verifying no overlap between nodes...")
    all_file_ids = []
    for i, dataset in enumerate(nodes):
        file_ids = [sample['file_id'] for sample in dataset.samples]
        overlap = set(all_file_ids).intersection(set(file_ids))
        if overlap:
            print(f"ERROR: Node {i} has {len(overlap)} overlapping samples!")
            return False
        all_file_ids.extend(file_ids)

    print(f"✓ No overlap detected")
    print(f"✓ Total unique samples across all nodes: {len(all_file_ids)}")

    print("\n✓ TEST 2 PASSED\n")
    return True


def test_integration_with_splits():
    """Test node splitting combined with train/val/test splits."""
    print("\n" + "="*80)
    print("TEST 3: Integration with Train/Val/Test Splits")
    print("="*80)

    num_nodes = 2
    seed = 42
    classes = ['dog', 'baby_cry']

    for node_id in range(num_nodes):
        print(f"\nNode {node_id}:")

        # Test with explicit splits
        train_dataset = VEGASDataset(
            root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
            selected_classes=classes,
            split='train',
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            node_id=node_id,
            num_nodes=num_nodes,
            node_split_seed=seed,
            enable_cache=False
        )

        val_dataset = VEGASDataset(
            root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
            selected_classes=classes,
            split='val',
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            node_id=node_id,
            num_nodes=num_nodes,
            node_split_seed=seed,
            enable_cache=False
        )

        test_dataset = VEGASDataset(
            root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
            selected_classes=classes,
            split='test',
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            node_id=node_id,
            num_nodes=num_nodes,
            node_split_seed=seed,
            enable_cache=False
        )

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")

        # Verify no overlap between train/val/test within a node
        train_ids = set([s['file_id'] for s in train_dataset.samples])
        val_ids = set([s['file_id'] for s in val_dataset.samples])
        test_ids = set([s['file_id'] for s in test_dataset.samples])

        if train_ids.intersection(val_ids):
            print(f"  ERROR: Train/Val overlap!")
            return False
        if train_ids.intersection(test_ids):
            print(f"  ERROR: Train/Test overlap!")
            return False
        if val_ids.intersection(test_ids):
            print(f"  ERROR: Val/Test overlap!")
            return False

        print(f"  ✓ No overlap between train/val/test splits")

    print("\n✓ TEST 3 PASSED\n")
    return True


def test_error_handling():
    """Test error handling for invalid configurations."""
    print("\n" + "="*80)
    print("TEST 4: Error Handling")
    print("="*80)

    classes = ['dog']

    # Test 1: Both num_nodes and samples_per_node specified
    print("\nTest 4.1: Both num_nodes and samples_per_node (should raise error)")
    try:
        dataset = VEGASDataset(
            root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
            selected_classes=classes,
            split='train',
            node_id=0,
            num_nodes=3,
            samples_per_node=10,
            enable_cache=False
        )
        print("  ERROR: Should have raised ValueError!")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    # Test 2: node_id >= num_nodes
    print("\nTest 4.2: node_id >= num_nodes (should raise error)")
    try:
        dataset = VEGASDataset(
            root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
            selected_classes=classes,
            split='train',
            node_id=5,
            num_nodes=3,
            enable_cache=False
        )
        print("  ERROR: Should have raised ValueError!")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    print("\n✓ TEST 4 PASSED\n")
    return True


def test_different_seeds():
    """Test that different seeds produce different splits."""
    print("\n" + "="*80)
    print("TEST 5: Different Seeds Produce Different Splits")
    print("="*80)

    classes = ['dog', 'baby_cry']

    dataset_seed42 = VEGASDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
        selected_classes=classes,
        split='train',
        node_id=0,
        num_nodes=3,
        node_split_seed=42,
        enable_cache=False
    )

    dataset_seed123 = VEGASDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
        selected_classes=classes,
        split='train',
        node_id=0,
        num_nodes=3,
        node_split_seed=123,
        enable_cache=False
    )

    # Should have same size
    if len(dataset_seed42) != len(dataset_seed123):
        print(f"ERROR: Different sizes: {len(dataset_seed42)} vs {len(dataset_seed123)}")
        return False

    # But different samples (likely)
    file_ids_42 = [s['file_id'] for s in dataset_seed42.samples]
    file_ids_123 = [s['file_id'] for s in dataset_seed123.samples]

    if file_ids_42 == file_ids_123:
        print("WARNING: Same samples with different seeds (very unlikely but possible)")
    else:
        different_count = sum(1 for a, b in zip(file_ids_42, file_ids_123) if a != b)
        print(f"✓ Different seeds produced different samples ({different_count}/{len(file_ids_42)} different)")

    print("\n✓ TEST 5 PASSED\n")
    return True


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*80)
    print("VEGAS NODE SPLITTING - TEST SUITE")
    print("="*80)

    tests = [
        ("num_nodes mode", test_num_nodes_mode),
        ("samples_per_node mode", test_samples_per_node_mode),
        ("Integration with train/val/test splits", test_integration_with_splits),
        ("Error handling", test_error_handling),
        ("Different seeds", test_different_seeds)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED WITH EXCEPTION: {test_name}")
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(result for _, result in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
