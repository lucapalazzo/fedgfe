"""
Test to verify that node split is applied BEFORE train/val/test split.
This ensures that num_samples_per_node is taken from the full dataset,
then train/val/test ratios are applied to that subset.
"""

import sys
sys.path.insert(0, '/home/lpala/fedgfe')

from system.datautils.dataset_vegas import VEGASDataset

def test_split_order():
    """
    Test that verifies:
    1. Node split is applied first on full dataset
    2. Train/val/test split is applied on node's subset
    3. Ratios are correctly applied to the node's data
    """

    print("=" * 80)
    print("Testing VEGAS Dataset Split Order")
    print("=" * 80)

    # Configuration
    selected_classes = ['dog', 'baby_cry']
    samples_per_node = 100  # Each node gets 100 samples per class
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    node_split_seed = 42

    print(f"\nConfiguration:")
    print(f"  Classes: {selected_classes}")
    print(f"  Samples per node: {samples_per_node} per class")
    print(f"  Train ratio: {train_ratio} (70%)")
    print(f"  Val ratio: {val_ratio} (10%)")
    print(f"  Test ratio: {test_ratio} (20%)")
    print(f"  Node split seed: {node_split_seed}")

    # Test Node 0
    print(f"\n{'-' * 80}")
    print(f"Node 0:")
    print(f"{'-' * 80}")

    # Create dataset with split=None to get all three splits
    dataset_node0 = VEGASDataset(
        selected_classes=selected_classes,
        split=None,  # Creates .train, .val, .test
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        node_id=0,
        samples_per_node=samples_per_node,
        node_split_seed=node_split_seed,
        enable_cache=False,
        enable_ast_cache=False,  # Disable AST cache to avoid permission issues
        load_audio=False,
        load_image=False
    )

    # Check train split
    train_samples = len(dataset_node0.train)
    val_samples = len(dataset_node0.val)
    test_samples = len(dataset_node0.test)
    total_samples = train_samples + val_samples + test_samples

    print(f"\nSamples per split:")
    print(f"  Train: {train_samples} ({train_samples/total_samples*100:.1f}%)")
    print(f"  Val:   {val_samples} ({val_samples/total_samples*100:.1f}%)")
    print(f"  Test:  {test_samples} ({test_samples/total_samples*100:.1f}%)")
    print(f"  Total: {total_samples}")

    # Expected values
    # Each class should have samples_per_node samples after node split
    # Then these are split 70/10/20
    expected_per_class = samples_per_node
    expected_total = expected_per_class * len(selected_classes)
    expected_train = int(expected_total * train_ratio)
    expected_val = int(expected_total * val_ratio)
    expected_test = int(expected_total * test_ratio)

    print(f"\nExpected samples:")
    print(f"  Total per node: {expected_total} ({expected_per_class} per class × {len(selected_classes)} classes)")
    print(f"  Expected train: ~{expected_train} (70% of {expected_total})")
    print(f"  Expected val:   ~{expected_val} (10% of {expected_total})")
    print(f"  Expected test:  ~{expected_test} (20% of {expected_total})")

    # Check per-class distribution
    print(f"\nPer-class distribution (train split):")
    train_class_counts = dataset_node0.train.get_samples_per_class()
    for class_name, count in train_class_counts.items():
        expected_train_per_class = int(samples_per_node * train_ratio)
        print(f"  {class_name}: {count} samples (expected ~{expected_train_per_class})")

    print(f"\nPer-class distribution (val split):")
    val_class_counts = dataset_node0.val.get_samples_per_class()
    for class_name, count in val_class_counts.items():
        expected_val_per_class = int(samples_per_node * val_ratio)
        print(f"  {class_name}: {count} samples (expected ~{expected_val_per_class})")

    print(f"\nPer-class distribution (test split):")
    test_class_counts = dataset_node0.test.get_samples_per_class()
    for class_name, count in test_class_counts.items():
        expected_test_per_class = int(samples_per_node * test_ratio)
        print(f"  {class_name}: {count} samples (expected ~{expected_test_per_class})")

    # Verify correctness
    print(f"\n{'-' * 80}")
    print("Verification:")
    print(f"{'-' * 80}")

    # Check if total is approximately correct
    tolerance = 5  # Allow 5 samples tolerance
    if abs(total_samples - expected_total) <= tolerance:
        print(f"✓ Total samples correct: {total_samples} ≈ {expected_total}")
    else:
        print(f"✗ Total samples incorrect: {total_samples} ≠ {expected_total}")
        return False

    # Check if ratios are approximately correct
    train_ratio_actual = train_samples / total_samples
    val_ratio_actual = val_samples / total_samples
    test_ratio_actual = test_samples / total_samples

    ratio_tolerance = 0.05  # 5% tolerance

    if abs(train_ratio_actual - train_ratio) <= ratio_tolerance:
        print(f"✓ Train ratio correct: {train_ratio_actual:.2%} ≈ {train_ratio:.2%}")
    else:
        print(f"✗ Train ratio incorrect: {train_ratio_actual:.2%} ≠ {train_ratio:.2%}")
        return False

    if abs(val_ratio_actual - val_ratio) <= ratio_tolerance:
        print(f"✓ Val ratio correct: {val_ratio_actual:.2%} ≈ {val_ratio:.2%}")
    else:
        print(f"✗ Val ratio incorrect: {val_ratio_actual:.2%} ≠ {val_ratio:.2%}")
        return False

    if abs(test_ratio_actual - test_ratio) <= ratio_tolerance:
        print(f"✓ Test ratio correct: {test_ratio_actual:.2%} ≈ {test_ratio:.2%}")
    else:
        print(f"✗ Test ratio incorrect: {test_ratio_actual:.2%} ≠ {test_ratio:.2%}")
        return False

    print(f"\n✓ All checks passed!")
    return True


if __name__ == "__main__":
    success = test_split_order()
    if success:
        print(f"\n{'=' * 80}")
        print("SUCCESS: Split order is correct!")
        print("Node split is applied BEFORE train/val/test split.")
        print(f"{'=' * 80}")
    else:
        print(f"\n{'=' * 80}")
        print("FAILED: Split order verification failed!")
        print(f"{'=' * 80}")
        sys.exit(1)
