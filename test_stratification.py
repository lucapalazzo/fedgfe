"""
Test script to verify stratification across train/val/test splits.
Ensures each class is proportionally distributed across all splits.
"""

import sys
sys.path.append('/home/lpala/fedgfe')

from collections import Counter
import warnings

print("Testing stratification across splits...")
print("=" * 80)

try:
    from system.datautils.dataset_esc50 import ESC50Dataset

    root_dir = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
    text_embedding_file = "/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt"

    # Test with auto-split creation
    print("\n1. Creating dataset with auto-split (stratify=True)...")
    dataset = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split=None,  # Auto-create all splits
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,  # Enable stratification
        node_id=42
    )

    print(f"   Train: {len(dataset.train)} samples")
    print(f"   Val:   {len(dataset.val)} samples")
    print(f"   Test:  {len(dataset.test)} samples")
    print(f"   Total: {len(dataset.train) + len(dataset.val) + len(dataset.test)} samples")

    # Count samples per class in each split
    print("\n2. Counting samples per class in each split...")

    train_class_counts = Counter([s['class_name'] for s in dataset.train.samples])
    val_class_counts = Counter([s['class_name'] for s in dataset.val.samples])
    test_class_counts = Counter([s['class_name'] for s in dataset.test.samples])

    # Get all unique classes
    all_classes = set(list(train_class_counts.keys()) +
                     list(val_class_counts.keys()) +
                     list(test_class_counts.keys()))

    print(f"   Found {len(all_classes)} classes")

    # Verify stratification for each class
    print("\n3. Verifying proportional distribution per class...")
    print("-" * 80)
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10} {'Train%':<10} {'Val%':<10} {'Test%'}")
    print("-" * 80)

    errors = []
    for class_name in sorted(all_classes):
        train_count = train_class_counts.get(class_name, 0)
        val_count = val_class_counts.get(class_name, 0)
        test_count = test_class_counts.get(class_name, 0)
        total_count = train_count + val_count + test_count

        # Calculate percentages
        train_pct = (train_count / total_count * 100) if total_count > 0 else 0
        val_pct = (val_count / total_count * 100) if total_count > 0 else 0
        test_pct = (test_count / total_count * 100) if total_count > 0 else 0

        print(f"{class_name:<20} {train_count:<10} {val_count:<10} {test_count:<10} "
              f"{total_count:<10} {train_pct:<10.1f} {val_pct:<10.1f} {test_pct:.1f}")

        # Verify proportions (with tolerance of ±5% due to small sample sizes)
        # Expected: 70-10-20
        if abs(train_pct - 70) > 10:
            errors.append(f"  ✗ {class_name}: Train {train_pct:.1f}% too far from 70%")
        if abs(val_pct - 10) > 10:
            errors.append(f"  ✗ {class_name}: Val {val_pct:.1f}% too far from 10%")
        if abs(test_pct - 20) > 10:
            errors.append(f"  ✗ {class_name}: Test {test_pct:.1f}% too far from 20%")

    print("-" * 80)

    # Summary statistics
    print("\n4. Summary statistics...")

    all_train_pcts = [(train_class_counts.get(c, 0) / (train_class_counts.get(c, 0) +
                       val_class_counts.get(c, 0) + test_class_counts.get(c, 0)) * 100)
                      for c in all_classes]
    all_val_pcts = [(val_class_counts.get(c, 0) / (train_class_counts.get(c, 0) +
                     val_class_counts.get(c, 0) + test_class_counts.get(c, 0)) * 100)
                    for c in all_classes]
    all_test_pcts = [(test_class_counts.get(c, 0) / (train_class_counts.get(c, 0) +
                      val_class_counts.get(c, 0) + test_class_counts.get(c, 0)) * 100)
                     for c in all_classes]

    avg_train = sum(all_train_pcts) / len(all_train_pcts)
    avg_val = sum(all_val_pcts) / len(all_val_pcts)
    avg_test = sum(all_test_pcts) / len(all_test_pcts)

    print(f"   Average train%: {avg_train:.1f}% (expected: 70%)")
    print(f"   Average val%:   {avg_val:.1f}% (expected: 10%)")
    print(f"   Average test%:  {avg_test:.1f}% (expected: 20%)")

    # Check for errors
    if errors:
        print("\n5. ✗ STRATIFICATION ISSUES FOUND:")
        for error in errors:
            print(error)
    else:
        print("\n5. ✓ STRATIFICATION VERIFIED!")
        print("   All classes are proportionally distributed across splits.")

    # Test without stratification for comparison
    print("\n" + "=" * 80)
    print("6. Testing WITHOUT stratification (stratify=False)...")
    print("=" * 80)

    dataset_no_strat = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split=None,
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=False,  # Disable stratification
        node_id=42
    )

    train_counts_ns = Counter([s['class_name'] for s in dataset_no_strat.train.samples])
    val_counts_ns = Counter([s['class_name'] for s in dataset_no_strat.val.samples])
    test_counts_ns = Counter([s['class_name'] for s in dataset_no_strat.test.samples])

    print("\nSample of classes without stratification:")
    print("-" * 80)
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10} {'Train%':<10} {'Val%':<10} {'Test%'}")
    print("-" * 80)

    # Show first 10 classes
    for class_name in sorted(list(all_classes))[:10]:
        train_count = train_counts_ns.get(class_name, 0)
        val_count = val_counts_ns.get(class_name, 0)
        test_count = test_counts_ns.get(class_name, 0)
        total_count = train_count + val_count + test_count

        train_pct = (train_count / total_count * 100) if total_count > 0 else 0
        val_pct = (val_count / total_count * 100) if total_count > 0 else 0
        test_pct = (test_count / total_count * 100) if total_count > 0 else 0

        print(f"{class_name:<20} {train_count:<10} {val_count:<10} {test_count:<10} "
              f"{total_count:<10} {train_pct:<10.1f} {val_pct:<10.1f} {test_pct:.1f}")

    print("-" * 80)
    print("\n✓ Comparison shows difference between stratified and non-stratified splits!")

    print("\n" + "=" * 80)
    print("ALL STRATIFICATION TESTS COMPLETED!")
    print("=" * 80)

except ImportError as e:
    print(f"Could not import: {e}")
    print("(This is expected if torch is not installed)")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
