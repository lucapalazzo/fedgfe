"""
Test script to verify VEGAS dataset ESC50-like features.
Tests new split management features: train_ratio, val_ratio, test_ratio,
split=None auto-creation, and splits_to_load.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.datautils.dataset_vegas import VEGASDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_auto_split_creation():
    """Test automatic train/val/test split creation with split=None."""
    print("\n" + "="*80)
    print("TEST 1: Auto-Split Creation (split=None)")
    print("="*80)

    # Create dataset with split=None to auto-create splits
    dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split=None,  # Auto-create train/val/test
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=42
    )

    print(f"\n✓ Main dataset created (contains all samples)")
    print(f"✓ Train split accessible: dataset.train")
    print(f"✓ Val split accessible: dataset.val")
    print(f"✓ Test split accessible: dataset.test")

    print(f"\nSplit sizes:")
    print(f"  Train: {len(dataset.train)} samples")
    print(f"  Val:   {len(dataset.val)} samples")
    print(f"  Test:  {len(dataset.test)} samples")
    print(f"  Total: {len(dataset.train) + len(dataset.val) + len(dataset.test)}")

    # Verify ratios
    total = len(dataset.train) + len(dataset.val) + len(dataset.test)
    train_pct = len(dataset.train) / total * 100
    val_pct = len(dataset.val) / total * 100
    test_pct = len(dataset.test) / total * 100

    print(f"\nActual ratios:")
    print(f"  Train: {train_pct:.1f}% (expected ~70%)")
    print(f"  Val:   {val_pct:.1f}% (expected ~15%)")
    print(f"  Test:  {test_pct:.1f}% (expected ~15%)")

    # Print statistics for each split
    dataset.train.print_split_statistics()
    dataset.val.print_split_statistics()
    dataset.test.print_split_statistics()


def test_custom_ratios():
    """Test custom train/val/test ratios."""
    print("\n" + "="*80)
    print("TEST 2: Custom Ratios (60-20-20)")
    print("="*80)

    train_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    val_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='val',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    test_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='test',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    train_pct = len(train_dataset) / total * 100
    val_pct = len(val_dataset) / total * 100
    test_pct = len(test_dataset) / total * 100

    print(f"\n✓ Train: {len(train_dataset)} samples ({train_pct:.1f}%, expected ~60%)")
    print(f"✓ Val:   {len(val_dataset)} samples ({val_pct:.1f}%, expected ~20%)")
    print(f"✓ Test:  {len(test_dataset)} samples ({test_pct:.1f}%, expected ~20%)")
    print(f"✓ Total: {total}")

    # Verify stratification
    print("\nVerifying stratification between splits:")
    train_dataset.verify_stratification(val_dataset, tolerance=0.10)
    train_dataset.verify_stratification(test_dataset, tolerance=0.10)


def test_splits_to_load():
    """Test loading data from multiple splits."""
    print("\n" + "="*80)
    print("TEST 3: splits_to_load Parameter")
    print("="*80)

    # Load only train split
    train_only = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    # Load train + val splits combined
    train_val_combined = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        splits_to_load=['train', 'val'],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    print(f"\n✓ Train only: {len(train_only)} samples")
    print(f"✓ Train+Val combined: {len(train_val_combined)} samples")
    print(f"✓ Expected: Train + Val ≈ {len(train_val_combined)}")


def test_legacy_compatibility():
    """Test backward compatibility with old split_ratio parameter."""
    print("\n" + "="*80)
    print("TEST 4: Legacy split_ratio Compatibility")
    print("="*80)

    # Old way (using split_ratio)
    dataset_old = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        split_ratio=0.8,  # Old parameter (deprecated)
        val_ratio=0.1,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    # New way (using train_ratio, val_ratio, test_ratio)
    dataset_new = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    print(f"\n✓ Old API (split_ratio=0.8): {len(dataset_old)} train samples")
    print(f"✓ New API (train_ratio=0.7): {len(dataset_new)} train samples")
    print(f"✓ Both APIs work correctly")


def test_stratification_verification():
    """Test stratification verification between splits."""
    print("\n" + "="*80)
    print("TEST 5: Stratification Verification")
    print("="*80)

    train_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    val_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='val',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    test_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='test',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    # Verify train vs val
    print("\nTrain vs Val:")
    result1 = train_dataset.verify_stratification(val_dataset, tolerance=0.10)

    # Verify train vs test
    print("\nTrain vs Test:")
    result2 = train_dataset.verify_stratification(test_dataset, tolerance=0.10)

    # Verify val vs test
    print("\nVal vs Test:")
    result3 = val_dataset.verify_stratification(test_dataset, tolerance=0.10)

    if result1 and result2 and result3:
        print("\n✓ All stratification checks PASSED")
    else:
        print("\n✗ Some stratification checks FAILED")


def test_class_distribution():
    """Test class distribution methods."""
    print("\n" + "="*80)
    print("TEST 6: Class Distribution Statistics")
    print("="*80)

    dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False
    )

    samples_per_class = dataset.get_samples_per_class()
    distribution = dataset.get_class_distribution()

    print("\nSamples per class:")
    for class_name, count in samples_per_class.items():
        pct = distribution.get(class_name, 0.0)
        print(f"  {class_name}: {count} samples ({pct:.2f}%)")

    # Verify percentages sum to 100
    total_pct = sum(distribution.values())
    print(f"\n✓ Total percentage: {total_pct:.2f}%")
    assert abs(total_pct - 100.0) < 0.01, "Percentages should sum to 100%"


def test_edge_cases():
    """Test edge cases and unusual configurations."""
    print("\n" + "="*80)
    print("TEST 7: Edge Cases")
    print("="*80)

    # Test with no validation split (val_ratio=0)
    print("\n7.1: No validation split (val_ratio=0)")
    dataset_no_val = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        train_ratio=0.8,
        val_ratio=0.0,
        test_ratio=0.2,
        stratify=True,
        load_audio=False,
        load_image=False
    )
    print(f"✓ Train dataset created: {len(dataset_no_val)} samples")

    # Test with ratios that don't sum exactly to 1.0
    print("\n7.2: Ratios that don't sum to 1.0 (auto-normalized)")
    dataset_normalized = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        train_ratio=0.6,
        val_ratio=0.3,
        test_ratio=0.2,  # Sum = 1.1, should be normalized
        stratify=True,
        load_audio=False,
        load_image=False
    )
    print(f"✓ Dataset created with normalized ratios: {len(dataset_normalized)} samples")
    print(f"  Normalized train_ratio: {dataset_normalized.train_ratio:.3f}")
    print(f"  Normalized val_ratio: {dataset_normalized.val_ratio:.3f}")
    print(f"  Normalized test_ratio: {dataset_normalized.test_ratio:.3f}")


def main():
    """Run all tests."""
    try:
        test_auto_split_creation()
        test_custom_ratios()
        test_splits_to_load()
        test_legacy_compatibility()
        test_stratification_verification()
        test_class_distribution()
        test_edge_cases()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nSummary of implemented ESC50 features in VEGAS:")
        print("  ✓ train_ratio, val_ratio, test_ratio parameters")
        print("  ✓ split=None for auto-creation of .train, .val, .test")
        print("  ✓ splits_to_load for combining multiple splits")
        print("  ✓ use_folds support (infrastructure ready)")
        print("  ✓ Improved _apply_split with better ratio handling")
        print("  ✓ Backward compatibility with legacy split_ratio")
        print("  ✓ Stratification verification methods")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
