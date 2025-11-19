"""
Test script for ESC50Dataset with custom split ratios and splits_to_load functionality.
"""

import sys
sys.path.append('/home/lpala/fedgfe')

from system.datautils.dataset_esc50 import ESC50Dataset

def test_custom_ratios():
    """Test custom train/val/test ratios."""
    print("\n" + "="*80)
    print("TEST 1: Custom Ratios (70-10-20)")
    print("="*80)

    root_dir = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
    text_embedding_file = "/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt"

    # Create datasets with 70-10-20 split
    train_ds = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='train',
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    val_ds = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='val',
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    test_ds = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='test',
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    total = len(train_ds) + len(val_ds) + len(test_ds)
    train_pct = (len(train_ds) / total) * 100
    val_pct = (len(val_ds) / total) * 100
    test_pct = (len(test_ds) / total) * 100

    print(f"Train: {len(train_ds)} samples ({train_pct:.1f}%)")
    print(f"Val:   {len(val_ds)} samples ({val_pct:.1f}%)")
    print(f"Test:  {len(test_ds)} samples ({test_pct:.1f}%)")
    print(f"Total: {total} samples")

    # Verify ratios are close to target
    assert abs(train_pct - 70) < 2, f"Train ratio {train_pct:.1f}% too far from 70%"
    assert abs(val_pct - 10) < 2, f"Val ratio {val_pct:.1f}% too far from 10%"
    assert abs(test_pct - 20) < 2, f"Test ratio {test_pct:.1f}% too far from 20%"
    print("✓ Ratios are correct!")


def test_custom_ratios_80_10_10():
    """Test different custom ratios: 80-10-10."""
    print("\n" + "="*80)
    print("TEST 2: Custom Ratios (80-10-10)")
    print("="*80)

    root_dir = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
    text_embedding_file = "/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt"

    # Create datasets with 80-10-10 split
    train_ds = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='train',
        use_folds=False,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify=True,
        node_id=42
    )

    val_ds = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='val',
        use_folds=False,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify=True,
        node_id=42
    )

    test_ds = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='test',
        use_folds=False,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify=True,
        node_id=42
    )

    total = len(train_ds) + len(val_ds) + len(test_ds)
    train_pct = (len(train_ds) / total) * 100
    val_pct = (len(val_ds) / total) * 100
    test_pct = (len(test_ds) / total) * 100

    print(f"Train: {len(train_ds)} samples ({train_pct:.1f}%)")
    print(f"Val:   {len(val_ds)} samples ({val_pct:.1f}%)")
    print(f"Test:  {len(test_ds)} samples ({test_pct:.1f}%)")
    print(f"Total: {total} samples")

    # Verify ratios are close to target
    assert abs(train_pct - 80) < 2, f"Train ratio {train_pct:.1f}% too far from 80%"
    assert abs(val_pct - 10) < 2, f"Val ratio {val_pct:.1f}% too far from 10%"
    assert abs(test_pct - 10) < 2, f"Test ratio {test_pct:.1f}% too far from 10%"
    print("✓ Ratios are correct!")


def test_splits_to_load():
    """Test loading only specific splits."""
    print("\n" + "="*80)
    print("TEST 3: Loading Specific Splits (train + val only)")
    print("="*80)

    root_dir = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
    text_embedding_file = "/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt"

    # Load only train and val splits combined
    combined_ds = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='all',  # This will be ignored due to splits_to_load
        splits_to_load=['train', 'val'],
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    # Compare with separate train and val datasets
    train_ds = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='train',
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    val_ds = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='val',
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    print(f"Combined (train+val): {len(combined_ds)} samples")
    print(f"Train only:           {len(train_ds)} samples")
    print(f"Val only:             {len(val_ds)} samples")
    print(f"Expected sum:         {len(train_ds) + len(val_ds)} samples")

    assert len(combined_ds) == len(train_ds) + len(val_ds), \
        "Combined dataset should equal train + val"
    print("✓ splits_to_load works correctly!")


def test_single_split_load():
    """Test loading a single split using splits_to_load."""
    print("\n" + "="*80)
    print("TEST 4: Loading Single Split via splits_to_load")
    print("="*80)

    root_dir = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
    text_embedding_file = "/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt"

    # Load test split using splits_to_load
    test_via_splits = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='all',
        splits_to_load=['test'],
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    # Load test split normally
    test_normal = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='test',
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    print(f"Test via splits_to_load: {len(test_via_splits)} samples")
    print(f"Test via split param:    {len(test_normal)} samples")

    assert len(test_via_splits) == len(test_normal), \
        "Both methods should return same number of samples"
    print("✓ Single split loading works correctly!")


def test_backwards_compatibility():
    """Test that old split_ratio parameter still works."""
    print("\n" + "="*80)
    print("TEST 5: Backwards Compatibility with split_ratio")
    print("="*80)

    root_dir = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
    text_embedding_file = "/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt"

    # Use old split_ratio parameter (deprecated but should still work)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        train_ds = ESC50Dataset(
            root_dir=root_dir,
            text_embedding_file=text_embedding_file,
            split='train',
            use_folds=False,
            split_ratio=0.8,  # Old parameter
            val_ratio=0.1,
            stratify=True,
            node_id=42
        )

        # Check if deprecation warning was raised
        assert len(w) > 0, "Should have raised deprecation warning"
        assert "deprecated" in str(w[0].message).lower()
        print("✓ Deprecation warning raised correctly")

    print(f"Train samples: {len(train_ds)}")
    print("✓ Backwards compatibility maintained!")


def test_auto_split_creation():
    """Test automatic creation of train/val/test splits when split=None."""
    print("\n" + "="*80)
    print("TEST 6: Auto-Split Creation (split=None)")
    print("="*80)

    root_dir = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
    text_embedding_file = "/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt"

    # Create dataset with split=None (auto-create all splits)
    dataset = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split=None,  # Auto-create train/val/test
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    # Check that train, val, test attributes exist
    assert hasattr(dataset, 'train'), "Dataset should have .train attribute"
    assert hasattr(dataset, 'val'), "Dataset should have .val attribute"
    assert hasattr(dataset, 'test'), "Dataset should have .test attribute"
    print("✓ All split attributes exist")

    # Check sizes
    print(f"\nSplit sizes:")
    print(f"  Train: {len(dataset.train)} samples")
    print(f"  Val:   {len(dataset.val)} samples")
    print(f"  Test:  {len(dataset.test)} samples")
    total = len(dataset.train) + len(dataset.val) + len(dataset.test)
    print(f"  Total: {total} samples")

    # Verify ratios
    train_pct = (len(dataset.train) / total) * 100
    val_pct = (len(dataset.val) / total) * 100
    test_pct = (len(dataset.test) / total) * 100

    assert abs(train_pct - 70) < 2, f"Train ratio {train_pct:.1f}% too far from 70%"
    assert abs(val_pct - 10) < 2, f"Val ratio {val_pct:.1f}% too far from 10%"
    assert abs(test_pct - 20) < 2, f"Test ratio {test_pct:.1f}% too far from 20%"
    print("✓ Ratios are correct!")

    # Compare with manual creation
    train_manual = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split='train',
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    assert len(dataset.train) == len(train_manual), \
        "Auto-created train should match manually created train"
    print("✓ Auto-created splits match manual creation!")


def test_auto_split_with_selected_classes():
    """Test auto-split with selected classes."""
    print("\n" + "="*80)
    print("TEST 7: Auto-Split with Selected Classes")
    print("="*80)

    root_dir = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
    text_embedding_file = "/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt"
    selected_classes = ["dog", "cat", "rooster", "pig", "cow"]

    # Create dataset with auto-split and selected classes
    dataset = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        selected_classes=selected_classes,
        split=None,  # Auto-create
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    print(f"Selected classes: {selected_classes}")
    print(f"Train: {len(dataset.train)} samples")
    print(f"Val:   {len(dataset.val)} samples")
    print(f"Test:  {len(dataset.test)} samples")

    # Verify all splits have only selected classes
    train_classes = set([s['class_name'] for s in dataset.train.samples])
    val_classes = set([s['class_name'] for s in dataset.val.samples])
    test_classes = set([s['class_name'] for s in dataset.test.samples])

    assert train_classes.issubset(set(selected_classes)), "Train has non-selected classes"
    assert val_classes.issubset(set(selected_classes)), "Val has non-selected classes"
    assert test_classes.issubset(set(selected_classes)), "Test has non-selected classes"
    print("✓ All splits contain only selected classes!")


if __name__ == "__main__":
    try:
        test_custom_ratios()
        test_custom_ratios_80_10_10()
        test_splits_to_load()
        test_single_split_load()
        test_backwards_compatibility()
        test_auto_split_creation()
        test_auto_split_with_selected_classes()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
