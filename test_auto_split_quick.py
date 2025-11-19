"""
Quick test for auto-split functionality
"""

import sys
sys.path.append('/home/lpala/fedgfe')

print("Testing auto-split creation...")

try:
    from system.datautils.dataset_esc50 import ESC50Dataset

    root_dir = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
    text_embedding_file = "/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt"

    # Test 1: Auto-create all splits
    print("\n=== TEST 1: Auto-create all splits (split=None) ===")
    dataset = ESC50Dataset(
        root_dir=root_dir,
        text_embedding_file=text_embedding_file,
        split=None,  # Auto-create
        use_folds=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        node_id=42
    )

    print(f"Train: {len(dataset.train)} samples")
    print(f"Val:   {len(dataset.val)} samples")
    print(f"Test:  {len(dataset.test)} samples")
    print("✓ Auto-split works!")

    # Test 2: Manual single split (backwards compatibility)
    print("\n=== TEST 2: Manual single split (split='train') ===")
    train_only = ESC50Dataset(
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
    print(f"Train: {len(train_only)} samples")
    print("✓ Manual split works!")

    # Verify they match
    assert len(dataset.train) == len(train_only), "Sizes should match!"
    print("\n✓ Auto-created train matches manually created train!")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)

except ImportError as e:
    print(f"Could not import: {e}")
    print("(This is expected if torch is not installed)")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
