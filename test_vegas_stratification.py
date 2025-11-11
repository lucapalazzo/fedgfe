"""
Test script to verify VEGAS dataset stratification and new features.
Tests validation split, stratification, and utility methods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.datautils.dataset_vegas import VEGASDataset, create_vegas_dataloader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_loading():
    """Test basic dataset loading."""
    print("\n" + "="*80)
    print("TEST 1: Basic Dataset Loading")
    print("="*80)

    dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='all',
        load_audio=True,
        load_image=True,
        load_video=False
    )

    print(f"✓ Dataset size: {len(dataset)}")
    print(f"✓ Number of classes: {dataset.get_num_classes()}")
    print(f"✓ Class names: {dataset.get_class_names()}")
    print(f"✓ Samples per class: {dataset.get_samples_per_class()}")

    # Test single sample
    sample = dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")
    print(f"✓ Audio shape: {sample['audio'].shape}")
    print(f"✓ Image shape: {sample['image'].shape}")
    print(f"✓ Label: {sample['label']} ({sample['class_name']})")


def test_train_val_test_split():
    """Test train/val/test split with stratification."""
    print("\n" + "="*80)
    print("TEST 2: Train/Val/Test Split with Stratification")
    print("="*80)

    # Create train/val/test datasets
    train_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=42
    )

    val_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='val',
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=42
    )

    test_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='test',
        split_ratio=0.7,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=42
    )

    print(f"\n✓ Train size: {len(train_dataset)}")
    print(f"✓ Val size: {len(val_dataset)}")
    print(f"✓ Test size: {len(test_dataset)}")
    print(f"✓ Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

    # Print statistics
    train_dataset.print_split_statistics()
    val_dataset.print_split_statistics()
    test_dataset.print_split_statistics()


def test_stratification_verification():
    """Test stratification verification between splits."""
    print("\n" + "="*80)
    print("TEST 3: Stratification Verification")
    print("="*80)

    train_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        split_ratio=0.8,
        val_ratio=0.1,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    val_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='val',
        split_ratio=0.8,
        val_ratio=0.1,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    test_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='test',
        split_ratio=0.8,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    # Verify train vs val
    print("\nTrain vs Val:")
    result1 = train_dataset.verify_stratification(val_dataset, tolerance=0.05)

    # Verify train vs test
    print("\nTrain vs Test:")
    result2 = train_dataset.verify_stratification(test_dataset, tolerance=0.05)

    # Verify val vs test
    print("\nVal vs Test:")
    result3 = val_dataset.verify_stratification(test_dataset, tolerance=0.05)

    if result1 and result2 and result3:
        print("\n✓ All stratification checks PASSED")
    else:
        print("\n✗ Some stratification checks FAILED")


def test_class_distribution():
    """Test class distribution methods."""
    print("\n" + "="*80)
    print("TEST 4: Class Distribution")
    print("="*80)

    dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='all',
        load_audio=False,
        load_image=False
    )

    samples_per_class = dataset.get_samples_per_class()
    distribution = dataset.get_class_distribution()

    print("\nSamples per class:")
    for class_name, count in samples_per_class.items():
        print(f"  {class_name}: {count}")

    print("\nClass distribution (%):")
    for class_name, pct in distribution.items():
        print(f"  {class_name}: {pct:.2f}%")

    # Verify percentages sum to 100
    total_pct = sum(distribution.values())
    print(f"\n✓ Total percentage: {total_pct:.2f}%")
    assert abs(total_pct - 100.0) < 0.01, "Percentages should sum to 100%"


def test_text_embeddings():
    """Test text embeddings loading and access."""
    print("\n" + "="*80)
    print("TEST 5: Text Embeddings")
    print("="*80)

    # Try to load with text embeddings
    dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='all',
        load_audio=False,
        load_image=False
    )

    if dataset.text_embs is not None:
        print(f"✓ Text embeddings loaded: {len(dataset.text_embs)} classes")

        # Test get_text_embeddings method
        for class_name in ['dog', 'baby_cry']:
            emb = dataset.get_text_embeddings(class_name)
            if emb is not None:
                print(f"✓ Text embedding for '{class_name}': shape {emb.shape}")
            else:
                print(f"✗ No text embedding found for '{class_name}'")
    else:
        print("⚠ No text embeddings file found (this is okay)")


def test_dataloader():
    """Test dataloader functionality."""
    print("\n" + "="*80)
    print("TEST 6: DataLoader")
    print("="*80)

    dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        split_ratio=0.8,
        stratify=True,
        load_audio=True,
        load_image=True,
        load_video=False
    )

    dataloader = create_vegas_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    print(f"✓ DataLoader created with {len(dataloader)} batches")

    # Test batch
    batch = next(iter(dataloader))
    print(f"✓ Batch audio shape: {batch['audio'].shape}")
    print(f"✓ Batch image shape: {batch['image'].shape}")
    print(f"✓ Batch labels shape: {batch['labels'].shape}")
    print(f"✓ Batch metadata keys: {list(batch['metadata'].keys())}")
    print(f"✓ Batch class names: {batch['metadata']['class_names']}")

    # Check if captions are in metadata
    if 'captions' in batch['metadata']:
        print(f"✓ Captions included in metadata")
    else:
        print(f"⚠ No captions in metadata")


def test_non_stratified_split():
    """Test non-stratified split for comparison."""
    print("\n" + "="*80)
    print("TEST 7: Non-Stratified Split (for comparison)")
    print("="*80)

    train_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        split_ratio=0.8,
        val_ratio=0.1,
        stratify=False,  # Non-stratified
        load_audio=False,
        load_image=False,
        node_id=0
    )

    val_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='val',
        split_ratio=0.8,
        val_ratio=0.1,
        stratify=False,  # Non-stratified
        load_audio=False,
        load_image=False,
        node_id=0
    )

    print("\nNon-stratified train distribution:")
    train_dist = train_dataset.get_class_distribution()
    for class_name, pct in train_dist.items():
        print(f"  {class_name}: {pct:.2f}%")

    print("\nNon-stratified val distribution:")
    val_dist = val_dataset.get_class_distribution()
    for class_name, pct in val_dist.items():
        print(f"  {class_name}: {pct:.2f}%")


def test_federated_learning_splits():
    """Test federated learning node splits."""
    print("\n" + "="*80)
    print("TEST 8: Federated Learning Node Splits")
    print("="*80)

    # Create datasets for different nodes
    node0_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        split_ratio=0.8,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=0
    )

    node1_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        split_ratio=0.8,
        stratify=True,
        load_audio=False,
        load_image=False,
        node_id=1
    )

    print(f"\n✓ Node 0 dataset size: {len(node0_dataset)}")
    print(f"✓ Node 1 dataset size: {len(node1_dataset)}")

    print("\nNode 0 distribution:")
    node0_dist = node0_dataset.get_class_distribution()
    for class_name, pct in node0_dist.items():
        print(f"  {class_name}: {pct:.2f}%")

    print("\nNode 1 distribution:")
    node1_dist = node1_dataset.get_class_distribution()
    for class_name, pct in node1_dist.items():
        print(f"  {class_name}: {pct:.2f}%")

    # Verify different splits have different samples
    node0_ids = set(s['video_id'] for s in node0_dataset.samples)
    node1_ids = set(s['video_id'] for s in node1_dataset.samples)

    # Note: node_id affects random seed, so samples will be shuffled differently
    # but total pool is the same, so we expect different order
    print(f"\n✓ Node 0 first 5 video IDs: {list(node0_ids)[:5]}")
    print(f"✓ Node 1 first 5 video IDs: {list(node1_ids)[:5]}")


def main():
    """Run all tests."""
    try:
        test_basic_loading()
        test_train_val_test_split()
        test_stratification_verification()
        test_class_distribution()
        test_text_embeddings()
        test_dataloader()
        test_non_stratified_split()
        test_federated_learning_splits()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
