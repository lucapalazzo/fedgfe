"""
Test script for ESC-50 dataset stratification.
Demonstrates train/val/test split with stratified sampling.
"""

import sys
sys.path.insert(0, '/home/lpala/fedgfe/system')

from datautils.dataset_esc50 import ESC50Dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_stratified_split():
    """Test stratified train/val/test split."""

    logger.info("=" * 80)
    logger.info("Testing ESC-50 Stratified Split")
    logger.info("=" * 80)

    # Select a subset of classes for testing
    selected_classes = ["dog", "rooster", "pig", "cow", "frog"]

    # Create train dataset with validation split
    train_dataset = ESC50Dataset(
        selected_classes=selected_classes,
        split='train',
        split_ratio=0.7,  # 70% train
        val_ratio=0.15,    # 15% validation (from total)
        stratify=True,     # Enable stratification
        use_folds=False,   # Don't use official folds
        enable_cache=False
    )

    # Create validation dataset
    val_dataset = ESC50Dataset(
        selected_classes=selected_classes,
        split='val',
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=True,
        use_folds=False,
        enable_cache=False
    )

    # Create test dataset
    test_dataset = ESC50Dataset(
        selected_classes=selected_classes,
        split='test',
        split_ratio=0.7,  # Test gets the remaining 15%
        val_ratio=0.15,
        stratify=True,
        use_folds=False,
        enable_cache=False
    )

    # Print statistics
    train_dataset.print_split_statistics()
    val_dataset.print_split_statistics()
    test_dataset.print_split_statistics()

    # Verify stratification between splits
    logger.info("\n" + "=" * 80)
    logger.info("Verifying Train vs Validation Stratification")
    logger.info("=" * 80)
    train_dataset.verify_stratification(val_dataset, tolerance=0.10)  # 10% tolerance

    logger.info("\n" + "=" * 80)
    logger.info("Verifying Train vs Test Stratification")
    logger.info("=" * 80)
    train_dataset.verify_stratification(test_dataset, tolerance=0.10)

    logger.info("\n" + "=" * 80)
    logger.info("Verifying Validation vs Test Stratification")
    logger.info("=" * 80)
    val_dataset.verify_stratification(test_dataset, tolerance=0.10)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Train samples: {len(train_dataset)} ({len(train_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
    logger.info(f"Val samples:   {len(val_dataset)} ({len(val_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
    logger.info(f"Test samples:  {len(test_dataset)} ({len(test_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
    logger.info(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")


def test_fold_based_split():
    """Test fold-based train/val/test split."""

    logger.info("\n" + "=" * 80)
    logger.info("Testing ESC-50 Fold-Based Split")
    logger.info("=" * 80)

    selected_classes = ["dog", "rooster", "pig", "cow", "frog"]

    # Train: folds 0, 1, 2
    train_dataset = ESC50Dataset(
        selected_classes=selected_classes,
        split='train',
        use_folds=True,
        train_folds=[0, 1, 2],
        enable_cache=False
    )

    # Validation: fold 3
    val_dataset = ESC50Dataset(
        selected_classes=selected_classes,
        split='val',
        use_folds=True,
        val_folds=[3],  # Explicit validation fold
        enable_cache=False
    )

    # Test: fold 4
    test_dataset = ESC50Dataset(
        selected_classes=selected_classes,
        split='test',
        use_folds=True,
        test_folds=[4],
        enable_cache=False
    )

    # Print statistics
    train_dataset.print_split_statistics()
    val_dataset.print_split_statistics()
    test_dataset.print_split_statistics()

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Train samples (folds {train_dataset.train_folds}): {len(train_dataset)}")
    logger.info(f"Val samples (folds {val_dataset.val_folds}):   {len(val_dataset)}")
    logger.info(f"Test samples (folds {test_dataset.test_folds}):  {len(test_dataset)}")
    logger.info(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")


def test_sample_loading():
    """Test that samples can be loaded correctly."""

    logger.info("\n" + "=" * 80)
    logger.info("Testing Sample Loading")
    logger.info("=" * 80)

    dataset = ESC50Dataset(
        selected_classes=["dog", "rooster"],
        split='train',
        split_ratio=0.8,
        val_ratio=0.1,
        stratify=True,
        use_folds=False,
        enable_cache=False
    )

    logger.info(f"\nLoading first 3 samples from dataset with {len(dataset)} total samples:")

    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        logger.info(f"\nSample {i}:")
        logger.info(f"  Class: {sample['class_name']}")
        logger.info(f"  Label: {sample['label'].item()}")
        logger.info(f"  Audio shape: {sample['audio'].shape}")
        logger.info(f"  Image shape: {sample['image'].shape}")
        logger.info(f"  File ID: {sample['file_id']}")
        logger.info(f"  Fold: {sample['fold']}")
        if 'text_emb' in sample:
            logger.info(f"  Text embedding available: Yes")


if __name__ == "__main__":
    # Run tests
    test_stratified_split()
    test_fold_based_split()
    test_sample_loading()

    logger.info("\n" + "=" * 80)
    logger.info("All tests completed!")
    logger.info("=" * 80)
