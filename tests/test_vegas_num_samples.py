"""
Test script for VEGAS dataset num_samples functionality.
Tests the ability to limit the number of samples per class.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from system.datautils.dataset_vegas import VEGASDataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_single_class_with_num_samples():
    """Test loading a single class with num_samples limit."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Single class with num_samples=50")
    logger.info("="*70)

    dataset = VEGASDataset(
        selected_classes=['baby_cry'],
        num_samples=50,
        split='all',
        load_audio=False,
        load_image=False,
        load_video=False
    )

    logger.info(f"Total samples loaded: {len(dataset)}")
    logger.info(f"Samples per class: {dataset.get_samples_per_class()}")

    assert len(dataset) <= 50, f"Expected <= 50 samples, got {len(dataset)}"
    logger.info("✓ Test passed!")

    return dataset


def test_multiple_classes_with_num_samples():
    """Test loading multiple classes with num_samples limit."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Multiple classes with num_samples=30")
    logger.info("="*70)

    dataset = VEGASDataset(
        selected_classes=['baby_cry', 'dog', 'chainsaw'],
        num_samples=30,
        split='all',
        load_audio=False,
        load_image=False,
        load_video=False
    )

    logger.info(f"Total samples loaded: {len(dataset)}")
    logger.info(f"Samples per class: {dataset.get_samples_per_class()}")

    samples_per_class = dataset.get_samples_per_class()
    for class_name, count in samples_per_class.items():
        assert count <= 30, f"Class {class_name} has {count} samples, expected <= 30"

    logger.info("✓ Test passed!")

    return dataset


def test_num_samples_with_splits():
    """Test num_samples with train/val/test splits."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: num_samples with train/val/test splits")
    logger.info("="*70)

    # Create dataset with auto-splits
    dataset = VEGASDataset(
        selected_classes=['baby_cry'],
        num_samples=50,
        split=None,  # Auto-create train/val/test
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        load_audio=False,
        load_image=False,
        load_video=False
    )

    logger.info(f"Train samples: {len(dataset.train)}")
    logger.info(f"Val samples: {len(dataset.val)}")
    logger.info(f"Test samples: {len(dataset.test)}")
    logger.info(f"Total (train+val+test): {len(dataset.train) + len(dataset.val) + len(dataset.test)}")

    total_samples = len(dataset.train) + len(dataset.val) + len(dataset.test)
    assert total_samples <= 50, f"Total samples {total_samples} exceeds limit of 50"

    logger.info("✓ Test passed!")

    return dataset


def test_no_num_samples_limit():
    """Test that without num_samples, all samples are loaded."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: No num_samples limit (all samples)")
    logger.info("="*70)

    dataset = VEGASDataset(
        selected_classes=['baby_cry'],
        split='all',
        load_audio=False,
        load_image=False,
        load_video=False
    )

    logger.info(f"Total samples loaded: {len(dataset)}")
    logger.info(f"Samples per class: {dataset.get_samples_per_class()}")

    logger.info("✓ Test passed!")

    return dataset


def test_all_classes_with_num_samples():
    """Test loading all classes with num_samples limit."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: All classes with num_samples=20")
    logger.info("="*70)

    dataset = VEGASDataset(
        num_samples=20,
        split='all',
        load_audio=False,
        load_image=False,
        load_video=False
    )

    logger.info(f"Total samples loaded: {len(dataset)}")
    logger.info(f"Number of classes: {dataset.get_num_classes()}")
    logger.info(f"Samples per class: {dataset.get_samples_per_class()}")

    samples_per_class = dataset.get_samples_per_class()
    for class_name, count in samples_per_class.items():
        assert count <= 20, f"Class {class_name} has {count} samples, expected <= 20"

    logger.info("✓ Test passed!")

    return dataset


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("VEGAS Dataset num_samples Feature Tests")
    logger.info("="*70)

    try:
        # Run all tests
        test_single_class_with_num_samples()
        test_multiple_classes_with_num_samples()
        test_num_samples_with_splits()
        test_no_num_samples_limit()
        test_all_classes_with_num_samples()

        logger.info("\n" + "="*70)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
