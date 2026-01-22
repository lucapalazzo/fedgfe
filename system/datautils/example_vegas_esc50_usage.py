"""
Example usage of VEGAS dataset with ESC50-like split management features.
Demonstrates the new capabilities for split handling.
"""

from dataset_vegas import VEGASDataset, create_vegas_dataloader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_auto_split_creation():
    """
    Example 1: Automatic split creation with split=None

    This is the RECOMMENDED approach for most use cases.
    """
    print("\n" + "="*80)
    print("Example 1: Auto-Split Creation (RECOMMENDED)")
    print("="*80)

    # Create dataset with automatic split creation
    dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split=None,  # Automatically creates .train, .val, .test
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=True,
        load_image=False,
        load_video=False
    )

    # Access individual splits
    train_loader = create_vegas_dataloader(dataset.train, batch_size=32, shuffle=True)
    val_loader = create_vegas_dataloader(dataset.val, batch_size=32, shuffle=False)
    test_loader = create_vegas_dataloader(dataset.test, batch_size=32, shuffle=False)

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return dataset


def example_2_custom_ratios():
    """
    Example 2: Using custom train/val/test ratios
    """
    print("\n" + "="*80)
    print("Example 2: Custom Split Ratios (60-20-20)")
    print("="*80)

    # Create datasets with custom ratios
    train_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        train_ratio=0.6,   # 60% for training
        val_ratio=0.2,     # 20% for validation
        test_ratio=0.2,    # 20% for testing
        stratify=True,
        load_audio=True,
        load_image=True
    )

    val_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='val',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify=True,
        load_audio=True,
        load_image=True
    )

    test_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='test',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify=True,
        load_audio=True,
        load_image=True
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def example_3_combine_splits():
    """
    Example 3: Combining multiple splits using splits_to_load

    Useful for fine-tuning where you want to use train+val together.
    """
    print("\n" + "="*80)
    print("Example 3: Combining Train + Val for Fine-tuning")
    print("="*80)

    # Combine train and val for fine-tuning
    finetune_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        splits_to_load=['train', 'val'],  # Combine these splits
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=True,
        load_image=False
    )

    # Keep test separate for final evaluation
    test_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='test',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=True,
        load_image=False
    )

    logger.info(f"Fine-tune dataset (train+val): {len(finetune_dataset)} samples")
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    return finetune_dataset, test_dataset


def example_4_federated_learning():
    """
    Example 4: Federated learning with node-specific splits

    Each node gets a reproducible but different split based on node_id.
    """
    print("\n" + "="*80)
    print("Example 4: Federated Learning Setup")
    print("="*80)

    # Node 0
    node0_train = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        train_ratio=0.8,
        val_ratio=0.0,  # No validation for federated training
        test_ratio=0.2,
        stratify=True,
        node_split_id=0,  # Node-specific data split
        load_audio=True,
        load_image=False
    )

    # Node 1
    node1_train = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        train_ratio=0.8,
        val_ratio=0.0,
        test_ratio=0.2,
        stratify=True,
        node_split_id=1,  # Different node, different data split
        load_audio=True,
        load_image=False
    )

    # Shared test set (same across all nodes)
    test_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='test',
        train_ratio=0.8,
        val_ratio=0.0,
        test_ratio=0.2,
        stratify=True,
        node_split_id=0,  # Use same node_split_id for consistent test set
        load_audio=True,
        load_image=False
    )

    logger.info(f"Node 0 train: {len(node0_train)} samples")
    logger.info(f"Node 1 train: {len(node1_train)} samples")
    logger.info(f"Shared test: {len(test_dataset)} samples")

    return node0_train, node1_train, test_dataset


def example_5_verify_stratification():
    """
    Example 5: Verifying stratification across splits
    """
    print("\n" + "="*80)
    print("Example 5: Stratification Verification")
    print("="*80)

    # Create stratified splits
    train_ds = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False
    )

    val_ds = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='val',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False
    )

    test_ds = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='test',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        load_audio=False,
        load_image=False
    )

    # Print statistics
    train_ds.print_split_statistics()
    val_ds.print_split_statistics()
    test_ds.print_split_statistics()

    # Verify stratification
    logger.info("\n=== Verifying Stratification ===")
    train_ds.verify_stratification(val_ds, tolerance=0.10)
    train_ds.verify_stratification(test_ds, tolerance=0.10)

    return train_ds, val_ds, test_ds


def example_6_no_validation_split():
    """
    Example 6: Training without validation (only train/test)
    """
    print("\n" + "="*80)
    print("Example 6: No Validation Split (Train/Test Only)")
    print("="*80)

    # Create train/test split without validation
    train_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        train_ratio=0.8,
        val_ratio=0.0,  # No validation
        test_ratio=0.2,
        stratify=True,
        load_audio=True,
        load_image=False
    )

    test_dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='test',
        train_ratio=0.8,
        val_ratio=0.0,
        test_ratio=0.2,
        stratify=True,
        load_audio=True,
        load_image=False
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset


def example_7_legacy_compatibility():
    """
    Example 7: Using legacy split_ratio parameter (backward compatibility)
    """
    print("\n" + "="*80)
    print("Example 7: Legacy API Compatibility")
    print("="*80)

    # Old API (still works but deprecated)
    dataset_old = VEGASDataset(
        selected_classes=['dog', 'baby_cry'],
        split='train',
        split_ratio=0.8,  # Legacy parameter (deprecated)
        val_ratio=0.1,
        stratify=True,
        load_audio=False,
        load_image=False
    )

    logger.info(f"Legacy API dataset: {len(dataset_old)} samples")
    logger.info("Note: split_ratio is deprecated, use train_ratio/val_ratio/test_ratio instead")

    return dataset_old


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("VEGAS Dataset - ESC50 Features Examples")
    print("="*80)

    # Example 1: Auto-split (RECOMMENDED)
    dataset = example_1_auto_split_creation()

    # Example 2: Custom ratios
    train_ds, val_ds, test_ds = example_2_custom_ratios()

    # Example 3: Combine splits
    finetune_ds, test_ds = example_3_combine_splits()

    # Example 4: Federated learning
    node0, node1, shared_test = example_4_federated_learning()

    # Example 5: Verify stratification
    train_stratified, val_stratified, test_stratified = example_5_verify_stratification()

    # Example 6: No validation
    train_no_val, test_no_val = example_6_no_validation_split()

    # Example 7: Legacy compatibility
    legacy_ds = example_7_legacy_compatibility()

    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
