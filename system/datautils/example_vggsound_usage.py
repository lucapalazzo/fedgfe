"""
Example usage of VGGSound dataset loader.
Demonstrates basic loading, class filtering, and split management.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dataset_vggsound import VGGSoundDataset, create_vggsound_dataloader


def example_basic_usage():
    """Basic dataset loading example."""
    print("=" * 80)
    print("Example 1: Basic VGGSound Dataset Loading")
    print("=" * 80)

    # Load dataset with default settings
    dataset = VGGSoundDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/vggsound",
        text_embedding_file="/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
        split='all',
        load_audio=True,
        load_video=False,
        load_image=False
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    print(f"Class names (first 10): {dataset.get_class_names()[:10]}")

    # Load a single sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0 details:")
        print(f"  Class: {sample['class_name']}")
        print(f"  Label: {sample['label']}")
        print(f"  Audio shape: {sample['audio'].shape if sample['audio'] is not None else None}")
        print(f"  File ID: {sample['file_id']}")
        print(f"  YouTube ID: {sample['ytid']}")


def example_class_filtering():
    """Example of loading specific classes."""
    print("\n" + "=" * 80)
    print("Example 2: Class Filtering")
    print("=" * 80)

    # Select only specific classes
    selected_classes = ['dog_barking', 'cat_meowing', 'baby_crying']

    dataset = VGGSoundDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/vggsound",
        text_embedding_file="/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
        selected_classes=selected_classes,
        split='all',
        load_audio=True
    )

    print(f"Selected classes: {selected_classes}")
    print(f"Total samples: {len(dataset)}")
    print(f"Active classes: {dataset.get_class_names()}")
    print(f"Samples per class: {dataset.get_samples_per_class()}")


def example_train_val_test_splits():
    """Example of automatic train/val/test split creation."""
    print("\n" + "=" * 80)
    print("Example 3: Automatic Train/Val/Test Splits")
    print("=" * 80)

    # Create dataset with automatic splits
    dataset = VGGSoundDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/vggsound",
        text_embedding_file="/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
        split=None,  # Auto-create splits
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        stratify=True,
        use_official_split=False,
        load_audio=True
    )

    print(f"Train samples: {len(dataset.train)}")
    print(f"Val samples: {len(dataset.val)}")
    print(f"Test samples: {len(dataset.test)}")

    print("\nTrain split class distribution:")
    for cls, pct in list(dataset.train.get_class_distribution().items())[:5]:
        print(f"  {cls}: {pct:.2f}%")


def example_official_splits():
    """Example using official train/test splits."""
    print("\n" + "=" * 80)
    print("Example 4: Official Train/Test Splits")
    print("=" * 80)

    # Load official train split
    train_dataset = VGGSoundDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/vggsound",
        text_embedding_file="/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
        split='train',
        use_official_split=True,
        load_audio=True
    )

    # Load official test split
    test_dataset = VGGSoundDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/vggsound",
        text_embedding_file="/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
        split='test',
        use_official_split=True,
        load_audio=True
    )

    print(f"Official train samples: {len(train_dataset)}")
    print(f"Official test samples: {len(test_dataset)}")


def example_limited_samples():
    """Example of loading limited samples per class."""
    print("\n" + "=" * 80)
    print("Example 5: Limited Samples Per Class")
    print("=" * 80)

    dataset = VGGSoundDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/vggsound",
        text_embedding_file="/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
        num_samples_per_class=100,  # Only 100 samples per class
        split='all',
        load_audio=True
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Samples per class: {dataset.get_samples_per_class()}")


def example_dataloader():
    """Example of creating a PyTorch DataLoader."""
    print("\n" + "=" * 80)
    print("Example 6: PyTorch DataLoader")
    print("=" * 80)

    dataset = VGGSoundDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/vggsound",
        text_embedding_file="/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
        split='train',
        use_official_split=True,
        load_audio=True,
        num_samples_per_class=50  # Limit for faster demo
    )

    dataloader = create_vggsound_dataloader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )

    print(f"DataLoader created with {len(dataloader)} batches")

    # Get first batch
    batch = next(iter(dataloader))
    print(f"\nFirst batch:")
    print(f"  Audio shape: {batch['audio'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Class names: {batch['class_name'][:3]}...")


def example_text_embeddings():
    """Example of accessing text embeddings."""
    print("\n" + "=" * 80)
    print("Example 7: Text Embeddings")
    print("=" * 80)

    dataset = VGGSoundDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/vggsound",
        text_embedding_file="/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
        split='all',
        load_audio=True
    )

    if len(dataset) > 0:
        sample = dataset[0]
        if 'text_emb' in sample and sample['text_emb'] is not None:
            print(f"Sample class: {sample['class_name']}")
            print(f"Text embedding shape: {sample['text_emb'].shape}")
            print(f"Text embedding type: {sample['text_emb'].dtype}")
        else:
            print("No text embeddings available in sample")


def example_splits_to_load():
    """Example of loading multiple splits together."""
    print("\n" + "=" * 80)
    print("Example 8: Loading Multiple Splits Together")
    print("=" * 80)

    # Load train + val splits together
    dataset = VGGSoundDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/vggsound",
        text_embedding_file="/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
        split='all',
        splits_to_load=['train', 'val'],
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        stratify=True,
        load_audio=True
    )

    print(f"Combined train+val samples: {len(dataset)}")
    print(f"This is useful for training with validation data included")


if __name__ == "__main__":
    print("VGGSound Dataset Usage Examples\n")

    # Run examples
    try:
        example_basic_usage()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_class_filtering()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_train_val_test_splits()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_official_splits()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        example_limited_samples()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    try:
        example_dataloader()
    except Exception as e:
        print(f"Example 6 failed: {e}")

    try:
        example_text_embeddings()
    except Exception as e:
        print(f"Example 7 failed: {e}")

    try:
        example_splits_to_load()
    except Exception as e:
        print(f"Example 8 failed: {e}")

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
