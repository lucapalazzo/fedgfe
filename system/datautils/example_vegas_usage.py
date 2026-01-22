#!/usr/bin/env python3
"""
Example usage of VEGASDataset with NodeData for federated learning.
Shows how to integrate custom datasets that inherit from torch.utils.data.Dataset.
"""

import sys
import argparse
import torch
sys.path.append('/home/lpala/fedgfe/system')

from datautils.node_dataset import NodeData
from datautils.dataset_vegas import VEGASDataset


def create_federated_vegas_node(args, node_id, selected_classes=None):
    """
    Create a federated learning node with VEGAS dataset.

    Args:
        args: Argument namespace with configuration
        node_id: ID of the federated node
        selected_classes: List of classes for this node

    Returns:
        NodeData object with VEGASDataset
    """

    # Create train dataset for this node
    train_dataset = VEGASDataset(
        selected_classes=selected_classes,
        split='train',
        split_ratio=0.8,
        node_split_id=node_id,  # Ensures consistent data splitting across nodes
        enable_cache=True,
        audio_sample_rate=16000,
        audio_duration=10.0,
        image_size=(224, 224)
    )

    # Create test dataset for this node
    test_dataset = VEGASDataset(
        selected_classes=selected_classes,
        split='test',
        split_ratio=0.8,
        node_split_id=node_id,  # Same node_split_id for consistent split
        enable_cache=True,
        audio_sample_rate=16000,
        audio_duration=10.0,
        image_size=(224, 224)
    )

    # Create NodeData with custom datasets
    node_data = NodeData(
        args=args,
        node_split_id=node_id,
        custom_train_dataset=train_dataset,
        custom_test_dataset=test_dataset
    )

    # Update dataset name and num_classes
    node_data.dataset = "VEGAS"
    node_data.num_classes = train_dataset.get_num_classes()

    return node_data


def example_federated_setup():
    """
    Example setup for federated learning with different nodes having different classes.
    """

    # Create mock args
    args = argparse.Namespace()
    args.dataset = "VEGAS"
    args.num_classes = 10
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define node configurations
    node_configs = [
        {'id': 0, 'classes': ['dog', 'cat', 'baby_cry']},  # Node 0: Animals and baby
        {'id': 1, 'classes': ['chainsaw', 'helicopter', 'printer']},  # Node 1: Machines
        {'id': 2, 'classes': ['drum', 'fireworks']},  # Node 2: Sounds
        {'id': 3, 'classes': None},  # Node 3: All classes
    ]

    # Create nodes
    nodes = []
    for config in node_configs:
        node = create_federated_vegas_node(
            args=args,
            node_split_id=config['id'],
            selected_classes=config['classes']
        )
        nodes.append(node)

        print(f"\n=== Node {config['id']} ===")
        print(f"Classes: {config['classes'] if config['classes'] else 'All'}")
        print(f"Train samples: {node.train_samples}")
        print(f"Test samples: {node.test_samples}")
        print(f"Number of classes: {node.num_classes}")

    return nodes


def example_data_loading(node_data):
    """
    Example of loading data from a node using custom dataset.
    """
    print(f"\n=== Loading Data from Node {node_data.id} ===")

    # Load train dataloader
    train_loader = node_data.load_train_data(batch_size=4)

    if train_loader:
        # Get a batch
        batch = next(iter(train_loader))

        print(f"Batch keys: {batch.keys()}")
        print(f"Audio shape: {batch['audio'].shape}")
        print(f"Image shape: {batch['image'].shape}")
        if batch['video'] is not None:
            print(f"Video shape: {batch['video'].shape}")
        print(f"Labels: {batch['labels']}")
        print(f"Class names: {batch['metadata']['class_names']}")

    # Load test dataloader
    test_loader = node_data.load_test_data(batch_size=4)

    if test_loader:
        print(f"Test dataloader created with {len(test_loader)} batches")


def example_integration_with_client():
    """
    Example showing how to integrate with clientA2V.
    """
    print("\n=== Integration with ClientA2V ===")

    # This would be done in clientA2V.__init__
    args = argparse.Namespace()
    args.dataset = "VEGAS"
    args.num_classes = 10
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get node configuration from JSON config
    node_config = {
        'dataset': 'VEGAS',
        'selected_classes': ['dog', 'baby_cry', 'chainsaw']
    }

    # Create VEGAS datasets
    train_dataset = VEGASDataset(
        selected_classes=node_config['selected_classes'],
        split='train',
        node_split_id=0
    )

    test_dataset = VEGASDataset(
        selected_classes=node_config['selected_classes'],
        split='test',
        node_split_id=0
    )

    # Create NodeData with VEGAS datasets
    node_data = NodeData(
        args=args,
        node_split_id=0,
        custom_train_dataset=train_dataset,
        custom_test_dataset=test_dataset
    )

    # Now the client can use node_data.load_train_data() and node_data.load_test_data()
    # which will automatically use the VEGAS dataset

    print(f"Node data created with {node_data.train_samples} train samples")
    print(f"Classes: {train_dataset.get_class_names()}")

    # Example training loop (would be in clientA2V.train())
    train_loader = node_data.load_train_data(batch_size=2)

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 2:  # Just show 2 batches
            break

        audio = batch['audio']  # [batch_size, audio_length]
        image = batch['image']  # [batch_size, 3, 224, 224]
        labels = batch['labels']  # [batch_size]

        print(f"\nBatch {batch_idx}:")
        print(f"  Audio: {audio.shape}")
        print(f"  Image: {image.shape}")
        print(f"  Labels: {labels}")

        # Here you would:
        # 1. Pass audio through Audio2Visual model
        # 2. Generate images
        # 3. Calculate loss
        # 4. Backpropagate


if __name__ == "__main__":
    print("Testing VEGAS Dataset integration with NodeData...\n")

    # Test federated setup
    nodes = example_federated_setup()

    # Test data loading from first node
    if nodes:
        example_data_loading(nodes[0])

    # Test integration example
    example_integration_with_client()

    print("\n✅ VEGAS Dataset successfully integrated with NodeData!")
    print("\nUsage in your code:")
    print("1. Create VEGASDataset with desired classes and node_id")
    print("2. Pass to NodeData as custom_train_dataset and custom_test_dataset")
    print("3. Use node_data.load_train_data() and node_data.load_test_data() as normal")
    print("4. The dataloaders will automatically use VEGAS multimodal data")