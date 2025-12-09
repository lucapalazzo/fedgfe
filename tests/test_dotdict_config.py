#!/usr/bin/env python3
"""
Test script to demonstrate DotDict functionality in config_loader.py
"""

import sys
import os
import argparse

# Add the system path to import config_loader
sys.path.append('/home/lpala/fedgfe/system')

from utils.config_loader import ConfigLoader, DotDict


def test_dotdict():
    """Test DotDict functionality."""
    print("=== Testing DotDict Functionality ===\n")

    # Create test data
    test_data = {
        "experiment": {
            "name": "test_experiment",
            "params": {
                "learning_rate": 0.001,
                "batch_size": 32
            }
        },
        "nodes": {
            "0": {
                "dataset": "cifar10",
                "selected_classes": ["cat", "dog"]
            },
            "1": {
                "dataset": "mnist",
                "selected_classes": [0, 1, 2]
            }
        },
        "features": ["feature1", "feature2", "feature3"]
    }

    # Convert to DotDict
    config = DotDict(test_data)

    print("1. Dot notation access:")
    print(f"   config.experiment.name = {config.experiment.name}")
    print(f"   config.experiment.params.learning_rate = {config.experiment.params.learning_rate}")
    print(f"   config.nodes.0.dataset = {config.nodes[0].dataset}")  # Note: numeric keys need bracket notation
    print(f"   config.nodes['1'].selected_classes = {config.nodes['1'].selected_classes}")

    print("\n2. Bracket notation access (traditional):")
    print(f"   config['experiment']['name'] = {config['experiment']['name']}")
    print(f"   config['nodes']['0']['dataset'] = {config['nodes']['0']['dataset']}")

    print("\n3. Mixed access:")
    print(f"   config.experiment['params'].batch_size = {config.experiment['params'].batch_size}")
    print(f"   config['nodes'].0.selected_classes = {config['nodes'][0].selected_classes}")

    print("\n4. List access:")
    print(f"   config.features[0] = {config.features[0]}")
    print(f"   config.features = {config.features}")

    print("\n5. Safe access with get():")
    print(f"   config.get('nonexistent', 'default') = {config.get('nonexistent', 'default')}")
    print(f"   config.experiment.get('nonexistent', 'default') = {config.experiment.get('nonexistent', 'default')}")

    print("\n6. Setting values:")
    config.new_field = "new_value"
    config.experiment.new_param = 42
    print(f"   config.new_field = {config.new_field}")
    print(f"   config.experiment.new_param = {config.experiment.new_param}")

    print("\n7. Convert back to dict:")
    regular_dict = config.to_dict()
    print(f"   type(regular_dict) = {type(regular_dict)}")
    print(f"   regular_dict['experiment']['new_param'] = {regular_dict['experiment']['new_param']}")


def test_config_loader():
    """Test ConfigLoader with DotDict."""
    print("\n\n=== Testing ConfigLoader with DotDict ===\n")

    # Create a mock args object
    args = argparse.Namespace()
    args.algorithm = "FedAvg"  # default value

    # Load the example config
    config_path = "/home/lpala/fedgfe/configs/example_class_selection.json"

    if os.path.exists(config_path):
        loader = ConfigLoader(config_path)
        updated_args = loader.merge_config_to_args(args)

        print("1. Access through args.json_config with dot notation:")
        if hasattr(updated_args, 'json_config'):
            print(f"   args.json_config.federation.algorithm = {updated_args.json_config.federation.algorithm}")
            print(f"   args.json_config.experiment.seed = {updated_args.json_config.experiment.seed}")

            # Access nodes configuration
            if hasattr(updated_args.json_config, 'nodes'):
                print(f"   args.json_config.nodes['0'].dataset = {updated_args.json_config.nodes['0'].dataset}")
                print(f"   args.json_config.nodes[0].selected_classes = {updated_args.json_config.nodes[0].selected_classes}")

        print("\n2. Traditional CLI args still work:")
        print(f"   args.algorithm = {args.algorithm}")
        print(f"   args.seed = {getattr(args, 'seed', 'not set')}")

        print("\n3. Raw config access:")
        if hasattr(updated_args, 'raw_json_config'):
            print(f"   type(args.raw_json_config) = {type(updated_args.raw_json_config)}")

        print("\n4. Node-specific class selection processing:")
        if hasattr(updated_args, 'nodes_tasks') and updated_args.nodes_tasks:
            for node_id, node_config in updated_args.nodes_tasks.items():
                if '_processed_classes' in node_config:
                    print(f"   Node {node_id}: {node_config.get('selected_classes', [])} -> {node_config['_processed_classes']}")
    else:
        print(f"Config file not found: {config_path}")


if __name__ == "__main__":
    test_dotdict()
    test_config_loader()

    print("\n\n=== Usage Examples ===")
    print("""
# Now you can access JSON config in your code like this:

# Traditional way (still works):
dataset_name = args.raw_json_config['nodes']['0']['dataset']

# New dot notation way:
dataset_name = args.json_config.nodes[0].dataset
algorithm = args.json_config.federation.algorithm
learning_rate = args.json_config.training.learning_rate

# Mixed access:
node_0_config = args.json_config.nodes[0]
selected_classes = node_0_config.selected_classes

# Safe access with defaults:
batch_size = args.json_config.training.get('batch_size', 32)
""")