#!/usr/bin/env python3
"""
Test script to verify node defaults work with merge_config_to_args.
"""

from system.utils.config_loader import ConfigLoader
import argparse
import json

# Create a simple config with nodes section
test_config = {
    "experiment": {
        "goal": "test_with_nodes"
    },
    "federation": {
        "num_clients": 2
    },
    "nodes": {
        "0": {
            "dataset": "VEGAS"
        },
        "1": {
            "dataset": "cifar10",
            "selected_classes": [0, 1, 2]
        }
    }
}

test_path = '/tmp/test_merge_nodes.json'
with open(test_path, 'w') as f:
    json.dump(test_config, f, indent=2)

print("=" * 70)
print("Testing Node Defaults with merge_config_to_args")
print("=" * 70)

print("\nOriginal config:")
print(json.dumps(test_config, indent=2))

# Create a mock argparse Namespace with some defaults
args = argparse.Namespace()
args.goal = 'default_goal'
args.device = 'cpu'
args.num_clients = 1
args.global_rounds = 50
args.nodes_datasets = ''
args.nodes_pretext_tasks = ''
args.nodes_downstream_tasks = ''

# Load and merge config
loader = ConfigLoader(test_path, apply_defaults=True)
args = loader.merge_config_to_args(args)

print("\n" + "=" * 70)
print("After merge_config_to_args:")
print("=" * 70)

# Check if nodes are in args
print("\nChecking args.nodes:")
if hasattr(args, 'nodes'):
    if args.nodes is not None:
        print(json.dumps(args.nodes.to_dict() if hasattr(args.nodes, 'to_dict') else dict(args.nodes), indent=2))
    else:
        print("args.nodes is None")
else:
    print("args.nodes attribute not found")

print("\n" + "=" * 70)
print("Checking individual node attributes in args.nodes:")
print("=" * 70)

if hasattr(args, 'nodes') and args.nodes is not None:
    nodes_dict = args.nodes.to_dict() if hasattr(args.nodes, 'to_dict') else dict(args.nodes)

    for node_id in ['0', '1']:
        print(f"\nNode {node_id}:")
        if node_id in nodes_dict:
            node = nodes_dict[node_id]
            for key in ['dataset', 'dataset_split', 'pretext_tasks', 'task_type', 'balance_classes']:
                value = node.get(key, 'NOT FOUND')
                print(f"  {key}: {value}")
        else:
            print(f"  Node {node_id} not found!")
else:
    print("Cannot check nodes - args.nodes is not available")

print("\n" + "=" * 70)
print("Checking num_clients:")
print(f"args.num_clients = {args.num_clients}")
print("=" * 70)
