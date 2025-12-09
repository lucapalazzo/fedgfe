#!/usr/bin/env python3
"""
Test script to verify node defaults are being applied correctly.
"""

from system.utils.config_loader import ConfigLoader
import json

# Create a simple config with nodes
test_config = {
    "experiment": {
        "goal": "test"
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

test_path = '/tmp/test_node_defaults.json'
with open(test_path, 'w') as f:
    json.dump(test_config, f, indent=2)

print("=" * 70)
print("Testing Node Defaults Application")
print("=" * 70)

print("\nOriginal config:")
print(json.dumps(test_config, indent=2))

# Load with defaults
loader = ConfigLoader(test_path, apply_defaults=True)
config = loader.load_config()

print("\n" + "=" * 70)
print("After loading with apply_defaults=True:")
print("=" * 70)

print("\nNodes section:")
if 'nodes' in config.to_dict():
    print(json.dumps(config.to_dict()['nodes'], indent=2))
else:
    print("No 'nodes' section found!")

print("\n" + "=" * 70)
print("Checking individual node attributes:")
print("=" * 70)

for node_id in ['0', '1']:
    print(f"\nNode {node_id}:")
    if 'nodes' in config.to_dict() and node_id in config.to_dict()['nodes']:
        node = config.to_dict()['nodes'][node_id]
        for key in ['dataset', 'dataset_split', 'pretext_tasks', 'task_type', 'balance_classes']:
            value = node.get(key, 'NOT FOUND')
            print(f"  {key}: {value}")
    else:
        print(f"  Node {node_id} not found!")
