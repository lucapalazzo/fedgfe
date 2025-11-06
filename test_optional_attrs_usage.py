#!/usr/bin/env python3
"""
Test using DotDict with optional attributes as they would be used in real code.
"""

from system.utils.config_loader import ConfigLoader
import json

# Create a config with nodes, some with optional attributes, some without
test_config = {
    "experiment": {
        "goal": "test_optional_attrs"
    },
    "nodes": {
        "0": {
            "dataset": "VEGAS"
            # No optional attributes
        },
        "1": {
            "dataset": "cifar10",
            "selected_classes": [0, 1, 2, 3, 4]
        },
        "2": {
            "dataset": "mnist",
            "excluded_classes": [5, 6, 7, 8, 9],
            "class_remapping": {"0": 0, "1": 1, "2": 2}
        }
    }
}

test_path = '/tmp/test_optional_attrs.json'
with open(test_path, 'w') as f:
    json.dump(test_config, f, indent=2)

print("=" * 70)
print("Testing Optional Attributes Usage")
print("=" * 70)

# Load config
loader = ConfigLoader(test_path, apply_defaults=True)
config = loader.load_config()

print("\nIterating through nodes and accessing optional attributes:")
print("-" * 70)

for node_id, node_config in config.nodes.items():
    print(f"\nNode {node_id}:")
    print(f"  dataset: {node_config.dataset}")
    print(f"  dataset_split: {node_config.dataset_split}")
    print(f"  task_type: {node_config.task_type}")

    # Access optional attributes directly (this would have caused AttributeError before)
    print(f"  selected_classes: {node_config.selected_classes}")
    print(f"  excluded_classes: {node_config.excluded_classes}")
    print(f"  class_labels: {node_config.class_labels}")
    print(f"  class_remapping: {node_config.class_remapping}")

    # Test conditional logic
    if node_config.selected_classes is not None:
        print(f"  → Node has selected_classes: {node_config.selected_classes}")

    if node_config.excluded_classes is not None:
        print(f"  → Node has excluded_classes: {node_config.excluded_classes}")

    if node_config.class_remapping is not None:
        print(f"  → Node has class_remapping: {node_config.class_remapping}")

print("\n" + "=" * 70)
print("Testing safe attribute access patterns:")
print("=" * 70)

node0 = config.nodes['0']

# Pattern 1: Direct access with None check
print("\n1. Direct access with None check:")
if node0.selected_classes is None:
    print("   ✓ No selected_classes for node 0")

# Pattern 2: Using 'in' operator
print("\n2. Using 'in' operator:")
if 'selected_classes' in node0:
    print("   Has selected_classes")
else:
    print("   ✓ No selected_classes key in node 0")

# Pattern 3: Using get() with default
print("\n3. Using get() with default:")
classes = node0.get('selected_classes', [])
print(f"   selected_classes (with default []): {classes}")

# Pattern 4: Accessing nested DotDict attributes
print("\n4. Accessing deeply nested attributes:")
print(f"   config.nodes.0.dataset: {config.nodes['0'].dataset}")
print(f"   config.nodes.0.selected_classes: {config.nodes['0'].selected_classes}")

print("\n" + "=" * 70)
print("SUCCESS: All patterns work correctly!")
print("=" * 70)
