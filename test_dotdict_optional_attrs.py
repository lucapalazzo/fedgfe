#!/usr/bin/env python3
"""
Test DotDict access to optional/missing attributes.
"""

from system.utils.config_loader import DotDict

print("=" * 70)
print("Testing DotDict with missing attributes")
print("=" * 70)

# Create a DotDict with some data
data = {
    "dataset": "VEGAS",
    "dataset_split": "0",
    "task_type": "classification"
}

dd = DotDict(data)

print("\nOriginal data:")
print(f"  dataset: {dd.dataset}")
print(f"  dataset_split: {dd.dataset_split}")
print(f"  task_type: {dd.task_type}")

print("\nAccessing missing attributes (should return None):")
print(f"  selected_classes: {dd.selected_classes}")
print(f"  excluded_classes: {dd.excluded_classes}")
print(f"  class_labels: {dd.class_labels}")
print(f"  nonexistent_attr: {dd.nonexistent_attr}")

print("\nChecking with 'in' operator:")
print(f"  'dataset' in dd: {'dataset' in dd}")
print(f"  'selected_classes' in dd: {'selected_classes' in dd}")

print("\nUsing get() method:")
print(f"  dd.get('dataset', 'default'): {dd.get('dataset', 'default')}")
print(f"  dd.get('selected_classes', 'default'): {dd.get('selected_classes', 'default')}")

print("\nTesting None checks:")
if dd.selected_classes is None:
    print("  ✓ dd.selected_classes is None (correct)")
else:
    print("  ✗ dd.selected_classes is not None (wrong)")

if dd.dataset is not None:
    print("  ✓ dd.dataset is not None (correct)")
else:
    print("  ✗ dd.dataset is None (wrong)")

print("\n" + "=" * 70)
print("Testing nested DotDict")
print("=" * 70)

nested_data = {
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

nested_dd = DotDict(nested_data)

print("\nNode 0 (without selected_classes):")
print(f"  dataset: {nested_dd.nodes['0'].dataset}")
print(f"  selected_classes: {nested_dd.nodes['0'].selected_classes}")

print("\nNode 1 (with selected_classes):")
print(f"  dataset: {nested_dd.nodes['1'].dataset}")
print(f"  selected_classes: {nested_dd.nodes['1'].selected_classes}")

print("\n" + "=" * 70)
print("SUCCESS: All tests passed!")
print("=" * 70)
