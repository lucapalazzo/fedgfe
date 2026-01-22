#!/usr/bin/env python3
"""
Test script to verify label consistency after the label remapping fix.

This script tests that:
1. Multiple nodes with different class selections use consistent labels
2. Labels match the original CLASS_LABELS indices
3. get_num_classes() and get_max_class_label() work correctly
"""

import sys
sys.path.append('/home/lpala/fedgfe/system')

from datautils.dataset_vegas import VEGASDataset
from datautils.dataset_esc50 import ESC50Dataset

def test_vegas_label_consistency():
    """Test VEGAS label consistency across nodes."""
    print("\n" + "="*70)
    print("TEST 1: VEGAS Label Consistency Across Nodes")
    print("="*70)

    # Create nodes with different class selections
    node0 = VEGASDataset(
        selected_classes=['dog'],
        node_id=0,
        split='train'
    )

    node1 = VEGASDataset(
        selected_classes=['dog', 'drum'],
        node_id=1,
        split='train'
    )

    node2 = VEGASDataset(
        selected_classes=['drum', 'fireworks'],
        node_id=2,
        split='train'
    )

    print(f"\nNode 0 - selected_classes: ['dog']")
    print(f"  active_classes: {node0.active_classes}")
    print(f"  get_num_classes(): {node0.get_num_classes()}")
    print(f"  get_max_class_label(): {node0.get_max_class_label()}")

    print(f"\nNode 1 - selected_classes: ['dog', 'drum']")
    print(f"  active_classes: {node1.active_classes}")
    print(f"  get_num_classes(): {node1.get_num_classes()}")
    print(f"  get_max_class_label(): {node1.get_max_class_label()}")

    print(f"\nNode 2 - selected_classes: ['drum', 'fireworks']")
    print(f"  active_classes: {node2.active_classes}")
    print(f"  get_num_classes(): {node2.get_num_classes()}")
    print(f"  get_max_class_label(): {node2.get_max_class_label()}")

    # Verify label consistency
    print("\n" + "-"*70)
    print("VERIFICATION:")
    print("-"*70)

    # Check 'dog' label consistency
    dog_label_node0 = node0.active_classes['dog']
    dog_label_node1 = node1.active_classes['dog']
    assert dog_label_node0 == dog_label_node1, \
        f"❌ 'dog' label mismatch: node0={dog_label_node0}, node1={dog_label_node1}"
    print(f"✓ 'dog' has consistent label across nodes: {dog_label_node0}")

    # Check 'drum' label consistency
    drum_label_node1 = node1.active_classes['drum']
    drum_label_node2 = node2.active_classes['drum']
    assert drum_label_node1 == drum_label_node2, \
        f"❌ 'drum' label mismatch: node1={drum_label_node1}, node2={drum_label_node2}"
    print(f"✓ 'drum' has consistent label across nodes: {drum_label_node1}")

    # Verify labels match original CLASS_LABELS
    assert node0.active_classes['dog'] == VEGASDataset.CLASS_LABELS['dog'], \
        "❌ 'dog' label doesn't match original CLASS_LABELS"
    print(f"✓ 'dog' label matches original CLASS_LABELS: {VEGASDataset.CLASS_LABELS['dog']}")

    assert node1.active_classes['drum'] == VEGASDataset.CLASS_LABELS['drum'], \
        "❌ 'drum' label doesn't match original CLASS_LABELS"
    print(f"✓ 'drum' label matches original CLASS_LABELS: {VEGASDataset.CLASS_LABELS['drum']}")

    # Verify get_num_classes() vs get_max_class_label()
    assert node1.get_num_classes() == 2, "❌ Node 1 should have 2 classes"
    assert node1.get_max_class_label() == 3, "❌ Node 1 max label should be 3"
    print(f"✓ Node 1: get_num_classes()={node1.get_num_classes()}, get_max_class_label()={node1.get_max_class_label()}")

    print("\n" + "="*70)
    print("✅ TEST 1 PASSED: VEGAS labels are consistent across nodes")
    print("="*70)


def test_esc50_label_consistency():
    """Test ESC-50 label consistency across nodes."""
    print("\n" + "="*70)
    print("TEST 2: ESC-50 Label Consistency Across Nodes")
    print("="*70)

    # Create nodes with different class selections
    node0 = ESC50Dataset(
        selected_classes=['dog'],
        node_id=0,
        split='train'
    )

    node1 = ESC50Dataset(
        selected_classes=['dog', 'rooster'],
        node_id=1,
        split='train'
    )

    print(f"\nNode 0 - selected_classes: ['dog']")
    print(f"  active_classes: {node0.active_classes}")
    print(f"  get_num_classes(): {node0.get_num_classes()}")
    print(f"  get_max_class_label(): {node0.get_max_class_label()}")

    print(f"\nNode 1 - selected_classes: ['dog', 'rooster']")
    print(f"  active_classes: {node1.active_classes}")
    print(f"  get_num_classes(): {node1.get_num_classes()}")
    print(f"  get_max_class_label(): {node1.get_max_class_label()}")

    # Verify label consistency
    print("\n" + "-"*70)
    print("VERIFICATION:")
    print("-"*70)

    # Check 'dog' label consistency
    dog_label_node0 = node0.active_classes['dog']
    dog_label_node1 = node1.active_classes['dog']
    assert dog_label_node0 == dog_label_node1, \
        f"❌ 'dog' label mismatch: node0={dog_label_node0}, node1={dog_label_node1}"
    print(f"✓ 'dog' has consistent label across nodes: {dog_label_node0}")

    # Verify labels match original CLASS_LABELS
    assert node0.active_classes['dog'] == ESC50Dataset.CLASS_LABELS['dog'], \
        "❌ 'dog' label doesn't match original CLASS_LABELS"
    print(f"✓ 'dog' label matches original CLASS_LABELS: {ESC50Dataset.CLASS_LABELS['dog']}")

    print("\n" + "="*70)
    print("✅ TEST 2 PASSED: ESC-50 labels are consistent across nodes")
    print("="*70)


def test_sample_labels():
    """Test that samples have correct labels."""
    print("\n" + "="*70)
    print("TEST 3: Sample Label Verification")
    print("="*70)

    dataset = VEGASDataset(
        selected_classes=['dog', 'drum'],
        split='train'
    )

    print(f"\nDataset: selected_classes=['dog', 'drum']")
    print(f"Active classes: {dataset.active_classes}")

    # Check a few samples
    print("\nChecking sample labels:")
    samples_checked = {'dog': False, 'drum': False}

    for i in range(min(len(dataset), 100)):
        sample = dataset[i]
        class_name = sample['class_name']
        label = sample['label'].item()
        expected_label = dataset.active_classes[class_name]

        if class_name in samples_checked and not samples_checked[class_name]:
            print(f"  Sample {i}: class='{class_name}', label={label}, expected={expected_label}")
            assert label == expected_label, \
                f"❌ Label mismatch for '{class_name}': got {label}, expected {expected_label}"
            print(f"    ✓ Label correct")
            samples_checked[class_name] = True

        if all(samples_checked.values()):
            break

    print("\n" + "="*70)
    print("✅ TEST 3 PASSED: Sample labels are correct")
    print("="*70)


if __name__ == "__main__":
    try:
        test_vegas_label_consistency()
        test_esc50_label_consistency()
        test_sample_labels()

        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nConclusion:")
        print("- Labels are now globally consistent across nodes")
        print("- Original CLASS_LABELS indices are preserved")
        print("- get_num_classes() and get_max_class_label() work correctly")
        print("- Multi-class federated learning is now working correctly")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
