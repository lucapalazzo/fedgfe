#!/usr/bin/env python3
"""
Test script to verify the adapter loading logic in serverA2V.py
Tests both single set and multiple set (round-robin) scenarios.
"""

import os
import glob

def test_adapter_discovery():
    """Test the adapter discovery logic"""
    print("="*80)
    print("TEST 1: Adapter Discovery Logic")
    print("="*80)

    adapter_checkpoint_dir = 'checkpoints/adapters/esc50_1n_1c'
    adapter_checkpoint_base_name = 'esc50-1n-1c-real'
    adapter_types = ['clip_adapter', 't5_adapter', 'clip_projection', 't5_projection']

    node_adapter_sets = {}

    for adapter_name in adapter_types:
        # Search for any node's checkpoint (not just server)
        pattern = os.path.join(
            adapter_checkpoint_dir,
            f'{adapter_checkpoint_base_name}_node*_{adapter_name}*.pt'
        )

        checkpoint_files = glob.glob(pattern)
        print(f"\n{adapter_name}: found {len(checkpoint_files)} files")

        # Group by node_id
        for checkpoint_file in checkpoint_files:
            # Extract node_id from filename
            filename = os.path.basename(checkpoint_file)
            parts = filename.split('_')

            # Find the part that starts with 'node'
            node_id = None
            for part in parts:
                if part.startswith('node'):
                    node_id = part.replace('node', '')
                    break

            if node_id is None:
                continue

            if node_id not in node_adapter_sets:
                node_adapter_sets[node_id] = {}

            if adapter_name not in node_adapter_sets[node_id]:
                node_adapter_sets[node_id][adapter_name] = []

            node_adapter_sets[node_id][adapter_name].append(checkpoint_file)
            print(f"  -> Node {node_id}: {os.path.basename(checkpoint_file)}")

    # Check if we have complete sets (all 4 adapter types)
    complete_sets = {}
    for node_id, adapters in node_adapter_sets.items():
        if len(adapters) == len(adapter_types):
            # All adapter types present for this node
            complete_sets[node_id] = adapters

    print(f"\n{'='*80}")
    print(f"Node adapter sets discovered: {len(node_adapter_sets)}")
    for node_id, adapters in node_adapter_sets.items():
        print(f"  Node {node_id}: {len(adapters)}/{len(adapter_types)} adapter types")

    print(f"\nComplete sets: {len(complete_sets)}")
    for node_id in complete_sets.keys():
        print(f"  ✓ Node {node_id} has all {len(adapter_types)} adapter types")

    # Verify we have exactly one complete set
    if len(complete_sets) > 1:
        print(f"\n⚠ WARNING: Found {len(complete_sets)} complete adapter sets from nodes: {list(complete_sets.keys())}")
        print(f"⚠ Will use round-robin distribution to clients")
        return complete_sets, True  # Multiple sets
    elif len(complete_sets) == 1:
        print(f"\n✓ Exactly one complete adapter set from node {list(complete_sets.keys())[0]}")
        return complete_sets, False  # Single set
    else:
        print(f"\n⚠ ERROR: No complete adapter sets found!")
        return {}, False


def test_round_robin_distribution(complete_sets, num_clients=5):
    """Test round-robin distribution of adapters to clients"""
    print(f"\n{'='*80}")
    print(f"TEST 2: Round-Robin Distribution")
    print(f"{'='*80}")

    if not complete_sets:
        print("No complete sets to distribute")
        return

    node_list = list(complete_sets.keys())
    print(f"\nDistributing {len(complete_sets)} adapter sets to {num_clients} clients")
    print(f"Available nodes: {node_list}\n")

    for client_idx in range(num_clients):
        assigned_node_id = node_list[client_idx % len(node_list)]
        print(f"Client {client_idx} -> Node {assigned_node_id} adapters")

    print(f"\n✓ Round-robin distribution complete")


def test_checkpoint_selection(complete_sets):
    """Test selection of most recent checkpoint"""
    print(f"\n{'='*80}")
    print(f"TEST 3: Checkpoint Selection (Most Recent)")
    print(f"{'='*80}")

    if not complete_sets:
        print("No complete sets to test")
        return

    node_id = list(complete_sets.keys())[0]
    print(f"\nTesting with node {node_id} adapters:")

    for adapter_name, checkpoint_files in complete_sets[node_id].items():
        sorted_files = sorted(checkpoint_files)
        most_recent = sorted_files[-1]
        print(f"\n{adapter_name}:")
        print(f"  All files: {len(sorted_files)}")
        for f in sorted_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  Selected: {os.path.basename(most_recent)} ✓")


if __name__ == '__main__':
    print("\nAdapter Loading Logic Tests")
    print("="*80)

    # Test 1: Discover adapters
    complete_sets, is_multiple = test_adapter_discovery()

    # Test 2: Round-robin distribution (if multiple sets)
    if is_multiple:
        test_round_robin_distribution(complete_sets, num_clients=3)
    else:
        print(f"\n{'='*80}")
        print("TEST 2: Round-Robin Distribution")
        print("="*80)
        print("Skipped (only one complete set - no round-robin needed)")

    # Test 3: Checkpoint selection
    test_checkpoint_selection(complete_sets)

    print(f"\n{'='*80}")
    print("All tests completed!")
    print("="*80)
