#!/usr/bin/env python3
"""
Test script to verify the 20 nodes with 10 classes configuration
Simulates the config: a2v_generator_vegas_20n_10c_200s_real.json
"""

import sys
sys.path.insert(0, '/home/lpala/fedgfe/system')

from datautils.dataset_vegas import VEGASDataset

def test_20n_10c_config():
    """Test configuration with 20 nodes and 10 classes (each class on 2 nodes)"""

    print("=" * 80)
    print("Testing 20 nodes with 10 classes configuration")
    print("Each class appears on 2 nodes with different data splits")
    print("=" * 80)

    # 10 classes from VEGAS
    classes = ['dog', 'chainsaw', 'drum', 'rail_transport', 'helicopter',
               'baby_cry', 'printer', 'snoring', 'water_flowing', 'fireworks']

    # Configuration similar to the JSON config
    config = {
        'samples_per_node': 200,
        'node_split_seed': 42,
        'train_ratio': 0.9,
        'val_ratio': 0.1,
        'test_ratio': 0.0,
        'split': 'train',
        'enable_ast_cache': False,
        'load_audio': False,
        'load_image': False,
        'load_video': False
    }

    # Create datasets for all 20 nodes
    datasets = {}

    print("\nCreating datasets for 20 nodes...")
    print("-" * 80)

    # Nodes 0-9: node_split_id=0
    # Nodes 10-19: node_split_id=1
    for node_idx in range(20):
        class_idx = node_idx % 10
        class_name = classes[class_idx]
        split_id = 0 if node_idx < 10 else 1

        print(f"Node {node_idx:2d}: class={class_name:15s}, node_split_id={split_id}")

        dataset = VEGASDataset(
            selected_classes=[class_name],
            node_split_id=split_id,
            **config
        )

        datasets[node_idx] = {
            'dataset': dataset,
            'class': class_name,
            'split_id': split_id,
            'file_ids': set([s['file_id'] for s in dataset.samples])
        }

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)

    # Verify that nodes with same class but different split_id have NO overlap
    print("\nChecking for overlaps between nodes with same class...")
    print("-" * 80)

    all_good = True
    for class_name in classes:
        # Find nodes with this class
        nodes_with_class = [(idx, data) for idx, data in datasets.items()
                           if data['class'] == class_name]

        if len(nodes_with_class) == 2:
            node0_idx, node0_data = nodes_with_class[0]
            node1_idx, node1_data = nodes_with_class[1]

            overlap = node0_data['file_ids'].intersection(node1_data['file_ids'])

            status = "✓ OK" if len(overlap) == 0 else "✗ FAIL"
            print(f"{status} Class '{class_name}': "
                  f"Node {node0_idx} ({len(node0_data['file_ids'])} samples) vs "
                  f"Node {node1_idx} ({len(node1_data['file_ids'])} samples) - "
                  f"Overlap: {len(overlap)}")

            if len(overlap) > 0:
                all_good = False
                print(f"     Overlapping files: {list(overlap)[:5]}...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_samples = sum(len(data['dataset']) for data in datasets.values())
    print(f"Total nodes: {len(datasets)}")
    print(f"Total samples across all nodes: {total_samples}")
    print(f"Average samples per node: {total_samples / len(datasets):.1f}")

    if all_good:
        print("\n✓ SUCCESS: All nodes have non-overlapping samples!")
        print("The configuration is correct for 20 nodes with 10 classes.")
    else:
        print("\n✗ FAILURE: Some nodes have overlapping samples!")

    print("=" * 80)

if __name__ == "__main__":
    test_20n_10c_config()
