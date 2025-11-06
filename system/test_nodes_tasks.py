#!/usr/bin/env python
"""
Test script to verify nodes_tasks JSON section parsing works correctly.
"""

import json
import sys
sys.path.append('/home/lpala/fedgfe/system')

from utils.config_loader import ConfigLoader

def test_nodes_tasks_parsing():
    """Test that nodes_tasks section is properly parsed and converted to CLI format."""

    # Load the updated JSON config
    try:
        loader = ConfigLoader('configs/jsrt_simple_classification.json')
        config = loader.load_config()

        print("=== LOADED JSON CONFIG ===")
        print(f"nodes_tasks section: {json.dumps(config.get('nodes_tasks', {}), indent=2)}")

        # Test the nodes_tasks parsing logic
        nodes_tasks_config = config['nodes_tasks']

        # Build CLI-compatible strings from per-node configuration
        datasets_list = []
        pretext_tasks_set = set()
        downstream_tasks_set = set()

        # Extract unique datasets and tasks from all nodes
        for node_id, node_config in nodes_tasks_config.items():
            print(f"\n--- Node {node_id} ---")
            print(f"  Dataset: {node_config.get('dataset', 'N/A')}")
            print(f"  Split: {node_config.get('dataset_split', 'N/A')}")
            print(f"  Task Type: {node_config.get('task_type', 'N/A')}")
            print(f"  Pretext Tasks: {node_config.get('pretext_tasks', [])}")

            if 'dataset' in node_config:
                dataset_name = node_config['dataset']
                dataset_split = node_config.get('dataset_split', '0')
                dataset_entry = f"{dataset_name}:1:{dataset_split}"
                if dataset_entry not in datasets_list:
                    datasets_list.append(dataset_entry)

            if 'pretext_tasks' in node_config:
                pretext_tasks_set.update(node_config['pretext_tasks'])

            if 'task_type' in node_config:
                downstream_tasks_set.add(node_config['task_type'])

        print("\n=== CONVERTED TO CLI FORMAT ===")
        print(f"nodes_datasets: {','.join(datasets_list)}")
        print(f"nodes_pretext_tasks: {','.join(pretext_tasks_set)}")
        print(f"nodes_downstream_tasks: {','.join(downstream_tasks_set)}")
        print(f"num_clients: {len(nodes_tasks_config)}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_nodes_tasks_parsing()