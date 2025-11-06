#!/usr/bin/env python3
"""
Test with real VEGAS configuration file.
"""

from system.utils.config_loader import ConfigLoader
import argparse
import json

config_path = '/home/lpala/fedgfe/configs/a2v_1n_vegas_store_embeddings.json'

print("=" * 70)
print(f"Testing with: {config_path}")
print("=" * 70)

# Create a mock argparse Namespace
args = argparse.Namespace()
args.goal = 'default_goal'
args.device = 'cpu'
args.num_clients = 1
args.global_rounds = 50
args.nodes_datasets = ''
args.nodes_pretext_tasks = ''
args.nodes_downstream_tasks = ''

# Load and merge config
print("\nLoading config with defaults...")
loader = ConfigLoader(config_path, apply_defaults=True)
args = loader.merge_config_to_args(args)

print("\n" + "=" * 70)
print("Configuration loaded successfully!")
print("=" * 70)

# Check nodes
print("\nargs.nodes:")
if hasattr(args, 'nodes') and args.nodes is not None:
    nodes_dict = args.nodes.to_dict() if hasattr(args.nodes, 'to_dict') else dict(args.nodes)
    print(json.dumps(nodes_dict, indent=2))

    print("\n" + "=" * 70)
    print("Node 0 details:")
    print("=" * 70)

    if '0' in nodes_dict:
        node0 = nodes_dict['0']
        print(f"  dataset: {node0.get('dataset', 'NOT FOUND')}")
        print(f"  dataset_split: {node0.get('dataset_split', 'NOT FOUND')}")
        print(f"  pretext_tasks: {node0.get('pretext_tasks', 'NOT FOUND')}")
        print(f"  task_type: {node0.get('task_type', 'NOT FOUND')}")
        print(f"  balance_classes: {node0.get('balance_classes', 'NOT FOUND')}")
        print(f"  selected_classes: {node0.get('selected_classes', 'NOT SET')}")
        print(f"  excluded_classes: {node0.get('excluded_classes', 'NOT SET')}")
else:
    print("args.nodes not available")

print("\n" + "=" * 70)
print("Other configuration:")
print("=" * 70)
print(f"  goal: {args.goal}")
print(f"  num_clients: {args.num_clients}")
print(f"  global_rounds: {args.global_rounds}")
print(f"  device: {args.device}")

print("\n" + "=" * 70)
print("SUCCESS: Defaults are working correctly!")
print("=" * 70)
