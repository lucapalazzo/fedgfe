"""
Nodes Tasks Parser module for handling per-node task configuration.
Supports both CLI arguments and JSON config integration.
"""

import json
import os
from typing import Dict, Any, Optional
import argparse


def parse_nodes_tasks_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    Parse nodes_tasks configuration from CLI arguments or config file.

    Args:
        args: Parsed command-line arguments

    Returns:
        Updated args namespace with nodes_tasks applied
    """
    nodes_tasks_config = None

    # Initialize nodes_tasks to None if not already present
    if not hasattr(args, 'nodes_tasks'):
        args.nodes_tasks = None

    # Priority 1: Direct CLI nodes_tasks_config
    if hasattr(args, 'nodes_tasks_config') and args.nodes_tasks_config:
        nodes_tasks_config = _load_nodes_tasks_from_cli(args.nodes_tasks_config)
        print(f"Loaded nodes_tasks from CLI: {args.nodes_tasks_config}")

    # Priority 2: From config file (already handled by config_loader)
    elif hasattr(args, 'nodes_tasks_from_config') and args.nodes_tasks_from_config:
        print("Using nodes_tasks from config file (already processed)")
        return args

    # Apply nodes_tasks configuration if found
    if nodes_tasks_config:
        # Validate configuration before applying
        if not validate_nodes_tasks_config(nodes_tasks_config):
            print("Error: Invalid nodes_tasks configuration detected")
            return args

        # Print summary before applying
        print_nodes_tasks_summary(nodes_tasks_config)

        # Store original nodes_tasks configuration in args
        args.nodes_tasks = nodes_tasks_config

        args = _apply_nodes_tasks_config(args, nodes_tasks_config)
        print(f"Applied nodes_tasks configuration for {len(nodes_tasks_config)} nodes")

    return args


def _load_nodes_tasks_from_cli(config_input: str) -> Optional[Dict[str, Any]]:
    """
    Load nodes_tasks configuration from CLI input.

    Args:
        config_input: JSON string or file path

    Returns:
        Parsed nodes_tasks configuration or None
    """
    try:
        # Try to parse as JSON string first
        if config_input.strip().startswith('{'):
            return json.loads(config_input)

        # Try to load as file path
        elif os.path.exists(config_input):
            with open(config_input, 'r') as f:
                full_config = json.load(f)
                return full_config.get('nodes_tasks', {})

        else:
            print(f"Warning: nodes_tasks_config '{config_input}' is neither valid JSON nor existing file")
            return None

    except json.JSONDecodeError as e:
        print(f"Error parsing nodes_tasks JSON: {e}")
        return None
    except Exception as e:
        print(f"Error loading nodes_tasks config: {e}")
        return None


def _apply_nodes_tasks_config(args: argparse.Namespace, nodes_tasks_config: Dict[str, Any]) -> argparse.Namespace:
    """
    Apply nodes_tasks configuration to args namespace.

    Args:
        args: Current args namespace
        nodes_tasks_config: nodes_tasks configuration dictionary

    Returns:
        Updated args namespace
    """
    # Build CLI-compatible strings from per-node configuration
    datasets_list = []
    pretext_tasks_set = set()
    downstream_tasks_set = set()

    # Extract unique datasets and tasks from all nodes
    for node_id, node_config in nodes_tasks_config.items():
        if 'dataset' in node_config:
            dataset_name = node_config['dataset']
            dataset_split = node_config.get('dataset_split', '0')

            # Build dataset string in format "dataset:total_splits:used_splits"
            # For simplicity, assume single split per node for now
            dataset_entry = f"{dataset_name}:1:{dataset_split}"
            if dataset_entry not in datasets_list:
                datasets_list.append(dataset_entry)

        if 'pretext_tasks' in node_config:
            pretext_tasks_set.update(node_config['pretext_tasks'])

        if 'task_type' in node_config:
            downstream_tasks_set.add(node_config['task_type'])

    # Apply to args namespace
    if datasets_list:
        args.nodes_datasets = ','.join(datasets_list)
        print(f"Set nodes_datasets: {args.nodes_datasets}")

    if pretext_tasks_set:
        args.nodes_pretext_tasks = ','.join(pretext_tasks_set)
        print(f"Set nodes_pretext_tasks: {args.nodes_pretext_tasks}")

    if downstream_tasks_set:
        args.nodes_downstream_tasks = ','.join(downstream_tasks_set)
        print(f"Set nodes_downstream_tasks: {args.nodes_downstream_tasks}")

    # Update num_clients based on number of nodes defined
    args.num_clients = len(nodes_tasks_config)
    print(f"Set num_clients: {args.num_clients}")

    return args


def validate_nodes_tasks_config(nodes_tasks_config: Dict[str, Any]) -> bool:
    """
    Validate nodes_tasks configuration structure.

    Args:
        nodes_tasks_config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(nodes_tasks_config, dict):
        print("Error: nodes_tasks must be a dictionary")
        return False

    if len(nodes_tasks_config) == 0:
        print("Warning: Empty nodes_tasks configuration")
        return True

    has_errors = False

    for node_id, node_config in nodes_tasks_config.items():
        if not isinstance(node_config, dict):
            print(f"Error: node '{node_id}' config must be a dictionary")
            has_errors = True
            continue

        # Check required fields
        if 'task_type' not in node_config:
            print(f"Warning: node '{node_id}' missing 'task_type' field")

        # Validate task_type values
        if 'task_type' in node_config:
            valid_task_types = ['classification', 'segmentation', 'regression']
            if node_config['task_type'] not in valid_task_types:
                print(f"Error: node '{node_id}' has invalid task_type '{node_config['task_type']}'. Valid: {valid_task_types}")
                has_errors = True

        # Validate pretext_tasks format
        if 'pretext_tasks' in node_config:
            if not isinstance(node_config['pretext_tasks'], list):
                print(f"Error: node '{node_id}' pretext_tasks must be a list")
                has_errors = True
            else:
                # Validate individual pretext tasks
                valid_pretext_tasks = ['image_rotation', 'patch_masking', 'byod', 'simclr']
                for task in node_config['pretext_tasks']:
                    if not isinstance(task, str):
                        print(f"Error: node '{node_id}' pretext task must be string, got {type(task)}")
                        has_errors = True
                    elif task not in valid_pretext_tasks:
                        print(f"Warning: node '{node_id}' unknown pretext task '{task}'. Known: {valid_pretext_tasks}")

        # Validate dataset field
        if 'dataset' in node_config:
            if not isinstance(node_config['dataset'], str):
                print(f"Error: node '{node_id}' dataset must be a string")
                has_errors = True

        # Validate dataset_split field
        if 'dataset_split' in node_config:
            split_val = node_config['dataset_split']
            if not isinstance(split_val, (str, int)):
                print(f"Error: node '{node_id}' dataset_split must be string or int")
                has_errors = True
            else:
                try:
                    int(split_val)
                except ValueError:
                    print(f"Error: node '{node_id}' dataset_split '{split_val}' is not a valid integer")
                    has_errors = True

        # Validate node_id format
        try:
            int(node_id)
        except ValueError:
            print(f"Warning: node_id '{node_id}' is not numeric - this may cause issues")

    return not has_errors


def print_nodes_tasks_summary(nodes_tasks_config: Dict[str, Any]):
    """
    Print a summary of the nodes_tasks configuration.

    Args:
        nodes_tasks_config: Configuration to summarize
    """
    print(f"\n=== Nodes Tasks Configuration Summary ===")
    print(f"Total nodes: {len(nodes_tasks_config)}")

    # Count task types
    task_type_counts = {}
    datasets_used = set()
    all_pretext_tasks = set()

    for node_id, node_config in nodes_tasks_config.items():
        # Count task types
        task_type = node_config.get('task_type', 'unknown')
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

        # Collect datasets
        if 'dataset' in node_config:
            datasets_used.add(node_config['dataset'])

        # Collect pretext tasks
        if 'pretext_tasks' in node_config:
            all_pretext_tasks.update(node_config['pretext_tasks'])

        print(f"  Node {node_id}: {task_type} on {node_config.get('dataset', 'N/A')}:{node_config.get('dataset_split', 'N/A')}")

    print(f"\nTask type distribution: {dict(task_type_counts)}")
    print(f"Datasets used: {list(datasets_used)}")
    print(f"Pretext tasks: {list(all_pretext_tasks)}")
    print("=" * 45)