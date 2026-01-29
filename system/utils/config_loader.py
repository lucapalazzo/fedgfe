"""
Configuration loader module for JSON-based experiment configuration.
Provides functionality to load JSON config and merge with command-line arguments.
"""

import json
import os
from typing import Dict, Any, Optional
import argparse


class DotDict:
    """
    A dictionary that supports dot notation access.
    Allows accessing dict['key'] as dict.key
    """
    def __init__(self, data=None):
        if data is None:
            data = {}
        self._data = {}
        for key, value in data.items():
            self._set_item(key, value)

    def _set_item(self, key, value):
        """Recursively convert nested dicts to DotDict."""
        if isinstance(value, dict):
            self._data[key] = DotDict(value)
        elif isinstance(value, list):
            self._data[key] = [
                DotDict(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            self._data[key] = value

    def __getattr__(self, key):
        """Enable dot notation access."""
        if key.startswith('_'):
            # Allow access to internal attributes
            return object.__getattribute__(self, key)
        # Return None for missing keys instead of raising AttributeError
        # This makes DotDict more user-friendly for optional attributes
        return self._data.get(key, None)

    def __setattr__(self, key, value):
        """Enable dot notation setting."""
        if key.startswith('_'):
            # Set internal attributes normally
            object.__setattr__(self, key, value)
        else:
            self._set_item(key, value)

    def __getitem__(self, key):
        """Enable bracket notation access."""
        # Support both string and integer keys
        if isinstance(key, int):
            key = str(key)
        return self._data[key]

    def __setitem__(self, key, value):
        """Enable bracket notation setting."""
        if isinstance(key, int):
            key = str(key)
        self._set_item(key, value)

    def __contains__(self, key):
        """Support 'in' operator."""
        if isinstance(key, int):
            key = str(key)
        return key in self._data

    def __len__(self):
        """Return the number of items in the DotDict."""
        return len(self._data)

    def __iter__(self):
        """Support iteration over keys."""
        return iter(self._data)

    def __repr__(self):
        """String representation."""
        return f"DotDict({self._data})"

    def __str__(self):
        """String representation."""
        return str(self._data)

    def keys(self):
        """Return keys."""
        return self._data.keys()

    def values(self):
        """Return values."""
        return self._data.values()

    def items(self):
        """Return items."""
        return self._data.items()

    def get(self, key, default=None):
        """Safe get with default value."""
        if isinstance(key, int):
            key = str(key)
        return self._data.get(key, default)

    def to_dict(self):
        """Convert back to regular dict recursively."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, DotDict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


class ConfigLoader:
    """Handles loading and processing of JSON configuration files."""

    # Default values for configuration parameters organized by section
    DEFAULT_VALUES = {
        'experiment': {
            'goal': 'test',
            'device': 'cuda',
            'device_id': '0',
            'runs': 1,
            'seed': -1,
            'optimize_memory': False,
            'debug': False,
            'optimize_memory_usage': False,
            'use_balooning': False
        },
        'federation': {
            'algorithm': 'FedAvg',
            'num_clients': 2,
            'global_rounds': 100,
            'local_epochs': 1,
            'join_ratio': 1.0,
            'eval_gap': 1,
            "generate_images_frequency": 0
            # 'model_aggregation': 'fedavg'
        },
        'model': {
            'backbone': 'cnn',
            'backbone_model': 'hf_vit',
            'pretrained': False,
            'num_classes': 10,
            'embedding_size': 768,
            'patch_size': 16
        },
        'training': {
            'optimizer': 'SGD',
            'learning_rate': 0.005,
            'batch_size': 10,
            'weight_decay': 0.001,
            'momentum': 0.9,
            'learning_rate_schedule': False
        },
        'dataset': {
            'name': 'mnist',
            'path': '',
            'image_size': -1,
            'transform': False,
            'partition': 'pat',
            'dir_alpha': 0.1,
            'balance': False
        },
        'rewind': {
            'epochs': 0,
            'ratio': 0.0,
            'interval': 0,
            'strategy': 'none'
        },
        'fedgfe': {
            'nodes_datasets': 'cifar10',
            'nodes_downstream_tasks': 'none',
            'nodes_pretext_tasks': '',
            'nodes_training_sequence': 'both',
            'cls_token_only': False,
            'limit_samples_number': 0
        },
        'feda2v': {
            'diffusion_type': 'sd',
            'use_act_loss': False,
            'use_text_loss': True,
            'use_image_loss': False,
            'audio_model_name': 'MIT/ast-finetuned-audioset-10-10-0.4593',
            'image_model_name': 'google/vit-base-patch16-224-in21k',
            'img_pipe_name': 'runwayml/stable-diffusion-v1-5',
            'img_lcm_lora_id': 'latent-consistency/lcm-lora-sdv1-5',
            'audio_pipe_name': 'cvssp/audioldm-l-full',
            'adapters_learning_rate': 0.0001,
            'adapters_weight_decay': 0.0001,
            'clip_adapter_learning_rate': 0.0001,
            'clip_adapter_weight_decay': 0.0001,
            'clip_adapter_learning_rate_schedule': False,
            't5_adapter_learning_rate': 0.001,
            't5_adapter_weight_decay': 0.001,
            't5_adapter_learning_rate_schedule': False,
            'steps': 1000,
            'mode': 'train_nodata',
            'ablation_type': 'only_t5',
            'controllability_type': 'volume',
            'target_img_path': 'data/img',
            'img_out_path': 'exp/img',
            'audio_out_path': 'exp/audio',
            'class_to_activate': 8,
            'single_class': False,
            'checkpoint_name': 'best_model',
            'project': 'sinestesia',
            'entity': 'ctlab-team',
            'wandb_mode': 'offline',
            'store_audio_embeddings': False,
            'audio_embedding_file_name': 'audio_embeddings.pt',
            'generate_nodes_images_frequency': 0,
            'generate_global_images_frequency': 0,
            "generate_low_memomy_footprint": False,
            'global_model_train': False,
            'global_model_train_epochs': 1,
            'global_model_train_from_nodes_adapters': False,
            'global_model_train_from_generator': False,
            'global_model_train_from_nodes_audio_embeddings': False,
            'global_model_train_inputs': 'none',
            'generate_from_clip_text_embeddings': False,
            'generate_from_t5_text_embeddings': False,
            'adapter_aggregation_method': 'none',
            'text_losses_summed': False,
            'compute_global_mean_from_class_means': False,
            'save_generated_images_splits': ['test', 'val'],
            'output_image_base_name': 'img',
            'generation_split_for_metrics': ['test', 'val'],
            'nodes_test_metrics_splits': ["val", "test"],
            'nodes_train_metrics_splits': ["train"],
            'server_test_metrics_splits': ["val", "test"],
            'server_train_metrics_splits': ["train"],
            # Adapter checkpoint settings
            'adapter_checkpoint_dir': 'checkpoints/adapters',
            'adapter_checkpoint_base_name': 'adapter',
            'adapter_save_checkpoint': False,
            'adapter_load_checkpoint': False,
            'adapter_checkpoint_frequency': 5,
            'adapter_checkpoint_per_type': True,
            # Embeddings checkpoint settings
            'generate_embeddings_only': False,
            'embeddings_checkpoint_dir': 'checkpoints/embeddings',
            # Generator settings
            'use_generator': False,
            'generator_type': 'vae',
            'generator_training_mode': False,
            'generator_only_mode': False,
            'use_pretrained_generators': False,
            'generator_checkpoint_dir': 'checkpoints/generators',
            'generator_checkpoint_base_name': 'generator',
            'generator_save_checkpoint': False,
            'generator_load_checkpoint': False,
            'generator_checkpoint_frequency': 5,
            'generator_granularity': 'unified',
            'generator_class_groups': None,
            'synthetic_samples_per_class': 'auto',
            'reset_generator_on_class_change': True,
            'shared_generator_in_only_mode': True

        },
        'wandb': {
            'disabled': False
        },
        'node': {
            # Default values for individual node configuration
            'dataset_split': 'all',
            'pretext_tasks': [],
            'task_type': 'classification',
            'selected_classes': None,
            'excluded_classes': None,
            'num_samples': None,
            'class_labels': None,
            'class_remapping': None,
            'limit_samples': None,
            'balance_classes': False,
            # ESC-50 specific parameters
            'use_folds': False,
            'train_folds': [0, 1, 2, 3],
            'test_folds': [4],
            # Node splitting parameters for federated learning
            'node_split_id': None,  # ID of this node's data split (0-indexed)
            'num_nodes': None,      # Total number of nodes (for equal distribution)
            'samples_per_node': None,  # Fixed number of samples per node per class
            'node_split_seed': 42   # Random seed for reproducible node splits
        }
    }

    # Common class label mappings for popular datasets
    DATASET_CLASS_LABELS = {
        'cifar10': {
            'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        },
        'cifar100': {
            # Fine labels (subset shown for brevity)
            'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4,
            'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9,
            # ... (full mapping would include all 100 classes)
        },
        'mnist': {
            'zero': 0, '0': 0, 'one': 1, '1': 1, 'two': 2, '2': 2,
            'three': 3, '3': 3, 'four': 4, '4': 4, 'five': 5, '5': 5,
            'six': 6, '6': 6, 'seven': 7, '7': 7, 'eight': 8, '8': 8,
            'nine': 9, '9': 9
        },
        'imagenet': {
            # Subset of ImageNet classes (can be extended)
            'tench': 0, 'goldfish': 1, 'great_white_shark': 2, 'tiger_shark': 3,
            'hammerhead': 4, 'electric_ray': 5, 'stingray': 6, 'cock': 7,
            # ... (full mapping would include all 1000 classes)
        }
    }

    def __init__(self, config_path: str, apply_defaults: bool = True):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the JSON configuration file
            apply_defaults: Whether to apply default values for missing parameters
        """
        self.config_path = config_path
        self.config = None
        self.apply_defaults = apply_defaults

    @classmethod
    def get_default(cls, section: str, key: str, fallback=None):
        """
        Get a default value for a specific parameter.

        Args:
            section: Section name (e.g., 'experiment', 'model', 'training')
            key: Parameter key within the section
            fallback: Value to return if not found in defaults

        Returns:
            Default value or fallback if not found
        """
        return cls.DEFAULT_VALUES.get(section, {}).get(key, fallback)

    @classmethod
    def get_section_defaults(cls, section: str) -> dict:
        """
        Get all default values for a specific section.

        Args:
            section: Section name (e.g., 'experiment', 'model', 'training')

        Returns:
            Dictionary of default values for the section, empty dict if not found
        """
        return cls.DEFAULT_VALUES.get(section, {}).copy()

    @classmethod
    def get_node_defaults(cls) -> dict:
        """
        Get all default values for individual node configuration.

        Returns:
            Dictionary of default values for node configuration
        """
        return cls.DEFAULT_VALUES.get('node', {}).copy()

    @classmethod
    def export_default_config(cls, output_path: Optional[str] = None, include_nodes_example: bool = False,
                             include_optional: bool = False) -> str:
        """
        Export a template configuration file with all default values.

        Args:
            output_path: Path to save the template JSON file. If None, returns JSON string.
            include_nodes_example: If True, includes example node configurations
            include_optional: If True, includes parameters with None values

        Returns:
            JSON string of the template configuration
        """
        template = cls.DEFAULT_VALUES.copy()

        # Remove 'node' section from template as it's not a top-level section
        if 'node' in template:
            node_defaults = template.pop('node')

            # Add example nodes if requested
            if include_nodes_example:
                # Filter out None values if not including optional parameters
                if not include_optional:
                    node_defaults = {k: v for k, v in node_defaults.items() if v is not None}

                template['nodes'] = {
                    '0': node_defaults.copy(),
                    '1': node_defaults.copy()
                }
                template['nodes']['0']['dataset'] = 'cifar10'
                template['nodes']['1']['dataset'] = 'mnist'

        json_str = json.dumps(template, indent=2)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
            print(f"Template configuration saved to: {output_path}")

        return json_str

    def _apply_default_values(self, config_dict: dict) -> dict:
        """
        Apply default values to configuration for missing parameters.

        Args:
            config_dict: The loaded configuration dictionary

        Returns:
            Updated configuration with defaults applied
        """
        result = config_dict.copy()

        # Apply defaults for top-level sections
        for section_name, default_values in self.DEFAULT_VALUES.items():
            # Skip 'node' section as it's for individual nodes, not a top-level section
            if section_name == 'node':
                continue

            # Create section if it doesn't exist
            if section_name not in result:
                result[section_name] = {}

            # Apply defaults for missing keys in this section
            section = result[section_name]
            for key, default_value in default_values.items():
                if key not in section:
                    section[key] = default_value
                    print(f"Applied default for {section_name}.{key}: {default_value}")

        # Apply defaults to individual nodes in 'nodes' or 'nodes_tasks' sections
        self._apply_node_defaults(result, 'nodes')
        self._apply_node_defaults(result, 'nodes_tasks')

        return result

    def _apply_node_defaults(self, config_dict: dict, section_name: str):
        """
        Apply default values to individual node configurations.

        Args:
            config_dict: The configuration dictionary
            section_name: Name of the nodes section ('nodes' or 'nodes_tasks')
        """
        if section_name not in config_dict:
            return

        nodes_section = config_dict[section_name]
        if not isinstance(nodes_section, dict):
            return

        node_defaults = self.DEFAULT_VALUES.get('node', {})

        # Iterate through each node and apply defaults
        for node_id, node_config in nodes_section.items():
            if not isinstance(node_config, dict):
                continue

            for key, default_value in node_defaults.items():
                if key not in node_config:
                    # Only apply if default is not None (allows opt-in parameters)
                    if default_value is not None:
                        node_config[key] = default_value
                        print(f"Applied default for {section_name}.{node_id}.{key}: {default_value}")

    def load_config(self) -> DotDict:
        """
        Load configuration from JSON file and convert to DotDict for dot notation access.

        Returns:
            DotDict containing the configuration with dot notation support

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            raw_config = json.load(f)

        # Apply default values for missing parameters if enabled
        if self.apply_defaults:
            raw_config = self._apply_default_values(raw_config)

        # Convert to DotDict for dot notation access
        self.config = DotDict(raw_config)

        # Store the raw dictionary as well for backwards compatibility
        self.raw_config = raw_config

        return self.config

    def merge_config_to_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        Merge JSON configuration into argparse Namespace.
        CLI arguments take precedence over JSON values.

        Args:
            args: Parsed command-line arguments

        Returns:
            Updated args namespace with JSON values merged
        """
        if self.config is None:
            self.load_config()

        # Add the JSON configuration to args for dot notation access
        args.json_config = self.config

        # Also store the raw config for backward compatibility
        args.raw_json_config = self.raw_config

        # Get default values for comparison (to detect CLI overrides)
        parser = argparse.ArgumentParser()
        self._add_all_arguments(parser, args)
        defaults = parser.parse_args()

        # Map JSON structure to CLI arguments
        self._map_experiment_config(args, defaults)
        self._map_federation_config(args, defaults)
        self._map_model_config(args, defaults)
        self._map_training_config(args, defaults)
        self._map_dataset_config(args, defaults)
        self._map_rewind_config(args, defaults)
        self._map_fedgfe_config(args, defaults)
        self._map_feda2v_config(args, defaults)
        self._map_nodes_tasks_config(args, defaults)
        self._map_wandb_config(args, defaults)

        # Calculate patch_count after all configs are loaded
        self._calculate_patch_count(args)

        return args

    def _safe_set_attr(self, args: argparse.Namespace, defaults: argparse.Namespace,
                       attr_name: str, config_dict: dict, config_key: str):
        """Safely set attribute if it exists in both args and defaults and wasn't overridden by CLI."""
        if (hasattr(args, attr_name) and hasattr(defaults, attr_name) and
            getattr(args, attr_name) == getattr(defaults, attr_name) and
            config_key in config_dict):
            setattr(args, attr_name, config_dict[config_key])

    def _process_class_selection(self, selected_classes, dataset_name=None, custom_labels=None):
        """
        Process class selection which can be numeric indices or text labels.

        Args:
            selected_classes: List of class identifiers (int, str, or mixed)
            dataset_name: Name of the dataset for label lookup
            custom_labels: Custom label mapping dictionary

        Returns:
            List of numeric class indices
        """
        if not selected_classes:
            return []

        processed = []

        # Use custom labels if provided, otherwise try dataset-specific labels
        label_map = custom_labels if custom_labels else {}
        if not label_map and dataset_name and dataset_name in self.DATASET_CLASS_LABELS:
            label_map = self.DATASET_CLASS_LABELS[dataset_name]

        for cls in selected_classes:
            if isinstance(cls, int):
                # Already numeric
                processed.append(cls)
            elif isinstance(cls, str):
                # Try to convert string to int first
                if cls.isdigit():
                    processed.append(int(cls))
                elif label_map and cls.lower() in label_map:
                    # Look up in label map
                    processed.append(label_map[cls.lower()])
                else:
                    # If no mapping found, try to use as-is or skip
                    print(f"Warning: Could not map class label '{cls}' to numeric index")
            elif isinstance(cls, list):
                # Handle range notation like [0, 5] meaning classes 0 through 5
                if len(cls) == 2 and isinstance(cls[0], int) and isinstance(cls[1], int):
                    processed.extend(range(cls[0], cls[1] + 1))

        # Remove duplicates and sort
        processed = sorted(list(set(processed)))
        return processed

    def _map_experiment_config(self, args: argparse.Namespace, defaults: argparse.Namespace):
        """Map experiment section from JSON to args."""
        if 'experiment' not in self.config:
            return

        exp_config = self.config['experiment']

        # Map experiment config using helper
        self._safe_set_attr(args, defaults, 'goal', exp_config, 'goal')
        self._safe_set_attr(args, defaults, 'device', exp_config, 'device')
        self._safe_set_attr(args, defaults, 'device_id', exp_config, 'device_id')
        self._safe_set_attr(args, defaults, 'times', exp_config, 'runs')
        self._safe_set_attr(args, defaults, 'seed', exp_config, 'seed')

    def _map_federation_config(self, args: argparse.Namespace, defaults: argparse.Namespace):
        """Map federation section from JSON to args."""
        if 'federation' not in self.config:
            return

        fed_config = self.config['federation']

        if hasattr(args, 'algorithm') and hasattr(defaults, 'algorithm') and args.algorithm == defaults.algorithm and 'algorithm' in fed_config:
            args.algorithm = fed_config['algorithm']

        if hasattr(args, 'num_clients') and hasattr(defaults, 'num_clients') and args.num_clients == defaults.num_clients and 'num_clients' in fed_config:
            args.num_clients = fed_config['num_clients']

        if hasattr(args, 'global_rounds') and hasattr(defaults, 'global_rounds') and args.global_rounds == defaults.global_rounds and 'global_rounds' in fed_config:
            args.global_rounds = fed_config['global_rounds']

        if hasattr(args, 'local_epochs') and hasattr(defaults, 'local_epochs') and args.local_epochs == defaults.local_epochs and 'local_epochs' in fed_config:
            args.local_epochs = fed_config['local_epochs']

        if hasattr(args, 'join_ratio') and hasattr(defaults, 'join_ratio') and args.join_ratio == defaults.join_ratio and 'join_ratio' in fed_config:
            args.join_ratio = fed_config['join_ratio']

        if hasattr(args, 'eval_gap') and hasattr(defaults, 'eval_gap') and args.eval_gap == defaults.eval_gap and 'eval_gap' in fed_config:
            args.eval_gap = fed_config['eval_gap']
                                       
        if hasattr(args, 'model_aggregation') and hasattr(defaults, 'model_aggregation') and args.model_aggregation == defaults.model_aggregation and 'model_aggregation' in fed_config:
            args.model_aggregation = fed_config['model_aggregation']

    def _map_model_config(self, args: argparse.Namespace, defaults: argparse.Namespace):
        """Map model section from JSON to args."""
        if 'model' not in self.config:
            return

        model_config = self.config['model']

        # Use safe_set_attr helper for all mappings
        self._safe_set_attr(args, defaults, 'model', model_config, 'backbone')
        self._safe_set_attr(args, defaults, 'nodes_backbone_model', model_config, 'backbone_model')
        self._safe_set_attr(args, defaults, 'model_pretrain', model_config, 'pretrained')
        self._safe_set_attr(args, defaults, 'num_classes', model_config, 'num_classes')
        self._safe_set_attr(args, defaults, 'embedding_size', model_config, 'embedding_size')
        self._safe_set_attr(args, defaults, 'patch_size', model_config, 'patch_size')

        # Support explicit patch_count from JSON (calculation happens later)
        if 'patch_count' in model_config:
            args.patch_count = model_config['patch_count']
            print(f"Using explicit patch_count: {model_config['patch_count']}")

    def _map_training_config(self, args: argparse.Namespace, defaults: argparse.Namespace):
        """Map training section from JSON to args."""
        if 'training' not in self.config:
            return

        train_config = self.config['training']

        # Use safe_set_attr helper for all mappings
        self._safe_set_attr(args, defaults, 'model_optimizer', train_config, 'optimizer')
        self._safe_set_attr(args, defaults, 'local_learning_rate', train_config, 'learning_rate')
        self._safe_set_attr(args, defaults, 'batch_size', train_config, 'batch_size')
        self._safe_set_attr(args, defaults, 'model_optimizer_weight_decay', train_config, 'weight_decay')
        self._safe_set_attr(args, defaults, 'model_optimizer_momentum', train_config, 'momentum')
        self._safe_set_attr(args, defaults, 'learning_rate_schedule', train_config, 'learning_rate_schedule')

    def _map_dataset_config(self, args: argparse.Namespace, defaults: argparse.Namespace):
        """Map dataset section from JSON to args."""
        if 'dataset' not in self.config:
            return

        dataset_config = self.config['dataset']

        # Use safe_set_attr helper for all mappings
        self._safe_set_attr(args, defaults, 'dataset', dataset_config, 'name')
        self._safe_set_attr(args, defaults, 'dataset_path', dataset_config, 'path')
        self._safe_set_attr(args, defaults, 'dataset_image_size', dataset_config, 'image_size')
        self._safe_set_attr(args, defaults, 'dataset_transform', dataset_config, 'transform')
        self._safe_set_attr(args, defaults, 'dataset_partition', dataset_config, 'partition')
        self._safe_set_attr(args, defaults, 'dataset_dir_alpha', dataset_config, 'dir_alpha')
        self._safe_set_attr(args, defaults, 'dataset_balance', dataset_config, 'balance')

    def _map_rewind_config(self, args: argparse.Namespace, defaults: argparse.Namespace):
        """Map rewind section from JSON to args."""
        if 'rewind' not in self.config:
            return

        rewind_config = self.config['rewind']

        # Use safe_set_attr helper for all mappings
        self._safe_set_attr(args, defaults, 'rewind_epochs', rewind_config, 'epochs')
        self._safe_set_attr(args, defaults, 'rewind_ratio', rewind_config, 'ratio')
        self._safe_set_attr(args, defaults, 'rewind_interval', rewind_config, 'interval')
        self._safe_set_attr(args, defaults, 'rewind_strategy', rewind_config, 'strategy')

    def _map_fedgfe_config(self, args: argparse.Namespace, defaults: argparse.Namespace):
        """Map FedGFE specific configuration from JSON to args."""
        if 'fedgfe' not in self.config:
            return

        fedgfe_config = self.config['fedgfe']

        # Use safe_set_attr helper for all mappings
        self._safe_set_attr(args, defaults, 'nodes_datasets', fedgfe_config, 'nodes_datasets')
        self._safe_set_attr(args, defaults, 'nodes_downstream_tasks', fedgfe_config, 'nodes_downstream_tasks')
        self._safe_set_attr(args, defaults, 'nodes_pretext_tasks', fedgfe_config, 'nodes_pretext_tasks')
        self._safe_set_attr(args, defaults, 'nodes_training_sequence', fedgfe_config, 'nodes_training_sequence')
        self._safe_set_attr(args, defaults, 'cls_token_only', fedgfe_config, 'cls_token_only')
        self._safe_set_attr(args, defaults, 'limit_samples_number', fedgfe_config, 'limit_samples_number')

    def _map_feda2v_config(self, args: argparse.Namespace, defaults: argparse.Namespace):
        """Map FedA2V (Audio2Visual) specific configuration from JSON to args."""
        if 'feda2v' not in self.config:
            return

        feda2v_config = self.config['feda2v']

        # Audio2Visual specific parameters
        self._safe_set_attr(args, defaults, 'diffusion_type', feda2v_config, 'diffusion_type')
        self._safe_set_attr(args, defaults, 'use_act_loss', feda2v_config, 'use_act_loss')
        self._safe_set_attr(args, defaults, 'use_text_loss', feda2v_config, 'use_text_loss')
        self._safe_set_attr(args, defaults, 'use_image_loss', feda2v_config, 'use_image_loss')

        # Model names and paths
        self._safe_set_attr(args, defaults, 'audio_model_name', feda2v_config, 'audio_model_name')
        self._safe_set_attr(args, defaults, 'image_model_name', feda2v_config, 'image_model_name')
        self._safe_set_attr(args, defaults, 'img_pipe_name', feda2v_config, 'img_pipe_name')
        self._safe_set_attr(args, defaults, 'img_lcm_lora_id', feda2v_config, 'img_lcm_lora_id')
        self._safe_set_attr(args, defaults, 'audio_pipe_name', feda2v_config, 'audio_pipe_name')

        # Training parameters specific to A2V
        self._safe_set_attr(args, defaults, 'lr1', feda2v_config, 'lr1')
        self._safe_set_attr(args, defaults, 'lr2', feda2v_config, 'lr2')
        self._safe_set_attr(args, defaults, 'lr3', feda2v_config, 'lr3')
        self._safe_set_attr(args, defaults, 'steps', feda2v_config, 'steps')

        # Mode and ablation settings
        self._safe_set_attr(args, defaults, 'mode', feda2v_config, 'mode')
        self._safe_set_attr(args, defaults, 'ablation_type', feda2v_config, 'ablation_type')
        self._safe_set_attr(args, defaults, 'controllability_type', feda2v_config, 'controllability_type')

        # Dataset and paths for A2V
        self._safe_set_attr(args, defaults, 'target_img_path', feda2v_config, 'target_img_path')
        self._safe_set_attr(args, defaults, 'img_out_path', feda2v_config, 'img_out_path')
        self._safe_set_attr(args, defaults, 'audio_out_path', feda2v_config, 'audio_out_path')
        self._safe_set_attr(args, defaults, 'class_to_activate', feda2v_config, 'class_to_activate')
        self._safe_set_attr(args, defaults, 'single_class', feda2v_config, 'single_class')

        # Checkpoint settings
        self._safe_set_attr(args, defaults, 'checkpoint_name', feda2v_config, 'checkpoint_name')

        # Adapter checkpoint settings
        self._safe_set_attr(args, defaults, 'adapter_checkpoint_dir', feda2v_config, 'adapter_checkpoint_dir')
        self._safe_set_attr(args, defaults, 'adapter_checkpoint_base_name', feda2v_config, 'adapter_checkpoint_base_name')
        self._safe_set_attr(args, defaults, 'adapter_save_checkpoint', feda2v_config, 'adapter_save_checkpoint')
        self._safe_set_attr(args, defaults, 'adapter_load_checkpoint', feda2v_config, 'adapter_load_checkpoint')
        self._safe_set_attr(args, defaults, 'adapter_checkpoint_frequency', feda2v_config, 'adapter_checkpoint_frequency')
        self._safe_set_attr(args, defaults, 'adapter_checkpoint_per_type', feda2v_config, 'adapter_checkpoint_per_type')

        # Embeddings checkpoint settings
        self._safe_set_attr(args, defaults, 'generate_embeddings_only', feda2v_config, 'generate_embeddings_only')
        self._safe_set_attr(args, defaults, 'embeddings_checkpoint_dir', feda2v_config, 'embeddings_checkpoint_dir')

        # WandB settings for A2V
        self._safe_set_attr(args, defaults, 'project', feda2v_config, 'project')
        self._safe_set_attr(args, defaults, 'entity', feda2v_config, 'entity')
        self._safe_set_attr(args, defaults, 'wandb_mode', feda2v_config, 'wandb_mode')

    def _map_nodes_tasks_config(self, args: argparse.Namespace, defaults: argparse.Namespace):
        """Map nodes_tasks or nodes section from JSON to args, converting to CLI format."""
        # Support both 'nodes_tasks' and 'nodes' keys for flexibility
        nodes_tasks_config = None
        nodes_config = None

        if 'nodes_tasks' in self.config:
            nodes_tasks_config = self.config['nodes_tasks']

        if 'nodes' in self.config:
            nodes_config = self.config['nodes']

        # Use whichever is available, prefer nodes_tasks if both exist
        working_config = nodes_tasks_config if nodes_tasks_config is not None else nodes_config

        # Return if neither section exists
        if working_config is None:
            return

        # Store original nodes configuration in args
        if nodes_tasks_config is not None:
            args.nodes_tasks = nodes_tasks_config
        if nodes_config is not None:
            args.nodes = nodes_config
        args.nodes_tasks_from_config = True

        # Build CLI-compatible strings from per-node configuration
        datasets_list = []
        pretext_tasks_set = set()
        downstream_tasks_set = set()

        # Extract unique datasets and tasks from all nodes
        for node_id, node_config in working_config.items():
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

            # Handle class selection for nodes
            if 'selected_classes' in node_config:
                # Process selected classes (can be list of ints or strings)
                selected_classes = node_config['selected_classes']

                # Get dataset name for this node
                dataset_name = node_config.get('dataset', None)

                # Get custom labels if provided
                custom_labels = node_config.get('class_labels', None)

                # Store processed classes in node config for later use
                node_config['_processed_classes'] = self._process_class_selection(
                    selected_classes, dataset_name, custom_labels
                )

                print(f"Node {node_id}: Selected classes {selected_classes} -> {node_config['_processed_classes']}")

            # Handle class labels mapping (text to numeric) for custom datasets
            if 'class_labels' in node_config:
                node_config['_class_labels'] = node_config['class_labels']

            # Handle excluded classes (classes to skip)
            if 'excluded_classes' in node_config:
                excluded_classes = node_config['excluded_classes']
                dataset_name = node_config.get('dataset', None)
                custom_labels = node_config.get('class_labels', None)

                # Process excluded classes
                node_config['_excluded_classes'] = self._process_class_selection(
                    excluded_classes, dataset_name, custom_labels
                )

                # If both selected and excluded are specified, remove excluded from selected
                if '_processed_classes' in node_config:
                    node_config['_processed_classes'] = [
                        c for c in node_config['_processed_classes']
                        if c not in node_config['_excluded_classes']
                    ]

            # Handle class remapping (map old indices to new ones)
            if 'class_remapping' in node_config:
                node_config['_class_remapping'] = node_config['class_remapping']

        # Convert to CLI format and apply only if not overridden
        if datasets_list:
            nodes_datasets_str = ','.join(datasets_list)
            self._safe_set_attr(args, defaults, 'nodes_datasets',
                              {'nodes_datasets': nodes_datasets_str}, 'nodes_datasets')

        if pretext_tasks_set:
            pretext_tasks_str = ','.join(pretext_tasks_set)
            self._safe_set_attr(args, defaults, 'nodes_pretext_tasks',
                              {'nodes_pretext_tasks': pretext_tasks_str}, 'nodes_pretext_tasks')

        if downstream_tasks_set:
            downstream_tasks_str = ','.join(downstream_tasks_set)
            self._safe_set_attr(args, defaults, 'nodes_downstream_tasks',
                              {'nodes_downstream_tasks': downstream_tasks_str}, 'nodes_downstream_tasks')

        # Update num_clients based on number of nodes defined
        if hasattr(args, 'num_clients'):
            args.num_clients = len(working_config)
            print(f"Set num_clients: {args.num_clients}")

    def _map_wandb_config(self, args: argparse.Namespace, defaults: argparse.Namespace):
        """Map wandb section from JSON to args."""
        if 'wandb' not in self.config:
            return

        wandb_config = self.config['wandb']

        # Map wandb.disabled to args.no_wandb
        self._safe_set_attr(args, defaults, 'no_wandb', wandb_config, 'disabled')

    def _calculate_patch_count(self, args: argparse.Namespace):
        """Calculate patch_count if not explicitly set and required parameters are available."""
        # Only calculate if patch_count not already set and we have the required parameters
        if (not hasattr(args, 'patch_count') or args.patch_count is None) and \
           hasattr(args, 'patch_size') and hasattr(args, 'dataset_image_size'):

            patch_size = getattr(args, 'patch_size', 0)
            img_size = getattr(args, 'dataset_image_size', 0)

            if patch_size > 0 and img_size > 0:
                patch_count = (img_size // patch_size) ** 2
                args.patch_count = patch_count
                print(f"Calculated patch_count: {patch_count} (img_size={img_size}, patch_size={patch_size})")

    def _add_all_arguments(self, parser: argparse.ArgumentParser, args: argparse.Namespace = None):
        """Add all arguments to parser (for getting defaults). Copy from main.py"""
        # This is a simplified version - we'll need to copy all arguments from main.py
        # For now, adding the most common ones

        if args is not None:
            # Add each attribute from the provided args namespace to the parser
            for arg_name, arg_val in vars(args).items():
                cli_name = f"--{arg_name}"
                # Only add if not already present (avoid duplicate arguments)
                if not any(cli_name == a.option_strings[0] for a in parser._actions if a.option_strings):
                    parser.add_argument(cli_name, default=arg_val)
            return
        

        parser.add_argument("--goal", type=str, default="test")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--device_id", type=str, default="0")
        parser.add_argument("--times", type=int, default=1)
        parser.add_argument("--seed", type=int, default=-1)
        parser.add_argument("--algorithm", type=str, default="FedAvg")
        parser.add_argument("--num_clients", type=int, default=2)
        parser.add_argument("--global_rounds", type=int, default=100)
        parser.add_argument("--local_epochs", type=int, default=1)
        parser.add_argument("--join_ratio", type=float, default=1.0)
        parser.add_argument("--eval_gap", type=int, default=1)
        parser.add_argument("--model", type=str, default="cnn")
        parser.add_argument("--nodes_backbone_model", type=str, default="hf_vit")
        parser.add_argument("--model_pretrain", type=bool, default=False)
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--embedding_size", type=int, default=768)
        parser.add_argument("--patch_size", type=int, default=16)
        parser.add_argument("--model_optimizer", type=str, default="SGD")
        parser.add_argument("--local_learning_rate", type=float, default=0.005)
        parser.add_argument("--batch_size", type=int, default=10)
        parser.add_argument("--model_optimizer_weight_decay", type=float, default=0.001)
        parser.add_argument("--model_optimizer_momentum", type=float, default=0.9)
        parser.add_argument("--learning_rate_schedule", type=bool, default=False)
        parser.add_argument("--dataset", type=str, default="mnist")
        parser.add_argument("--dataset_path", type=str, default="")
        parser.add_argument("--dataset_image_size", type=int, default=-1)
        parser.add_argument("--dataset_transform", type=bool, default=False)
        parser.add_argument("--dataset_partition", type=str, default="pat")
        parser.add_argument("--dataset_dir_alpha", type=float, default=0.1)
        parser.add_argument("--dataset_balance", type=bool, default=False)
        parser.add_argument("--rewind_epochs", type=int, default=0)
        parser.add_argument("--rewind_ratio", type=float, default=0)
        parser.add_argument("--rewind_interval", type=int, default=0)
        parser.add_argument("--rewind_strategy", type=str, default="none")
        parser.add_argument("--nodes_datasets", type=str, default="cifar10")
        parser.add_argument("--nodes_downstream_tasks", type=str, default="none")
        parser.add_argument("--nodes_pretext_tasks", type=str, default="")
        parser.add_argument("--nodes_training_sequence", type=str, default="both")
        parser.add_argument("--cls_token_only", type=bool, default=False)
        parser.add_argument("--limit_samples_number", type=int, default=0)
        parser.add_argument("--no_wandb", type=bool, default=False)

        # FedA2V (Audio2Visual) specific parameters
        parser.add_argument("--diffusion_type", type=str, default="sd")
        parser.add_argument("--use_act_loss", type=bool, default=False)
        parser.add_argument("--use_text_loss", type=bool, default=True)
        parser.add_argument("--use_image_loss", type=bool, default=False)
        parser.add_argument("--audio_model_name", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
        parser.add_argument("--image_model_name", type=str, default="google/vit-base-patch16-224-in21k")
        parser.add_argument("--img_pipe_name", type=str, default="runwayml/stable-diffusion-v1-5")
        parser.add_argument("--img_lcm_lora_id", type=str, default="latent-consistency/lcm-lora-sdv1-5")
        parser.add_argument("--audio_pipe_name", type=str, default="cvssp/audioldm-l-full")
        parser.add_argument("--lr1", type=float, default=0.001)
        parser.add_argument("--lr2", type=float, default=0.0001)
        parser.add_argument("--lr3", type=float, default=0.0001)
        parser.add_argument("--steps", type=int, default=1000)
        parser.add_argument("--mode", type=str, default="train_nodata")
        parser.add_argument("--ablation_type", type=str, default="only_t5")
        parser.add_argument("--controllability_type", type=str, default="volume")
        parser.add_argument("--target_img_path", type=str, default="data/img")
        parser.add_argument("--img_out_path", type=str, default="exp/img")
        parser.add_argument("--audio_out_path", type=str, default="exp/audio")
        parser.add_argument("--class_to_activate", type=int, default=8)
        parser.add_argument("--single_class", type=bool, default=False)
        parser.add_argument("--checkpoint_name", type=str, default="best_model")
        parser.add_argument("--project", type=str, default="sinestesia")
        parser.add_argument("--entity", type=str, default="ctlab-team")
        parser.add_argument("--wandb_mode", type=str, default="offline")


def load_config_to_args(config_path: str, args: argparse.Namespace, apply_defaults: bool = True) -> argparse.Namespace:
    """
    Convenience function to load JSON config and merge with args.

    Args:
        config_path: Path to JSON configuration file
        args: Parsed command-line arguments
        apply_defaults: Whether to apply default values for missing parameters

    Returns:
        Updated args namespace
    """
    if config_path is None:
        return args

    loader = ConfigLoader(config_path, apply_defaults=apply_defaults)
    return loader.merge_config_to_args(args)