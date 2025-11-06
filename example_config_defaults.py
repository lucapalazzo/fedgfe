#!/usr/bin/env python3
"""
Example script demonstrating the default values functionality in ConfigLoader.
"""

from system.utils.config_loader import ConfigLoader
import json

print("=" * 70)
print("EXAMPLE: ConfigLoader Default Values")
print("=" * 70)

# Example 1: Get default values for specific sections
print("\n1. Getting default values for specific sections:")
print("-" * 70)

training_defaults = ConfigLoader.get_section_defaults('training')
print(f"Training defaults: {json.dumps(training_defaults, indent=2)}")

node_defaults = ConfigLoader.get_node_defaults()
print(f"\nNode defaults: {json.dumps(node_defaults, indent=2)}")

# Example 2: Get a specific default value
print("\n\n2. Getting specific default values:")
print("-" * 70)

lr_default = ConfigLoader.get_default('training', 'learning_rate', 0.01)
print(f"Default learning rate: {lr_default}")

batch_size = ConfigLoader.get_default('training', 'batch_size')
print(f"Default batch size: {batch_size}")

# Example 3: Export a template configuration
print("\n\n3. Exporting template configuration:")
print("-" * 70)

# Export without node examples
template_path = '/tmp/config_template.json'
ConfigLoader.export_default_config(template_path, include_nodes_example=False)
print(f"Basic template saved to: {template_path}")

# Export with node examples (without optional parameters)
template_with_nodes_path = '/tmp/config_template_with_nodes.json'
ConfigLoader.export_default_config(template_with_nodes_path, include_nodes_example=True, include_optional=False)
print(f"Template with node examples (no optionals) saved to: {template_with_nodes_path}")

# Export with node examples (including optional parameters)
template_full_path = '/tmp/config_template_full.json'
ConfigLoader.export_default_config(template_full_path, include_nodes_example=True, include_optional=True)
print(f"Template with all parameters saved to: {template_full_path}")

# Example 4: Load a minimal config and see defaults applied
print("\n\n4. Loading a minimal configuration with automatic defaults:")
print("-" * 70)

# Create a minimal config file
minimal_config = {
    "experiment": {
        "goal": "test_experiment"
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

minimal_config_path = '/tmp/minimal_config.json'
with open(minimal_config_path, 'w') as f:
    json.dump(minimal_config, f, indent=2)

print(f"Created minimal config: {json.dumps(minimal_config, indent=2)}")

# Load with defaults
print("\nLoading with apply_defaults=True:")
loader = ConfigLoader(minimal_config_path, apply_defaults=True)
config = loader.load_config()

print("\nResulting configuration (with defaults applied):")
print(json.dumps(config.to_dict(), indent=2))

print("\n" + "=" * 70)
print("Example completed!")
print("=" * 70)
