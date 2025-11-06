# JSON Configuration System Guide

## Overview

The JSON configuration system allows you to define complex federated learning experiments using structured JSON files instead of long command-line arguments. This guide covers all supported sections and features.

## Basic Usage

```bash
# Use JSON configuration
python main.py --config configs/my_experiment.json

# Override specific JSON values with CLI
python main.py --config configs/my_experiment.json --algorithm FedProx --global_rounds 200
```

## JSON Schema

### Complete Example

```json
{
  "experiment": {
    "goal": "test",
    "device": "cuda",
    "device_id": "0",
    "runs": 1,
    "seed": 42
  },
  "federation": {
    "algorithm": "FedGFE",
    "num_clients": 8,
    "global_rounds": 100,
    "local_epochs": 5,
    "join_ratio": 1.0,
    "eval_gap": 1
  },
  "model": {
    "backbone": "vit",
    "backbone_model": "hf_vit",
    "pretrained": false,
    "num_classes": 10,
    "embedding_size": 768,
    "patch_size": 16
  },
  "training": {
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "batch_size": 32,
    "weight_decay": 0.01,
    "momentum": 0.9,
    "learning_rate_schedule": true
  },
  "dataset": {
    "name": "JSRT-8C-ClaSSeg",
    "path": "/path/to/dataset",
    "image_size": 224,
    "transform": true,
    "partition": "pat",
    "dir_alpha": 0.1,
    "balance": true
  },
  "fedgfe": {
    "nodes_datasets": "JSRT-8C-ClaSSeg:8:0;1;2;3",
    "nodes_downstream_tasks": "classification",
    "nodes_pretext_tasks": "image_rotation,patch_masking",
    "nodes_training_sequence": "both",
    "cls_token_only": false,
    "limit_samples_number": 0
  },
  "nodes_tasks": {
    "0": {
      "task_type": "classification",
      "pretext_tasks": ["image_rotation", "byod"],
      "dataset": "JSRT-1C-ClaSSeg",
      "dataset_split": "0"
    },
    "1": {
      "task_type": "segmentation",
      "pretext_tasks": ["patch_masking"],
      "dataset": "JSRT-1C-ClaSSeg",
      "dataset_split": "1"
    }
  },
  "wandb": {
    "disabled": true
  },
  "rewind": {
    "epochs": 0,
    "ratio": 0.0,
    "interval": 0,
    "strategy": "none"
  }
}
```

## Section Descriptions

### 1. `experiment` - General Experiment Settings

Controls overall experiment configuration.

```json
{
  "experiment": {
    "goal": "test",           // → --goal
    "device": "cuda",         // → --device
    "device_id": "0",         // → --device_id
    "runs": 1,                // → --times
    "seed": 42                // → --seed
  }
}
```

### 2. `federation` - Federated Learning Parameters

Defines federation algorithm and client behavior.

```json
{
  "federation": {
    "algorithm": "FedGFE",     // → --algorithm
    "num_clients": 8,          // → --num_clients
    "global_rounds": 100,      // → --global_rounds
    "local_epochs": 5,         // → --local_epochs
    "join_ratio": 1.0,         // → --join_ratio
    "eval_gap": 1              // → --eval_gap
  }
}
```

### 3. `model` - Model Architecture

Specifies the model architecture and parameters.

```json
{
  "model": {
    "backbone": "vit",              // → --model
    "backbone_model": "hf_vit",     // → --nodes_backbone_model
    "pretrained": false,            // → --model_pretrain
    "num_classes": 10,              // → --num_classes
    "embedding_size": 768,          // → --embedding_size
    "patch_size": 16,               // → --patch_size
    "patch_count": 196              // → --patch_count (or auto-calculated)
  }
}
```

**Nota**: `patch_count` può essere:
- Specificato esplicitamente nel JSON
- Calcolato automaticamente: `(dataset_image_size / patch_size)²`
- Per esempio: con `image_size=224` e `patch_size=16` → `patch_count=196`

### 4. `training` - Training Configuration

Controls optimizer and training hyperparameters.

```json
{
  "training": {
    "optimizer": "AdamW",              // → --model_optimizer
    "learning_rate": 0.001,            // → --local_learning_rate
    "batch_size": 32,                  // → --batch_size
    "weight_decay": 0.01,              // → --model_optimizer_weight_decay
    "momentum": 0.9,                   // → --model_optimizer_momentum
    "learning_rate_schedule": true     // → --learning_rate_schedule
  }
}
```

### 5. `dataset` - Dataset Configuration

Specifies dataset and preprocessing settings.

```json
{
  "dataset": {
    "name": "JSRT-8C-ClaSSeg",    // → --dataset
    "path": "/path/to/dataset",   // → --dataset_path
    "image_size": 224,            // → --dataset_image_size
    "transform": true,            // → --dataset_transform
    "partition": "pat",           // → --dataset_partition
    "dir_alpha": 0.1,             // → --dataset_dir_alpha
    "balance": true               // → --dataset_balance
  }
}
```

### 6. `fedgfe` - FedGFE Specific Settings

Controls FedGFE algorithm-specific parameters.

```json
{
  "fedgfe": {
    "nodes_datasets": "JSRT-8C-ClaSSeg:8:0;1;2;3",    // → --nodes_datasets
    "nodes_downstream_tasks": "classification",         // → --nodes_downstream_tasks
    "nodes_pretext_tasks": "image_rotation,patch_masking", // → --nodes_pretext_tasks
    "nodes_training_sequence": "both",                  // → --nodes_training_sequence
    "cls_token_only": false,                            // → --cls_token_only
    "limit_samples_number": 0                           // → --limit_samples_number
  }
}
```

### 7. `nodes_tasks` - Per-Node Task Configuration ⭐ **NEW**

Define specific tasks and datasets for individual nodes. This section is **automatically converted** to the existing CLI format.

```json
{
  "nodes_tasks": {
    "0": {
      "task_type": "classification",           // Downstream task type
      "pretext_tasks": ["image_rotation"],     // List of pretext tasks
      "dataset": "JSRT-1C-ClaSSeg",           // Dataset for this node
      "dataset_split": "0"                    // Split ID for this node
    },
    "1": {
      "task_type": "segmentation",
      "pretext_tasks": ["patch_masking", "byod"],
      "dataset": "JSRT-1C-ClaSSeg",
      "dataset_split": "1"
    }
  }
}
```

**Automatic Conversion:**
- Extracts unique datasets and creates `nodes_datasets` string
- Aggregates all pretext tasks into `nodes_pretext_tasks` string
- Collects all task types into `nodes_downstream_tasks` string
- Sets `num_clients` to number of nodes defined

### 8. `wandb` - Weights & Biases Settings ⭐ **NEW**

Control experiment tracking and logging.

```json
{
  "wandb": {
    "disabled": true    // → --no_wandb
  }
}
```

### 9. `rewind` - Model Rewinding Configuration

Controls model rewinding behavior (FedRewind algorithm).

```json
{
  "rewind": {
    "epochs": 0,           // → --rewind_epochs
    "ratio": 0.0,          // → --rewind_ratio
    "interval": 0,         // → --rewind_interval
    "strategy": "none"     // → --rewind_strategy
  }
}
```

## CLI Override Behavior

**CLI arguments always take precedence over JSON values.** This allows you to:

1. Define a base configuration in JSON
2. Override specific parameters via command line

```bash
# JSON has algorithm: "FedGFE", but CLI overrides to FedProx
python main.py --config base.json --algorithm FedProx --global_rounds 200
```

## Multi-Dataset Configuration

You can configure multiple datasets and splits using the `nodes_datasets` syntax:

```json
{
  "fedgfe": {
    "nodes_datasets": "JSRT-8C:8:0;1;2,cifar10:4:0;1"
  }
}
```

Format: `dataset_name:total_splits:used_splits`

## Per-Node vs Global Configuration

Two approaches are supported:

1. **Global approach** - Use `fedgfe` section for uniform configuration
2. **Per-node approach** - Use `nodes_tasks` for node-specific configuration

The `nodes_tasks` section automatically generates the appropriate `fedgfe` values.

## Example Use Cases

### Simple Classification (Single Dataset)

```json
{
  "federation": {"algorithm": "FedGFE", "num_clients": 1},
  "dataset": {"name": "JSRT-1C-ClaSSeg"},
  "fedgfe": {
    "nodes_downstream_tasks": "classification",
    "nodes_pretext_tasks": "",
    "nodes_training_sequence": "downstream"
  },
  "wandb": {"disabled": true}
}
```

### Multi-Node Multi-Task Setup

```json
{
  "nodes_tasks": {
    "0": {"task_type": "classification", "pretext_tasks": ["image_rotation"]},
    "1": {"task_type": "segmentation", "pretext_tasks": ["patch_masking"]},
    "2": {"task_type": "classification", "pretext_tasks": ["byod"]}
  }
}
```

### Medical Imaging Research

```json
{
  "federation": {"algorithm": "FedGFE", "global_rounds": 50},
  "model": {"backbone": "vit", "num_classes": 2},
  "dataset": {"name": "JSRT-8C-ClaSSeg"},
  "training": {"optimizer": "AdamW", "learning_rate": 0.001},
  "fedgfe": {
    "nodes_datasets": "JSRT-8C-ClaSSeg:8:0;1;2;3;4;5;6;7",
    "nodes_downstream_tasks": "classification,segmentation",
    "nodes_pretext_tasks": "image_rotation,patch_masking,byod"
  }
}
```

## Error Handling

The system includes robust error handling:

- **File not found**: Clear error message with file path
- **Invalid JSON**: JSON syntax error details
- **Missing sections**: Gracefully skipped, defaults used
- **Type mismatches**: Safe conversion with fallbacks

## Migration from CLI

To convert existing CLI commands to JSON:

1. Create base JSON structure
2. Map CLI arguments to appropriate JSON sections
3. Test with `--config` parameter
4. Gradually move more parameters to JSON

## Best Practices

1. **Start simple** - Begin with basic sections, add complexity gradually
2. **Use per-node config** - For heterogeneous setups, use `nodes_tasks`
3. **Keep CLI overrides minimal** - Use for quick experiments only
4. **Validate configurations** - Test JSON configs before long runs
5. **Version control** - Store JSON configs in version control
6. **Comment approach** - Use descriptive file names for different experiments

## Troubleshooting

### Common Issues

1. **JSON syntax errors** - Use a JSON validator
2. **Boolean values** - Use `true`/`false`, not `True`/`False`
3. **String vs number** - Ensure proper types (e.g., `"0"` vs `0`)
4. **Missing quotes** - All keys and strings must be quoted

### Debug Mode

Run with verbose output to see configuration loading:

```bash
python main.py --config experiment.json --verbose
```

This will show which JSON values are loaded and which CLI values override them.