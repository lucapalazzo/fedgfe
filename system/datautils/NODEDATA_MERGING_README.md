# NodeData Dataset Merging

Documentation for merging datasets from multiple `NodeData` instances to create unified dataloaders.

## Overview

`NodeData` now supports merging datasets from multiple nodes, enabling scenarios where you need to:
- Combine data from multiple federated learning nodes
- Create centralized evaluation sets
- Mix different splits from different nodes
- Aggregate data for ensemble training

## New Methods

### 1. `merge_datasets_from_nodes` (Static Method)

Merge datasets from multiple NodeData instances.

```python
@staticmethod
def merge_datasets_from_nodes(node_data_list, split_type='train', return_dataset=False)
```

**Parameters:**
- `node_data_list`: List of NodeData instances to merge
- `split_type`: Type of split to merge ('train', 'val', or 'test')
- `return_dataset`: If True, returns ConcatDataset; if False, returns list of datasets

**Returns:** ConcatDataset or list of individual datasets

### 2. `create_merged_dataloader` (Static Method)

Create a DataLoader from merged datasets of multiple NodeData instances.

```python
@staticmethod
def create_merged_dataloader(node_data_list, split_type='train', batch_size=32,
                            shuffle=True, num_workers=0, drop_last=False, **kwargs)
```

**Parameters:**
- `node_data_list`: List of NodeData instances to merge
- `split_type`: Type of split to merge ('train', 'val', or 'test')
- `batch_size`: Batch size for DataLoader
- `shuffle`: Whether to shuffle the data
- `num_workers`: Number of workers for data loading
- `drop_last`: Whether to drop the last incomplete batch
- `**kwargs`: Additional arguments to pass to DataLoader

**Returns:** DataLoader with merged datasets

### 3. `merge_with_nodes` (Instance Method)

Instance method to merge this node's dataset with other nodes' datasets.

```python
def merge_with_nodes(self, other_nodes, split_type='train', batch_size=32,
                    shuffle=True, num_workers=0, drop_last=False, **kwargs)
```

**Parameters:**
- `other_nodes`: List of other NodeData instances to merge with
- Other parameters same as `create_merged_dataloader`

**Returns:** DataLoader with merged datasets including this node

### 4. `create_mixed_split_dataloader` (Static Method)

Create a DataLoader from mixed splits of multiple NodeData instances.

```python
@staticmethod
def create_mixed_split_dataloader(node_data_list, split_configs, batch_size=32,
                                 shuffle=True, num_workers=0, drop_last=False, **kwargs)
```

**Parameters:**
- `node_data_list`: List of NodeData instances
- `split_configs`: List of split types corresponding to each node
  - e.g., ['train', 'val', 'train']
- Other parameters same as `create_merged_dataloader`

**Returns:** DataLoader with merged datasets from specified splits

## Usage Examples

### Example 1: Merge Training Data from Multiple Nodes

```python
from system.datautils.node_dataset import NodeData
from system.datautils.dataset_esc50 import ESC50Dataset

# Create 3 nodes with different data
nodes = []
for node_id in range(3):
    train_data = ESC50Dataset(
        selected_classes=['dog', 'cat', 'crow'],
        split='train',
        node_id=node_id,
        stratify=True
    )

    node = NodeData(
        args=args,
        node_id=node_id,
        custom_train_dataset=train_data
    )
    nodes.append(node)

# Method 1: Using static method
merged_train_loader = NodeData.create_merged_dataloader(
    nodes,
    split_type='train',
    batch_size=32,
    shuffle=True
)

print(f"Merged dataloader has {len(merged_train_loader.dataset)} samples")

# Use in training
for batch in merged_train_loader:
    # Training code...
    pass
```

### Example 2: Node Instance Method

```python
# Node 0 merges its data with nodes 1 and 2
merged_loader = nodes[0].merge_with_nodes(
    [nodes[1], nodes[2]],
    split_type='train',
    batch_size=32,
    shuffle=True
)

# Equivalent to:
# merged_loader = NodeData.create_merged_dataloader([nodes[0], nodes[1], nodes[2]], ...)
```

### Example 3: Merge Validation Data

```python
# Create nodes with train/val/test splits
nodes = []
for node_id in range(3):
    train_data = ESC50Dataset(split='train', node_id=node_id, stratify=True)
    val_data = ESC50Dataset(split='val', node_id=node_id, stratify=True)
    test_data = ESC50Dataset(split='test', node_id=node_id, stratify=True)

    node = NodeData(
        args=args,
        node_id=node_id,
        custom_train_dataset=train_data,
        custom_val_dataset=val_data,
        custom_test_dataset=test_data
    )
    nodes.append(node)

# Merge validation data from all nodes
merged_val_loader = NodeData.create_merged_dataloader(
    nodes,
    split_type='val',
    batch_size=32,
    shuffle=False  # Don't shuffle validation data
)

# Evaluate model on merged validation set
model.eval()
with torch.no_grad():
    for batch in merged_val_loader:
        # Validation code...
        pass
```

### Example 4: Mixed Split Dataloader

Combine different splits from different nodes:

```python
# Scenario:
# - Use train split from node 0
# - Use val split from node 1
# - Use test split from node 2

mixed_loader = NodeData.create_mixed_split_dataloader(
    [nodes[0], nodes[1], nodes[2]],
    split_configs=['train', 'val', 'test'],
    batch_size=32,
    shuffle=True
)

print(f"Mixed dataloader combines:")
print(f"  - Train from Node 0: {len(nodes[0].train_dataset)} samples")
print(f"  - Val from Node 1: {len(nodes[1].val_dataset)} samples")
print(f"  - Test from Node 2: {len(nodes[2].test_dataset)} samples")
print(f"  - Total: {len(mixed_loader.dataset)} samples")
```

### Example 5: Centralized Evaluation

Create a centralized evaluation set from all nodes' test data:

```python
# Merge test data from all nodes for final evaluation
centralized_test_loader = NodeData.create_merged_dataloader(
    nodes,
    split_type='test',
    batch_size=64,
    shuffle=False,
    num_workers=4
)

# Final model evaluation
test_accuracy = evaluate_model(model, centralized_test_loader)
print(f"Centralized test accuracy: {test_accuracy:.4f}")
```

### Example 6: Federated Learning with Data Sharing

Scenario: Some nodes share data while others don't

```python
# Nodes 0 and 1 share data, node 2 keeps private
shared_nodes = [nodes[0], nodes[1]]
private_node = nodes[2]

# Create shared dataloader for nodes 0 and 1
shared_train_loader = NodeData.create_merged_dataloader(
    shared_nodes,
    split_type='train',
    batch_size=32
)

# Private dataloader for node 2
private_train_loader = private_node.load_train_data(batch_size=32)

# Train on shared data
for batch in shared_train_loader:
    # Train shared model...
    pass

# Train on private data
for batch in private_train_loader:
    # Train private model...
    pass
```

### Example 7: Get Merged Dataset (Without DataLoader)

```python
# Get merged dataset without creating DataLoader
merged_dataset = NodeData.merge_datasets_from_nodes(
    nodes,
    split_type='train',
    return_dataset=True
)

print(f"Merged dataset size: {len(merged_dataset)}")

# Create custom DataLoader with specific settings
from torch.utils.data import DataLoader

custom_loader = DataLoader(
    merged_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
```

### Example 8: Progressive Data Aggregation

Incrementally add data from nodes:

```python
# Start with node 0
all_nodes = [nodes[0]]
base_loader = NodeData.create_merged_dataloader(all_nodes, split_type='train', batch_size=32)
print(f"Round 0: {len(base_loader.dataset)} samples")

# Add node 1
all_nodes.append(nodes[1])
loader_r1 = NodeData.create_merged_dataloader(all_nodes, split_type='train', batch_size=32)
print(f"Round 1: {len(loader_r1.dataset)} samples")

# Add node 2
all_nodes.append(nodes[2])
loader_r2 = NodeData.create_merged_dataloader(all_nodes, split_type='train', batch_size=32)
print(f"Round 2: {len(loader_r2.dataset)} samples")
```

### Example 9: Cross-Node Validation

Each node validates on other nodes' data:

```python
for i, node in enumerate(nodes):
    # Get this node's model
    model = node.model

    # Validate on all other nodes' validation data
    other_nodes = [n for j, n in enumerate(nodes) if j != i]

    cross_val_loader = NodeData.create_merged_dataloader(
        other_nodes,
        split_type='val',
        batch_size=32,
        shuffle=False
    )

    # Evaluate
    accuracy = evaluate_model(model, cross_val_loader)
    print(f"Node {i} cross-validation accuracy: {accuracy:.4f}")
```

### Example 10: Stratified Merging

Ensure merged dataset maintains class balance:

```python
# Create nodes with stratified splits
nodes = []
for node_id in range(3):
    # Each node has stratified data
    train_data = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=True,
        node_id=node_id
    )

    node = NodeData(
        args=args,
        node_id=node_id,
        custom_train_dataset=train_data
    )
    nodes.append(node)

# Merge maintains stratification across nodes
merged_loader = NodeData.create_merged_dataloader(
    nodes,
    split_type='train',
    batch_size=32
)

# Verify class distribution
from collections import Counter
all_labels = []
for batch_data, batch_labels in merged_loader:
    all_labels.extend(batch_labels.tolist())

label_counts = Counter(all_labels)
total = len(all_labels)
print("\nMerged Dataset Class Distribution:")
for label, count in sorted(label_counts.items()):
    percentage = (count / total) * 100
    print(f"  Class {label}: {count} samples ({percentage:.2f}%)")
```

## Use Cases

### 1. Centralized Training
Combine data from all nodes for centralized training:
```python
centralized_loader = NodeData.create_merged_dataloader(all_nodes, 'train', batch_size=64)
train_centralized_model(model, centralized_loader)
```

### 2. Ensemble Evaluation
Evaluate ensemble on combined test set:
```python
test_loader = NodeData.create_merged_dataloader(all_nodes, 'test', batch_size=32)
ensemble_accuracy = evaluate_ensemble(models, test_loader)
```

### 3. Cross-Validation
Create cross-validation folds from different nodes:
```python
# Use node 0 and 1 for training, node 2 for validation
train_loader = NodeData.create_merged_dataloader([nodes[0], nodes[1]], 'train')
val_loader = NodeData.create_merged_dataloader([nodes[2]], 'train')
```

### 4. Data Augmentation
Combine original and augmented data from different nodes:
```python
# Node 0: original data, Node 1: augmented data
combined_loader = NodeData.create_merged_dataloader([node0, node1], 'train')
```

### 5. Partial Aggregation
Selectively merge subsets of nodes:
```python
# Merge only high-performing nodes
high_perf_nodes = [n for n in nodes if n.performance > threshold]
merged_loader = NodeData.create_merged_dataloader(high_perf_nodes, 'train')
```

## Important Notes

### 1. Memory Considerations
- Merged datasets use `ConcatDataset`, which is memory-efficient
- No data duplication occurs - only references to original datasets
- Suitable for large-scale federated learning

### 2. Shuffling
- Shuffling is applied at the merged dataset level
- Samples from different nodes are intermixed during training
- Disable shuffling for validation/test sets

### 3. Batch Composition
- Batches may contain samples from different nodes
- Useful for learning cross-node patterns
- Can be controlled with batch_size and drop_last

### 4. Node Consistency
- All nodes should have compatible dataset formats
- Labels should use consistent encoding
- Transformations should be compatible

### 5. Performance
- Use `num_workers > 0` for parallel data loading
- Use `pin_memory=True` when using GPU
- Consider `persistent_workers=True` for faster epoch transitions

## Error Handling

```python
try:
    merged_loader = NodeData.create_merged_dataloader(
        nodes,
        split_type='train',
        batch_size=32
    )
except ValueError as e:
    print(f"Error merging datasets: {e}")
    # Handle error (e.g., no datasets found, invalid split_type)
```

Common errors:
- **ValueError**: Invalid `split_type` (must be 'train', 'val', or 'test')
- **ValueError**: No datasets found in provided nodes
- **ValueError**: Length mismatch in `create_mixed_split_dataloader`

## Performance Comparison

### Before (Sequential Loading)
```python
# Train on each node separately
for node in nodes:
    train_loader = node.load_train_data(batch_size=32)
    train_model(model, train_loader)
```

### After (Merged Loading)
```python
# Train on all nodes together
merged_loader = NodeData.create_merged_dataloader(nodes, 'train', batch_size=32)
train_model(model, merged_loader)
```

Benefits:
- ✓ Single training loop
- ✓ Better batch diversity
- ✓ Simplified code
- ✓ Easier to manage

## Complete Example: Federated Learning with Merging

```python
from system.datautils.node_dataset import NodeData
from system.datautils.dataset_esc50 import ESC50Dataset

# Setup
num_nodes = 5
args = setup_args()

# Create nodes with stratified splits
print("Creating nodes...")
nodes = []
for node_id in range(num_nodes):
    train_data = ESC50Dataset(
        selected_classes=['dog', 'cat', 'crow', 'frog', 'hen'],
        split='train',
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=True,
        node_id=node_id
    )

    val_data = ESC50Dataset(
        selected_classes=['dog', 'cat', 'crow', 'frog', 'hen'],
        split='val',
        split_ratio=0.7,
        val_ratio=0.15,
        stratify=True,
        node_id=node_id
    )

    test_data = ESC50Dataset(
        selected_classes=['dog', 'cat', 'crow', 'frog', 'hen'],
        split='test',
        split_ratio=0.7,
        stratify=True,
        node_id=node_id
    )

    node = NodeData(
        args=args,
        node_id=node_id,
        custom_train_dataset=train_data,
        custom_val_dataset=val_data,
        custom_test_dataset=test_data
    )
    nodes.append(node)
    print(f"  {node}")

# Round 1: Local training (no merging)
print("\nRound 1: Local training")
for node in nodes:
    local_train_loader = node.load_train_data(batch_size=32)
    train_local_model(node.model, local_train_loader)

# Round 2: Collaborative training (with merging)
print("\nRound 2: Collaborative training")
merged_train_loader = NodeData.create_merged_dataloader(
    nodes,
    split_type='train',
    batch_size=64,
    shuffle=True,
    num_workers=4
)
train_collaborative_model(global_model, merged_train_loader)

# Validation on merged validation set
print("\nValidation")
merged_val_loader = NodeData.create_merged_dataloader(
    nodes,
    split_type='val',
    batch_size=64,
    shuffle=False
)
val_accuracy = evaluate_model(global_model, merged_val_loader)
print(f"Validation accuracy: {val_accuracy:.4f}")

# Final evaluation on merged test set
print("\nFinal Evaluation")
centralized_test_loader = NodeData.create_merged_dataloader(
    nodes,
    split_type='test',
    batch_size=64,
    shuffle=False,
    num_workers=4
)
test_accuracy = evaluate_model(global_model, centralized_test_loader)
print(f"Test accuracy: {test_accuracy:.4f}")

# Cross-node evaluation
print("\nCross-node evaluation")
for i, node in enumerate(nodes):
    other_nodes = [n for j, n in enumerate(nodes) if j != i]
    cross_test_loader = NodeData.create_merged_dataloader(
        other_nodes,
        split_type='test',
        batch_size=32,
        shuffle=False
    )
    cross_accuracy = evaluate_model(node.model, cross_test_loader)
    print(f"Node {i} cross-test accuracy: {cross_accuracy:.4f}")
```

## Summary

The dataset merging functionality in `NodeData` provides:

✅ **Multiple Merge Strategies**: Static methods, instance methods, mixed splits
✅ **Flexible Split Selection**: Merge any combination of train/val/test
✅ **Memory Efficient**: Uses `ConcatDataset` without data duplication
✅ **Easy to Use**: Simple API with sensible defaults
✅ **Federated Learning Ready**: Designed for multi-node scenarios
✅ **Compatible**: Works with ESC50Dataset, VEGASDataset, and custom datasets
✅ **Scalable**: Handles any number of nodes
✅ **Error Handling**: Clear error messages and validation

These features enable sophisticated federated learning scenarios including centralized evaluation, ensemble training, cross-validation, and progressive data aggregation.
