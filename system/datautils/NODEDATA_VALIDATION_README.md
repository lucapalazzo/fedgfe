# NodeData Validation Split Support

Documentation for the new validation split and stratification features added to `NodeData`.

## Overview

`NodeData` now supports automatic train/validation/test splits with optional stratification using scikit-learn's `train_test_split`. This provides consistent, reproducible splits that preserve class distributions across all three sets.

## New Features

### 1. Validation Split Support
- **New Parameter**: `val_ratio` (default: 0.1)
- Automatically creates a validation set from the training data
- Supports both custom validation datasets and automatic splitting

### 2. Stratification
- **New Parameter**: `stratify` (default: True)
- Uses scikit-learn's `train_test_split` for stratified sampling
- Preserves class distribution across train/val/test sets
- Falls back to random split if stratification fails

### 3. Flexible Split Ratios
- **New Parameter**: `split_ratio` (default: 0.8)
- Controls the train/(val+test) split
- Combined with `val_ratio` for three-way splits

## Usage

### Basic Usage with Custom Datasets

```python
from system.datautils.node_dataset import NodeData
from system.datautils.dataset_esc50 import ESC50Dataset

# Create train, val, test datasets with stratification
train_dataset = ESC50Dataset(
    selected_classes=['dog', 'cat', 'crow'],
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    node_id=0
)

val_dataset = ESC50Dataset(
    selected_classes=['dog', 'cat', 'crow'],
    split='val',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    node_id=0
)

test_dataset = ESC50Dataset(
    selected_classes=['dog', 'cat', 'crow'],
    split='test',
    split_ratio=0.8,
    stratify=True,
    node_id=0
)

# Create NodeData with custom datasets
node_data = NodeData(
    args=args,
    node_id=0,
    dataset_split_id=0,
    custom_train_dataset=train_dataset,
    custom_val_dataset=val_dataset,
    custom_test_dataset=test_dataset
)

# Load dataloaders
train_loader = node_data.load_train_data(batch_size=32)
val_loader = node_data.load_val_data(batch_size=32)
test_loader = node_data.load_test_data(batch_size=32)
```

### Automatic Splitting with Stratification

```python
from torch.utils.data import Dataset

# Create your custom dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['image'], self.data[idx]['label']

# Create dataset
full_dataset = MyDataset(data)

# NodeData will automatically split with stratification
node_data = NodeData(
    args=args,
    node_id=0,
    dataset_split_id=0,
    dataset=full_dataset,
    split_ratio=0.8,      # 80% for train+val
    val_ratio=0.1,        # 10% for val
    stratify=True         # Use stratification
)

# Access the splits
print(node_data)  # Shows train/val/test sample counts

# Load dataloaders
train_loader = node_data.load_train_data(batch_size=32)
val_loader = node_data.load_val_data(batch_size=32)
test_loader = node_data.load_test_data(batch_size=32)
```

### Non-Stratified Split

```python
# Disable stratification for faster splitting
node_data = NodeData(
    args=args,
    node_id=0,
    dataset_split_id=0,
    dataset=full_dataset,
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=False  # Random split without stratification
)
```

## New Parameters

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `custom_val_dataset` | Dataset | None | Custom validation dataset |
| `split_ratio` | float | 0.8 | Ratio for train split (train+val vs test) |
| `val_ratio` | float | 0.1 | Ratio for validation split from total dataset |
| `stratify` | bool | True | Whether to use stratified sampling |

## New Methods

### `load_val_data(batch_size, dataset_limit=0, prefix="", dataset_dir_prefix="")`
Load validation data and create dataloader.

```python
val_loader = node_data.load_val_data(batch_size=32)
```

**Returns**: DataLoader for validation set or None if no validation data exists.

### `get_val_dataset(dataset_limit=0)`
Get the validation dataset without creating a DataLoader.

```python
val_dataset = node_data.get_val_dataset()
```

**Returns**: Validation Dataset object or None.

### `val_stats_get()`
Get statistics for validation dataset.

```python
val_labels_count, val_labels_percent = node_data.val_stats_get()
```

**Returns**: Tuple of (labels_count, labels_percent) for validation set.

### `unload_val_data()`
Unload validation data to free memory.

```python
node_data.unload_val_data()
```

## Updated Methods

### `__str__()`
Now includes validation sample count:

```python
print(node_data)
# Output: Node 0 split id 0 dataset esc50 train samples 800 val samples 100 test samples 200
```

### `to(device)`
Now handles validation data transfer to device:

```python
node_data.to('cuda')  # Moves train, val, and test data to CUDA
```

## Split Behavior

### Three-Way Split Logic

1. **Initial Split**: Dataset → Train+Val (split_ratio) + Test (1 - split_ratio)
2. **Secondary Split**: Train+Val → Train + Val (val_ratio relative to total)

**Example with 1000 samples**:
- `split_ratio=0.8`, `val_ratio=0.1`
- Test: 200 samples (20%)
- Train+Val: 800 samples (80%)
  - Val: 100 samples (10% of 1000)
  - Train: 700 samples (70% of 1000)

### Stratification Process

When `stratify=True`:

1. **Extract Labels**: Attempts to extract labels from each sample
   - Supports tuples: `(data, label)`
   - Supports dicts: `{'label': ...}` or `{'labels': ...}`
   - Falls back to random split if extraction fails

2. **First Split**: Stratified train+val vs test
   ```python
   train_val_indices, test_indices = train_test_split(
       indices,
       test_size=test_size,
       stratify=labels,
       random_state=42 + node_id
   )
   ```

3. **Second Split**: Stratified train vs val
   ```python
   train_indices, val_indices = train_test_split(
       train_val_indices,
       test_size=val_size,
       stratify=train_val_labels,
       random_state=42 + node_id
   )
   ```

4. **Reproducibility**: Uses `random_state=42 + node_id` for consistent splits per node

## Integration with Custom Datasets

### ESC50Dataset Example

```python
# Create stratified datasets
train_data = ESC50Dataset(
    selected_classes=['dog', 'cat', 'crow'],
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    node_id=0
)

val_data = ESC50Dataset(
    selected_classes=['dog', 'cat', 'crow'],
    split='val',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    node_id=0
)

test_data = ESC50Dataset(
    selected_classes=['dog', 'cat', 'crow'],
    split='test',
    split_ratio=0.8,
    stratify=True,
    node_id=0
)

# Wrap in NodeData
node_data = NodeData(
    args=args,
    node_id=0,
    custom_train_dataset=train_data,
    custom_val_dataset=val_data,
    custom_test_dataset=test_data
)
```

### VEGASDataset Example

```python
from system.datautils.dataset_vegas import VEGASDataset

# Create stratified VEGAS datasets
train_data = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='train',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True,
    load_audio=True,
    load_image=True,
    node_id=0
)

val_data = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='val',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True,
    load_audio=True,
    load_image=True,
    node_id=0
)

test_data = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='test',
    split_ratio=0.7,
    stratify=True,
    load_audio=True,
    load_image=True,
    node_id=0
)

# Wrap in NodeData
node_data = NodeData(
    args=args,
    node_id=0,
    custom_train_dataset=train_data,
    custom_val_dataset=val_data,
    custom_test_dataset=test_data
)
```

## Federated Learning Support

Each node can have different splits with consistent stratification:

```python
# Node 0
node0_data = NodeData(
    args=args,
    node_id=0,
    dataset_split_id=0,
    dataset=full_dataset,
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True
)

# Node 1
node1_data = NodeData(
    args=args,
    node_id=1,
    dataset_split_id=1,
    dataset=full_dataset,
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True
)

# Each node gets different samples but maintains class balance
print(node0_data)  # Node 0 split
print(node1_data)  # Node 1 split (different samples)
```

## Error Handling

### No Validation Data

If no validation data is available:

```python
val_loader = node_data.load_val_data(batch_size=32)
# Prints: "No validation dataset available for client X"
# Returns: None
```

### Stratification Failure

If stratification fails (e.g., cannot extract labels):

```python
# Automatically falls back to random split
# Prints: "Stratification failed (...), using random split"
```

## Best Practices

### 1. Always Use Stratification for Classification

```python
node_data = NodeData(
    args=args,
    dataset=dataset,
    stratify=True  # Recommended for balanced splits
)
```

### 2. Verify Split Statistics

```python
# Check class distribution
train_stats, train_percent = node_data.train_stats_get()
val_stats, val_percent = node_data.val_stats_get()
test_stats, test_percent = node_data.test_stats_get()

print("Train distribution:", train_percent)
print("Val distribution:", val_percent)
print("Test distribution:", test_percent)
```

### 3. Use Consistent node_id for Reproducibility

```python
# Same node_id = same splits
node_data1 = NodeData(args=args, node_id=0, dataset=dataset)
node_data2 = NodeData(args=args, node_id=0, dataset=dataset)
# node_data1 and node_data2 will have identical splits
```

### 4. Custom Datasets with Pre-Split Data

For maximum control, use pre-split custom datasets:

```python
# Split at dataset level
train_dataset = MyDataset(split='train', ...)
val_dataset = MyDataset(split='val', ...)
test_dataset = MyDataset(split='test', ...)

# No automatic splitting needed
node_data = NodeData(
    args=args,
    custom_train_dataset=train_dataset,
    custom_val_dataset=val_dataset,
    custom_test_dataset=test_dataset
)
```

## Migration Guide

### From Old Code (2-way split)

```python
# Old code
node_data = NodeData(
    args=args,
    node_id=0,
    custom_train_dataset=train_dataset,
    custom_test_dataset=test_dataset
)
```

### To New Code (3-way split)

```python
# New code - add validation
node_data = NodeData(
    args=args,
    node_id=0,
    custom_train_dataset=train_dataset,
    custom_val_dataset=val_dataset,  # New!
    custom_test_dataset=test_dataset
)

# Or use automatic splitting
node_data = NodeData(
    args=args,
    node_id=0,
    dataset=full_dataset,
    split_ratio=0.8,    # New!
    val_ratio=0.1,      # New!
    stratify=True       # New!
)
```

## Complete Example

```python
from system.datautils.node_dataset import NodeData
from system.datautils.dataset_esc50 import ESC50Dataset

# Create stratified ESC50 datasets
train_dataset = ESC50Dataset(
    selected_classes=['dog', 'cat', 'crow', 'frog', 'hen'],
    split='train',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True,
    use_folds=False,
    node_id=0,
    enable_cache=True
)

val_dataset = ESC50Dataset(
    selected_classes=['dog', 'cat', 'crow', 'frog', 'hen'],
    split='val',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True,
    use_folds=False,
    node_id=0,
    enable_cache=True
)

test_dataset = ESC50Dataset(
    selected_classes=['dog', 'cat', 'crow', 'frog', 'hen'],
    split='test',
    split_ratio=0.7,
    stratify=True,
    use_folds=False,
    node_id=0,
    enable_cache=True
)

# Verify stratification
print("=== Train Set ===")
train_dataset.print_split_statistics()

print("\n=== Validation Set ===")
val_dataset.print_split_statistics()

print("\n=== Test Set ===")
test_dataset.print_split_statistics()

# Check stratification quality
train_dataset.verify_stratification(val_dataset, tolerance=0.05)
train_dataset.verify_stratification(test_dataset, tolerance=0.05)

# Create NodeData
node_data = NodeData(
    args=args,
    node_id=0,
    dataset_split_id=0,
    custom_train_dataset=train_dataset,
    custom_val_dataset=val_dataset,
    custom_test_dataset=test_dataset
)

print(f"\n{node_data}")

# Create dataloaders
train_loader = node_data.load_train_data(batch_size=32)
val_loader = node_data.load_val_data(batch_size=32)
test_loader = node_data.load_test_data(batch_size=32)

# Training loop
for epoch in range(num_epochs):
    # Train
    model.train()
    for batch in train_loader:
        # Training code...
        pass

    # Validate
    model.eval()
    for batch in val_loader:
        # Validation code...
        pass

# Final evaluation
model.eval()
for batch in test_loader:
    # Test code...
    pass
```

## Summary

The updated `NodeData` class now provides:

✅ **Validation Split Support**: Automatic validation set creation with `val_ratio`
✅ **Stratification**: Class-balanced splits using sklearn's `train_test_split`
✅ **Flexible Ratios**: Control train/val/test split sizes independently
✅ **Reproducibility**: Consistent splits based on `node_id`
✅ **Backward Compatible**: Existing 2-way splits still work
✅ **Custom Dataset Support**: Works with ESC50Dataset, VEGASDataset, and any custom Dataset
✅ **Automatic Fallback**: Falls back to random split if stratification fails
✅ **Complete API**: `load_val_data()`, `get_val_dataset()`, `val_stats_get()`, `unload_val_data()`

These features enable proper validation during federated learning with balanced class distributions across all nodes and splits.
