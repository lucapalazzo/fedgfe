# VEGAS Dataset - Quick Reference Guide

## 🚀 Quick Start

### Basic Usage (Recommended)
```python
from system.datautils.dataset_vegas import VEGASDataset
from torch.utils.data import DataLoader

# Auto-create train/val/test splits
dataset = VEGASDataset(
    split=None,  # Auto-creates .train, .val, .test
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Create dataloaders
train_loader = DataLoader(dataset.train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset.val, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset.test, batch_size=32, shuffle=False)
```

---

## 📋 Common Patterns

### 1. Standard Training with Validation
```python
dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split=None,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True
)

for epoch in range(num_epochs):
    train(dataset.train)
    validate(dataset.val)

evaluate(dataset.test)
```

### 2. Custom Split Ratios
```python
# 60% train, 20% val, 20% test
dataset = VEGASDataset(
    split=None,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2
)
```

### 3. Fine-tuning (Combine Train + Val)
```python
# Use train+val for fine-tuning
finetune_ds = VEGASDataset(
    splits_to_load=['train', 'val'],
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Keep test separate
test_ds = VEGASDataset(split='test', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### 4. No Validation (Train/Test Only)
```python
dataset = VEGASDataset(
    split=None,
    train_ratio=0.8,
    val_ratio=0.0,  # No validation
    test_ratio=0.2
)

train_loader = DataLoader(dataset.train, ...)
test_loader = DataLoader(dataset.test, ...)
```

### 5. Federated Learning
```python
# Different nodes get reproducible but different splits
node0_train = VEGASDataset(split='train', node_id=0, train_ratio=0.8, test_ratio=0.2)
node1_train = VEGASDataset(split='train', node_id=1, train_ratio=0.8, test_ratio=0.2)

# Same test set for all nodes
test_ds = VEGASDataset(split='test', node_id=0, train_ratio=0.8, test_ratio=0.2)
```

---

## 🎯 Parameters Quick Reference

### Essential Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `split` | `None` | `None`, `'train'`, `'val'`, `'test'`, `'all'` |
| `train_ratio` | `0.7` | Training data percentage (70%) |
| `val_ratio` | `0.1` | Validation data percentage (10%) |
| `test_ratio` | `0.2` | Test data percentage (20%) |
| `stratify` | `True` | Use stratified sampling |

### Advanced Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `splits_to_load` | `None` | Combine splits: `['train', 'val']` |
| `node_id` | `None` | Federated learning node identifier |
| `use_folds` | `False` | Use fold-based splitting |
| `selected_classes` | `None` | Filter specific classes |
| `excluded_classes` | `None` | Exclude specific classes |

### Data Loading Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `load_audio` | `True` | Load audio data |
| `load_image` | `False` | Load image data |
| `load_video` | `False` | Load video data |

---

## 🔧 Utility Methods

### Statistics
```python
# Print split statistics
dataset.train.print_split_statistics()

# Get samples per class
samples_per_class = dataset.train.get_samples_per_class()

# Get class distribution (percentages)
distribution = dataset.train.get_class_distribution()
```

### Verification
```python
# Verify stratification between splits
dataset.train.verify_stratification(dataset.val, tolerance=0.05)
dataset.train.verify_stratification(dataset.test, tolerance=0.05)
```

### Class Information
```python
# Get class names
class_names = dataset.get_class_names()

# Get number of classes
num_classes = dataset.get_num_classes()

# Get class labels
class_labels = dataset.get_class_labels()
```

---

## ⚠️ Common Pitfalls

### 1. Ratios Don't Sum to 1.0
```python
# ❌ Wrong: Will auto-normalize with warning
dataset = VEGASDataset(train_ratio=0.6, val_ratio=0.3, test_ratio=0.3)  # Sum = 1.2

# ✅ Correct
dataset = VEGASDataset(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)  # Sum = 1.0
```

### 2. Inconsistent Splits Across Datasets
```python
# ❌ Wrong: Different node_id will give different splits
train_ds = VEGASDataset(split='train', node_id=0, train_ratio=0.8)
test_ds = VEGASDataset(split='test', node_id=1, train_ratio=0.8)  # Different node_id!

# ✅ Correct: Use same node_id for consistent splits
train_ds = VEGASDataset(split='train', node_id=0, train_ratio=0.8)
test_ds = VEGASDataset(split='test', node_id=0, train_ratio=0.8)
```

### 3. Manual Split Creation vs Auto-Split
```python
# ❌ Verbose: Manual creation
train = VEGASDataset(split='train', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
val = VEGASDataset(split='val', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
test = VEGASDataset(split='test', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# ✅ Better: Auto-creation
dataset = VEGASDataset(split=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
# Access via dataset.train, dataset.val, dataset.test
```

---

## 🎓 Cheat Sheet

### Scenario → Code

| Scenario | Code |
|----------|------|
| Standard 70-10-20 split | `VEGASDataset(split=None)` |
| Custom 60-20-20 split | `VEGASDataset(split=None, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)` |
| No validation | `VEGASDataset(split=None, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2)` |
| Fine-tuning (train+val) | `VEGASDataset(splits_to_load=['train', 'val'], ...)` |
| Federated node 0 | `VEGASDataset(split='train', node_id=0, ...)` |
| Only dog and cat classes | `VEGASDataset(selected_classes=['dog', 'cat'], ...)` |
| Exclude noisy classes | `VEGASDataset(excluded_classes=['chainsaw', 'printer'], ...)` |
| Load all modalities | `VEGASDataset(load_audio=True, load_image=True, load_video=True)` |

---

## 🆚 Old API vs New API

### Split Creation
```python
# Old API (still works)
train = VEGASDataset(split='train', split_ratio=0.8, val_ratio=0.1)
val = VEGASDataset(split='val', split_ratio=0.8, val_ratio=0.1)
test = VEGASDataset(split='test', split_ratio=0.8, val_ratio=0.1)

# New API (recommended)
ds = VEGASDataset(split=None, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
train, val, test = ds.train, ds.val, ds.test
```

### Custom Ratios
```python
# Old API (deprecated but works)
train = VEGASDataset(split='train', split_ratio=0.8, val_ratio=0.1)
# split_ratio=0.8 means 80% for train+val, 20% for test

# New API (clearer)
train = VEGASDataset(split='train', train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
# Explicit percentages for each split
```

---

## 📊 Example: Complete Training Loop

```python
from system.datautils.dataset_vegas import VEGASDataset, create_vegas_dataloader
from torch.utils.data import DataLoader

# 1. Create dataset with auto-splits
dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split=None,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True,
    load_audio=True,
    load_image=False
)

# 2. Create dataloaders
train_loader = create_vegas_dataloader(dataset.train, batch_size=32, shuffle=True)
val_loader = create_vegas_dataloader(dataset.val, batch_size=32, shuffle=False)
test_loader = create_vegas_dataloader(dataset.test, batch_size=32, shuffle=False)

# 3. Print statistics
print(f"Train: {len(dataset.train)} samples")
print(f"Val: {len(dataset.val)} samples")
print(f"Test: {len(dataset.test)} samples")

dataset.train.print_split_statistics()

# 4. Verify stratification
dataset.train.verify_stratification(dataset.val, tolerance=0.05)

# 5. Training loop
for epoch in range(num_epochs):
    # Train
    for batch in train_loader:
        audio = batch['audio']
        labels = batch['labels']
        # ... training code ...

    # Validate
    for batch in val_loader:
        audio = batch['audio']
        labels = batch['labels']
        # ... validation code ...

# 6. Final evaluation
for batch in test_loader:
    audio = batch['audio']
    labels = batch['labels']
    # ... test code ...
```

---

## 🔍 Debugging Tips

### Check Split Sizes
```python
dataset = VEGASDataset(split=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

total = len(dataset.train) + len(dataset.val) + len(dataset.test)
print(f"Train: {len(dataset.train)/total*100:.1f}%")
print(f"Val: {len(dataset.val)/total*100:.1f}%")
print(f"Test: {len(dataset.test)/total*100:.1f}%")
```

### Verify Stratification
```python
dataset.train.print_split_statistics()
dataset.val.print_split_statistics()
dataset.test.print_split_statistics()

dataset.train.verify_stratification(dataset.val, tolerance=0.10)
```

### Check Data Loading
```python
sample = dataset.train[0]
print("Sample keys:", sample.keys())
print("Audio shape:", sample['audio'].shape)
print("Label:", sample['label'], "Class:", sample['class_name'])
```

---

## 📚 More Information

- **Full Documentation**: See [VEGAS_ESC50_FEATURES.md](VEGAS_ESC50_FEATURES.md)
- **Implementation Details**: See [VEGAS_IMPLEMENTATION_SUMMARY.md](VEGAS_IMPLEMENTATION_SUMMARY.md)
- **Usage Examples**: See [example_vegas_esc50_usage.py](system/datautils/example_vegas_esc50_usage.py)
- **Changelog**: See [VEGAS_CHANGELOG.md](VEGAS_CHANGELOG.md)
