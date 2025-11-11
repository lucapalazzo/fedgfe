# ESC50Dataset vs VEGASDataset - Feature Comparison

Comprehensive comparison of the ESC50 and VEGAS dataset implementations.

## Overview

Both datasets implement multimodal audio-visual data loading with support for federated learning, but have different characteristics and use cases.

## Quick Comparison Table

| Feature | ESC50Dataset | VEGASDataset |
|---------|--------------|--------------|
| **Classes** | 50 | 10 |
| **Total Samples** | 2000 | Varies by class |
| **Audio Duration** | 5 seconds | 10 seconds |
| **Modalities** | Audio, Image | Audio, Image, Video |
| **Fold Structure** | Yes (5 folds) | No |
| **Video Support** | No | Yes |
| **Stratification** | Yes | Yes |
| **Validation Split** | Yes | Yes |
| **Text Embeddings** | Yes | Yes |
| **Audio Embeddings** | Yes | Yes |
| **Caching** | Yes | Yes |
| **Federated Learning** | Yes | Yes |

## Detailed Feature Comparison

### 1. Dataset Organization

#### ESC50Dataset
```
esc50-v2.0.0-full/
├── audio/
│   ├── fold00/
│   │   ├── 1-100032-A-0.wav
│   │   └── ...
│   ├── fold01/
│   └── ...
├── image/
│   ├── dog_0.png
│   ├── dog_1.png
│   └── ...
├── fold00.json
├── fold01.json
├── class_labels.json
└── captions.json
```

- **Organized by folds**: 5 official folds for cross-validation
- **Separate audio/image directories**: Audio organized by fold, images by class
- **JSON metadata**: Fold assignments in JSON files

#### VEGASDataset
```
VEGAS/
├── baby_cry/
│   ├── audios/
│   │   ├── video_00000.wav
│   │   └── ...
│   ├── img/
│   │   ├── video_00000.jpg
│   │   └── ...
│   ├── videos/
│   │   ├── video_00000.mp4
│   │   └── ...
│   └── video_info.csv
├── chainsaw/
│   └── ...
└── captions.json
```

- **Organized by class folders**: Each class has its own directory
- **Multimodal structure**: Audio, image, video in separate subdirectories
- **CSV metadata**: video_info.csv for additional information

### 2. Split Strategies

#### ESC50Dataset
```python
# Option 1: Use official folds
dataset = ESC50Dataset(
    use_folds=True,
    train_folds=[0, 1, 2, 3],
    val_folds=None,  # Will split from train_folds
    test_folds=[4],
    stratify=True
)

# Option 2: Custom split
dataset = ESC50Dataset(
    use_folds=False,
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True
)
```

**Advantages**:
- Official folds for reproducible research
- Can use validation folds if needed
- Standard benchmark comparison

#### VEGASDataset
```python
# Only custom splits available
dataset = VEGASDataset(
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True
)
```

**Advantages**:
- Simpler configuration
- Flexible split ratios
- Automatic validation set creation

### 3. Modality Support

#### ESC50Dataset
```python
dataset = ESC50Dataset(
    load_audio=True,
    load_image=True
)

# Returns
sample = {
    'audio': Tensor[T],          # Audio waveform
    'image': Tensor[C, H, W],    # Image
    'label': int,
    'class_name': str,
    # ... metadata
}
```

**Supported Modalities**:
- Audio (WAV files, 5 seconds)
- Image (PNG files, generated spectrograms/visualizations)

#### VEGASDataset
```python
dataset = VEGASDataset(
    load_audio=True,
    load_image=True,
    load_video=True
)

# Returns
sample = {
    'audio': Tensor[T],             # Audio waveform
    'image': Tensor[C, H, W],       # Single frame
    'video': Tensor[T, C, H, W],    # Video sequence
    'label': int,
    'class_name': str,
    # ... metadata
}
```

**Supported Modalities**:
- Audio (WAV files, 10 seconds)
- Image (JPG files, key frames)
- Video (MP4 files, full sequences)

### 4. Stratification

Both datasets support stratification using scikit-learn's `train_test_split`:

#### ESC50Dataset
```python
train_dataset = ESC50Dataset(
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,  # Maintains class balance
    use_folds=False
)

val_dataset = ESC50Dataset(
    split='val',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    use_folds=False
)

# Verify stratification
train_dataset.verify_stratification(val_dataset, tolerance=0.05)
```

#### VEGASDataset
```python
train_dataset = VEGASDataset(
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True  # Maintains class balance
)

val_dataset = VEGASDataset(
    split='val',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True
)

# Verify stratification
train_dataset.verify_stratification(val_dataset, tolerance=0.05)
```

**Both provide**:
- Stratified sampling to preserve class distribution
- Verification methods to check stratification quality
- Statistics printing for analysis

### 5. Common Features

Both datasets share these features:

#### Class Filtering
```python
# Select specific classes
esc50_dataset = ESC50Dataset(
    selected_classes=['dog', 'cat', 'crow']
)

vegas_dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw']
)

# Exclude specific classes
esc50_dataset = ESC50Dataset(
    excluded_classes=['rain', 'wind']
)

vegas_dataset = VEGASDataset(
    excluded_classes=['snoring', 'printer']
)
```

#### Text Embeddings
```python
# Both support text embeddings
esc50_dataset = ESC50Dataset(
    text_embedding_file="esc50_text_embs_dict.pt"
)

vegas_dataset = VEGASDataset(
    embedding_file="vegas_text_embs_dict.pt"
)

# Access embeddings
text_emb = dataset.get_text_embeddings('dog')
```

#### Audio Embeddings
```python
# ESC50
esc50_dataset = ESC50Dataset(
    audio_embedding_file="esc50_audio_embs.pt"
)

# VEGAS
vegas_dataset = VEGASDataset(
    audio_embedding_file="vegas_audio_embs.pt"
)
vegas_dataset.load_audio_embeddings_from_file("path.pt")
vegas_dataset.filter_audio_embeddings_from_file()
```

#### Caching
```python
# Both support caching
dataset = ESC50Dataset(
    enable_cache=True,
    cache_dir="/tmp/esc50_cache"
)

dataset = VEGASDataset(
    enable_cache=True,
    cache_dir="/tmp/vegas_cache"
)

# Clear cache
dataset.clear_cache()
```

#### Federated Learning
```python
# Node-specific splits (both datasets)
node0_dataset = ESC50Dataset(
    split='train',
    node_id=0,
    stratify=True
)

node0_dataset = VEGASDataset(
    split='train',
    node_id=0,
    stratify=True
)
```

### 6. Utility Methods

Both datasets provide the same utility methods:

```python
# Get class information
class_names = dataset.get_class_names()
class_labels = dataset.get_class_labels()
num_classes = dataset.get_num_classes()

# Get distribution
samples_per_class = dataset.get_samples_per_class()
distribution = dataset.get_class_distribution()

# Print statistics
dataset.print_split_statistics()

# Verify stratification
dataset1.verify_stratification(dataset2, tolerance=0.05)

# Clear cache
dataset.clear_cache()
```

### 7. DataLoader Collate Functions

#### ESC50Dataset Batch Format
```python
batch = {
    'audio': Tensor[B, T],
    'image': Tensor[B, C, H, W],
    'labels': Tensor[B],
    'text_embs': Tensor[B, D],      # Optional
    'audio_embs': Tensor[B, D],     # Optional
    'metadata': {
        'class_names': List[str],
        'file_ids': List[str],
        'sample_indices': List[int],
        'folds': List[int],
        'captions': List[str]
    }
}
```

#### VEGASDataset Batch Format
```python
batch = {
    'audio': Tensor[B, T],
    'image': Tensor[B, C, H, W],
    'video': Tensor[B, T, C, H, W],  # Optional
    'video_mask': Tensor[B],         # Validity mask
    'labels': Tensor[B],
    'metadata': {
        'class_names': List[str],
        'video_ids': List[str],
        'sample_indices': List[int],
        'ytids': List[str],
        'start_seconds': List[float],
        'end_seconds': List[float],
        'captions': List[str]
    }
}
```

### 8. Use Case Recommendations

#### Use ESC50Dataset when:
- You need the standard ESC-50 benchmark
- You want to use official fold splits
- You need environmental sound classification
- You want to compare with published results
- You have 50 diverse sound classes
- You only need audio and images

#### Use VEGASDataset when:
- You need video modality support
- You're working with longer clips (10 seconds)
- You need fewer, specific sound classes
- You want more flexible data organization
- You need audio-visual-video multimodal learning
- You're building custom benchmarks

## Code Migration Guide

### Converting ESC50 code to VEGAS

```python
# ESC50 code
esc50_dataset = ESC50Dataset(
    root_dir="/path/to/esc50",
    text_embedding_file="esc50_text_embs.pt",
    selected_classes=['dog', 'cat'],
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    use_folds=False,
    stratify=True,
    node_id=0,
    enable_cache=True
)

# Equivalent VEGAS code
vegas_dataset = VEGASDataset(
    root_dir="/path/to/vegas",
    embedding_file="vegas_text_embs.pt",
    selected_classes=['dog', 'baby_cry'],
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    node_id=0,
    enable_cache=True,
    load_video=False  # New parameter for video
)
```

**Key Differences**:
1. Remove `use_folds`, `train_folds`, `test_folds` parameters
2. Change `text_embedding_file` to `embedding_file`
3. Add `load_video` parameter if needed
4. Adjust class names to match VEGAS classes
5. Consider longer audio duration (10s vs 5s)

### Converting VEGAS code to ESC50

```python
# VEGAS code
vegas_dataset = VEGASDataset(
    root_dir="/path/to/vegas",
    embedding_file="vegas_text_embs.pt",
    selected_classes=['dog', 'baby_cry'],
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    load_video=True
)

# Equivalent ESC50 code
esc50_dataset = ESC50Dataset(
    root_dir="/path/to/esc50",
    text_embedding_file="esc50_text_embs.pt",
    selected_classes=['dog', 'baby_crying'],  # Note: different class names
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    use_folds=False,  # New parameter
    stratify=True
)
# Note: No video support in ESC50
```

**Key Differences**:
1. Change `embedding_file` to `text_embedding_file`
2. Add `use_folds=False` for custom splits
3. Remove `load_video` parameter
4. Adjust class names to match ESC50 classes
5. Consider shorter audio duration (5s vs 10s)

## Performance Considerations

### ESC50Dataset
- **Pros**:
  - Smaller files (5-second audio)
  - Faster loading per sample
  - Well-established benchmark
  - Many pre-computed features available
- **Cons**:
  - Limited to 2 modalities
  - Fixed fold structure
  - Shorter context window

### VEGASDataset
- **Pros**:
  - 3 modalities (audio, image, video)
  - Longer context (10 seconds)
  - Flexible organization
  - Good for video-based learning
- **Cons**:
  - Larger files (especially videos)
  - Slower loading with videos
  - Requires more storage
  - Fewer pre-computed features

## Best Practices

### For Both Datasets

1. **Enable Caching** for repeated experiments:
```python
dataset = Dataset(enable_cache=True, cache_dir="/fast/disk/cache")
```

2. **Use Stratification** for balanced splits:
```python
dataset = Dataset(split='train', stratify=True, val_ratio=0.1)
```

3. **Verify Splits** before training:
```python
train_dataset.print_split_statistics()
train_dataset.verify_stratification(val_dataset)
```

4. **Consistent node_id** for reproducibility:
```python
dataset = Dataset(node_id=42, split='train')
```

5. **Select Modalities** to optimize loading:
```python
# Only load what you need
dataset = Dataset(load_audio=True, load_image=False, load_video=False)
```

## Summary

Both `ESC50Dataset` and `VEGASDataset` provide:
- ✓ Stratified train/val/test splits
- ✓ Class filtering and selection
- ✓ Text and audio embeddings support
- ✓ Federated learning support
- ✓ Caching for performance
- ✓ Comprehensive utility methods
- ✓ Flexible data loading

Choose based on your specific needs:
- **ESC50**: Standard benchmark, audio+image, 50 classes
- **VEGAS**: Video support, audio+image+video, 10 classes, longer clips

Both implementations follow the same design patterns, making it easy to switch between them or use both in the same project.
