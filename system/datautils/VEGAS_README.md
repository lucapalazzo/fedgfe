# VEGAS Dataset Implementation

Implementation of the VEGAS dataset loader for multimodal Audio2Visual tasks with support for federated learning.

## Features

- **Multimodal Data Loading**: Support for audio, image, and video modalities
- **Class Filtering**: Select or exclude specific classes
- **Stratified Splitting**: Balanced train/val/test splits preserving class distribution
- **Validation Split**: Automatic validation set creation from training data
- **Federated Learning Support**: Consistent node-specific data splits
- **Caching**: Optional disk caching for faster loading
- **Text Embeddings**: Support for precomputed text embeddings
- **Audio Embeddings**: Support for precomputed audio embeddings
- **Flexible Transforms**: Custom transforms for each modality

## Dataset Structure

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
└── captions.json (optional)
```

## Classes

VEGAS contains 10 classes:
- baby_cry (0)
- chainsaw (1)
- dog (2)
- drum (3)
- fireworks (4)
- helicopter (5)
- printer (6)
- rail_transport (7)
- snoring (8)
- water_flowing (9)

## Basic Usage

```python
from system.datautils.dataset_vegas import VEGASDataset, create_vegas_dataloader

# Load full dataset
dataset = VEGASDataset(
    root_dir="/path/to/VEGAS",
    split="train",
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    load_audio=True,
    load_image=True,
    load_video=False
)

# Create dataloader
dataloader = create_vegas_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate through data
for batch in dataloader:
    audio = batch['audio']        # [B, T]
    images = batch['image']        # [B, C, H, W]
    labels = batch['labels']       # [B]
    metadata = batch['metadata']   # dict with sample info
```

## Advanced Usage

### Class Selection

```python
# Select specific classes
dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='train'
)

# Exclude specific classes
dataset = VEGASDataset(
    excluded_classes=['snoring', 'printer'],
    split='train'
)

# Use class indices instead of names
dataset = VEGASDataset(
    selected_classes=[0, 1, 2],  # baby_cry, chainsaw, dog
    split='train'
)
```

### Train/Val/Test Splits with Stratification

```python
# Create stratified train/val/test splits
train_dataset = VEGASDataset(
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,  # Ensures balanced class distribution
    node_id=0
)

val_dataset = VEGASDataset(
    split='val',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    node_id=0
)

test_dataset = VEGASDataset(
    split='test',
    split_ratio=0.8,
    stratify=True,
    node_id=0
)

# Verify stratification
print("Train statistics:")
train_dataset.print_split_statistics()

print("\nValidation statistics:")
val_dataset.print_split_statistics()

# Check distributions are similar
is_stratified = train_dataset.verify_stratification(val_dataset, tolerance=0.05)
print(f"Stratification check: {'PASSED' if is_stratified else 'FAILED'}")
```

### Federated Learning Setup

```python
# Node 0 - gets first portion
node0_dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='train',
    split_ratio=0.8,
    stratify=True,
    node_id=0,
    enable_cache=True
)

# Node 1 - gets different portion
node1_dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='train',
    split_ratio=0.8,
    stratify=True,
    node_id=1,
    enable_cache=True
)

print(f"Node 0: {len(node0_dataset)} samples")
print(f"Node 1: {len(node1_dataset)} samples")

# Each node gets consistent splits across runs
node0_dataset.print_split_statistics()
node1_dataset.print_split_statistics()
```

### Working with Embeddings

```python
# Load with text embeddings
dataset = VEGASDataset(
    embedding_file="/path/to/vegas_text_embs_dict.pt",
    split='train'
)

# Access text embeddings
for sample in dataset:
    if 'text_emb' in sample:
        text_emb = sample['text_emb']  # Text embedding for the class

# Get text embeddings for specific class
text_emb = dataset.get_text_embeddings('dog')

# Load audio embeddings from file
dataset.load_audio_embeddings_from_file("/path/to/audio_embeddings.pt")
dataset.filter_audio_embeddings_from_file()
```

### Custom Transforms

```python
import torchaudio.transforms as AT
import torchvision.transforms as VT

# Audio transforms
audio_transform = AT.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)

# Image transforms
image_transform = VT.Compose([
    VT.Resize((224, 224)),
    VT.RandomHorizontalFlip(),
    VT.ToTensor(),
    VT.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

dataset = VEGASDataset(
    transform_audio=audio_transform,
    transform_image=image_transform,
    split='train'
)
```

### Caching

```python
# Enable caching for faster loading
dataset = VEGASDataset(
    split='train',
    enable_cache=True,
    cache_dir="/tmp/vegas_cache"
)

# First load: slow (loads from disk)
# Subsequent loads: fast (loads from cache)

# Clear cache if needed
dataset.clear_cache()
```

## Dataset Information Methods

```python
# Get class information
class_names = dataset.get_class_names()
class_labels = dataset.get_class_labels()
num_classes = dataset.get_num_classes()

# Get sample distribution
samples_per_class = dataset.get_samples_per_class()
# {'dog': 100, 'baby_cry': 95, 'chainsaw': 98}

distribution = dataset.get_class_distribution()
# {'dog': 33.4, 'baby_cry': 32.1, 'chainsaw': 34.5}

# Print comprehensive statistics
dataset.print_split_statistics()
```

## Batch Format

```python
batch = next(iter(dataloader))

# Tensors
batch['audio']      # [B, T] - Audio waveforms
batch['image']      # [B, C, H, W] - Images
batch['video']      # [B, T, C, H, W] - Videos (if load_video=True)
batch['labels']     # [B] - Class labels
batch['video_mask'] # [B] - Mask for valid videos

# Metadata dictionary
batch['metadata']['class_names']    # List of class names
batch['metadata']['video_ids']      # List of video IDs
batch['metadata']['sample_indices'] # List of sample indices
batch['metadata']['ytids']          # List of YouTube IDs
batch['metadata']['start_seconds']  # List of start times
batch['metadata']['end_seconds']    # List of end times
batch['metadata']['captions']       # List of captions (if available)
```

## Parameters Reference

### VEGASDataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_dir` | str | "/home/lpala/fedgfe/dataset/Audio/VEGAS" | Root directory of VEGAS dataset |
| `embedding_file` | str | "/home/lpala/fedgfe/dataset/Audio/vegas_text_embs_dict.pt" | Path to text embeddings |
| `audio_embedding_file` | str | None | Path to audio embeddings |
| `selected_classes` | List[str/int] | None | Classes to include |
| `excluded_classes` | List[str/int] | None | Classes to exclude |
| `split` | str | "all" | 'train', 'val', 'test', or 'all' |
| `split_ratio` | float | 0.8 | Train split ratio |
| `val_ratio` | float | 0.1 | Validation split ratio from train |
| `stratify` | bool | True | Use stratified sampling |
| `node_id` | int | None | Node ID for federated learning |
| `enable_cache` | bool | False | Enable sample caching |
| `cache_dir` | str | "/tmp/vegas_cache" | Cache directory |
| `audio_sample_rate` | int | 16000 | Audio sample rate |
| `audio_duration` | float | 10.0 | Audio duration in seconds |
| `image_size` | tuple | (224, 224) | Image size (H, W) |
| `video_fps` | int | 25 | Video frames per second |
| `transform_audio` | callable | None | Audio transform function |
| `transform_image` | callable | None | Image transform function |
| `transform_video` | callable | None | Video transform function |
| `load_audio` | bool | True | Whether to load audio |
| `load_image` | bool | False | Whether to load images |
| `load_video` | bool | False | Whether to load videos |

## Differences from ESC50Dataset

1. **Video Support**: VEGAS includes video files, ESC50 doesn't
2. **Duration**: VEGAS clips are 10 seconds (vs 5 for ESC50)
3. **Class Count**: VEGAS has 10 classes (vs 50 for ESC50)
4. **Structure**: VEGAS organizes files by class folders, ESC50 uses folds
5. **No Folds**: VEGAS doesn't have predefined fold splits like ESC50

## Stratification

The dataset supports stratified splitting to ensure that class distributions are preserved across train/validation/test sets:

```python
# Enable stratification
dataset = VEGASDataset(
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True  # Uses sklearn's train_test_split internally
)

# Verify stratification quality
train_dist = dataset.get_class_distribution()
# {'dog': 33.3%, 'baby_cry': 33.3%, 'chainsaw': 33.4%}

val_dataset = VEGASDataset(split='val', split_ratio=0.8, val_ratio=0.1, stratify=True)
val_dist = val_dataset.get_class_distribution()
# {'dog': 33.2%, 'baby_cry': 33.5%, 'chainsaw': 33.3%}

# Distributions should be similar (within tolerance)
is_stratified = dataset.verify_stratification(val_dataset, tolerance=0.05)
```

## Example: Complete Training Pipeline

```python
from system.datautils.dataset_vegas import VEGASDataset, create_vegas_dataloader

# Create stratified train/val/test splits
train_dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='train',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    load_audio=True,
    load_image=True,
    enable_cache=True,
    node_id=0
)

val_dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='val',
    split_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
    load_audio=True,
    load_image=True,
    enable_cache=True,
    node_id=0
)

test_dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='test',
    split_ratio=0.8,
    stratify=True,
    load_audio=True,
    load_image=True,
    enable_cache=True,
    node_id=0
)

# Print statistics
print("=== Training Set ===")
train_dataset.print_split_statistics()

print("\n=== Validation Set ===")
val_dataset.print_split_statistics()

print("\n=== Test Set ===")
test_dataset.print_split_statistics()

# Verify stratification
train_dataset.verify_stratification(val_dataset, tolerance=0.05)
train_dataset.verify_stratification(test_dataset, tolerance=0.05)

# Create dataloaders
train_loader = create_vegas_dataloader(train_dataset, batch_size=32, shuffle=True)
val_loader = create_vegas_dataloader(val_dataset, batch_size=32, shuffle=False)
test_loader = create_vegas_dataloader(test_dataset, batch_size=32, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        audio = batch['audio']
        images = batch['image']
        labels = batch['labels']
        # Train model...
```

## Notes

- Audio files are automatically resampled to target sample rate
- Images are automatically resized to target size
- Videos are sampled at uniform intervals to match target FPS
- Missing modalities return dummy tensors (-1)
- The `node_id` parameter ensures reproducible splits across runs
- Text embeddings should be a dictionary: `{class_name: embedding_tensor}`
- Audio embeddings should be a dictionary: `{audio_filename: {'embeddings': tensor, 'class_name': str}}`

## Troubleshooting

**Issue**: Samples not loading
- Check that root_dir points to correct VEGAS folder
- Verify class folders exist (baby_cry, chainsaw, etc.)
- Ensure audios/ and img/ subfolders exist in each class folder

**Issue**: Class distribution not balanced
- Enable stratification: `stratify=True`
- Verify using `print_split_statistics()` and `verify_stratification()`

**Issue**: Embeddings not loading
- Check embedding file path
- Ensure embedding keys match class names (lowercase)
- For audio embeddings, use `load_audio_embeddings_from_file()` then `filter_audio_embeddings_from_file()`

**Issue**: Cache issues
- Clear cache: `dataset.clear_cache()`
- Disable cache temporarily: `enable_cache=False`
