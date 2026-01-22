# VGGSound Dataset Implementation

This document describes the VGGSound dataset integration into the federated Audio2Visual framework.

## Overview

VGGSound is a large-scale audio-visual dataset containing 200k+ clips spanning 300+ sound classes. This implementation provides full integration with the existing federated learning infrastructure, matching the functionality of ESC-50 and VEGAS datasets.

## Files Created/Modified

### New Files

1. **[system/datautils/dataset_vggsound.py](system/datautils/dataset_vggsound.py)**
   - Complete VGGSound dataset loader class
   - Supports audio, video, and image loading
   - Includes train/val/test splitting with stratification
   - Compatible with PyTorch DataLoader

2. **[system/datautils/example_vggsound_usage.py](system/datautils/example_vggsound_usage.py)**
   - 8 comprehensive usage examples
   - Demonstrates all key features

3. **[configs/a2v_generator_vggsound_1n_all_classes.json](configs/a2v_generator_vggsound_1n_all_classes.json)**
   - Configuration for training VAE generators on all 50 VGGSound classes
   - Single node setup

4. **[configs/a2v_generator_vggsound_10n_1c.json](configs/a2v_generator_vggsound_10n_1c.json)**
   - Configuration for distributed training (10 nodes, 1 class each)
   - Similar to Vegas setup

### Modified Files

1. **[system/flcore/servers/serverA2V.py](system/flcore/servers/serverA2V.py)**
   - Added VGGSoundDataset import (line 44)
   - Added VGGSound dataset loading logic (lines 3384-3431)
   - Supports audio embedding caching and loading

## Dataset Structure

```
dataset/Audio/vggsound/
├── train/
│   ├── audios/          # 174,576 .wav files (10s @ 16kHz)
│   └── video/           # Corresponding .mp4 files
├── test/
│   ├── audios/          # 14,600 .wav files
│   └── video/           # Corresponding .mp4 files
└── vggsound_text_embs_dict.pt  # Text embeddings (50 classes)
```

## Key Features

### 1. **Multimodal Loading**
```python
dataset = VGGSoundDataset(
    load_audio=True,   # Load audio waveforms
    load_video=True,   # Load video frames
    load_image=True    # Extract single frame from video
)
```

### 2. **Automatic Splits**
```python
# Auto-create train/val/test splits
dataset = VGGSoundDataset(
    split=None,          # Triggers auto-split
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True
)

# Access splits
print(len(dataset.train))  # 70% of data
print(len(dataset.val))    # 10% of data
print(len(dataset.test))   # 20% of data
```

### 3. **Official Train/Test Splits**
```python
# Use dataset's official splits
train_dataset = VGGSoundDataset(
    split='train',
    use_official_split=True  # Use official train folder
)

test_dataset = VGGSoundDataset(
    split='test',
    use_official_split=True  # Use official test folder
)
```

### 4. **Class Filtering**
```python
dataset = VGGSoundDataset(
    selected_classes=['dog barking', 'baby crying'],
    excluded_classes=['airplane flyby']
)
```

### 5. **Sample Limiting**
```python
dataset = VGGSoundDataset(
    num_samples_per_class=100  # Limit to 100 samples per class
)
```

### 6. **Text/Audio Embeddings**
```python
dataset = VGGSoundDataset(
    text_embedding_file="vggsound_text_embs_dict.pt",
    audio_embedding_file="vggsound_audio_embs_dict.pt"
)

sample = dataset[0]
print(sample['text_emb'].shape)   # Class text embedding
print(sample['audio_emb'].shape)  # Per-sample audio embedding
```

## Configuration Examples

### Single Node - All Classes
```json
{
  "nodes": {
    "0": {
      "dataset": "VGGSound",
      "selected_classes": [
        "dog barking",
        "baby crying",
        "chainsawing trees",
        // ... all 50 classes
      ],
      "diffusion_type": "flux"
    }
  }
}
```

### Multi-Node - Distributed
```json
{
  "federation": {
    "num_clients": 10
  },
  "nodes": {
    "0": {"dataset": "VGGSound", "selected_classes": ["dog barking"]},
    "1": {"dataset": "VGGSound", "selected_classes": ["baby crying"]},
    // ... one class per node
  }
}
```

## Available Classes (50)

The current VGGSound text embeddings contain 50 classes:

- **Animals**: dog barking, baby crying, cat purring, cow lowing, sheep bleating, owl hooting, etc.
- **Vehicles**: airplane flyby, train horning, ambulance siren, car engine idling, etc.
- **Music**: playing drum kit, playing harp, orchestra, singing choir, playing bassoon, etc.
- **Nature**: waterfall burbling, sea waves, volcano explosion, hail, stream burbling, etc.
- **Human**: people cheering, people crowd, people marching, scuba diving, skiing, etc.

Full list available in: [vggsound_class_names.txt](vggsound_class_names.txt)

## Usage in Training

### Running Generator Training

```bash
# Train on all 50 classes (1 node)
python main.py --config configs/a2v_generator_vggsound_1n_all_classes.json

# Train on 10 classes (10 nodes, federated)
python main.py --config configs/a2v_generator_vggsound_10n_1c.json
```

### Programmatic Usage

```python
from datautils.dataset_vggsound import VGGSoundDataset, create_vggsound_dataloader

# Create dataset
dataset = VGGSoundDataset(
    root_dir="/path/to/vggsound",
    text_embedding_file="vggsound_text_embs_dict.pt",
    split='train',
    selected_classes=['dog barking', 'baby crying']
)

# Create DataLoader
dataloader = create_vggsound_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# Iterate over batches
for batch in dataloader:
    audio = batch['audio']          # [B, 160000] - 10s @ 16kHz
    labels = batch['labels']        # [B]
    class_names = batch['class_name']  # List[str]

    if 'text_emb' in batch:
        text_emb = batch['text_emb']  # List of embeddings
```

## Important Notes

### ⚠️ Metadata Requirement

The current implementation uses a **placeholder** for class assignment because VGGSound requires metadata files (CSV/JSON) mapping filenames to class labels.

**For production use**, you need to:

1. Create metadata files (e.g., `vggsound_train_metadata.csv`, `vggsound_test_metadata.csv`)
2. Format:
   ```csv
   filename,class_name,ytid,start_time
   0-00uubiDNU_000210.wav,dog_barking,0-00uubiDNU,210
   ```
3. Modify `_load_samples_from_split()` in `dataset_vggsound.py` to read from metadata

### Memory Considerations

- **Audio**: 10 seconds @ 16kHz = 160,000 samples per file
- **Video**: ~250 frames @ 25fps for 10-second clips
- Use `num_samples_per_class` to limit dataset size during development

### Caching

Enable caching for faster repeated loading:

```python
dataset = VGGSoundDataset(
    enable_cache=True,
    cache_dir="/tmp/vggsound_cache"
)
```

## API Compatibility

VGGSoundDataset follows the same API as ESC50Dataset and VEGASDataset:

| Method | Description |
|--------|-------------|
| `__len__()` | Returns number of samples |
| `__getitem__(idx)` | Returns sample dict with audio, labels, embeddings, etc. |
| `get_class_names()` | Returns list of class names |
| `get_num_classes()` | Returns number of classes |
| `get_samples_per_class()` | Returns dict of sample counts per class |
| `get_class_distribution()` | Returns class distribution as percentages |
| `print_split_statistics()` | Prints dataset statistics |
| `verify_stratification()` | Verifies stratified splitting |
| `clear_cache()` | Clears cached data |
| `to(device)` | Moves tensors to device |

## Testing

Run the example script to verify installation:

```bash
cd system/datautils
python example_vggsound_usage.py
```

This will run 8 comprehensive examples demonstrating all features.

## Integration with Server

The server automatically detects VGGSound datasets in node configurations:

```python
# In serverA2V.py (lines 3384-3431)
elif node_config.dataset == "VGGSound":
    node_dataset = VGGSoundDataset(
        selected_classes=selected_classes,
        num_samples_per_class=num_samples_per_class,
        # ... other parameters
    )
```

Supports:
- Audio embedding caching across nodes
- Text embedding loading
- Official and custom train/val/test splits
- Per-node class filtering

## Future Enhancements

1. **Metadata Integration**: Add CSV/JSON metadata parsing for proper class assignment
2. **Full Dataset Support**: Extend to all 300+ VGGSound classes
3. **Video Processing**: Add video preprocessing pipelines (currently basic frame extraction)
4. **Audio Augmentation**: Add spectral augmentation, time stretching, pitch shifting
5. **Multi-modal Embeddings**: Support for joint audio-visual embeddings

## Troubleshooting

### Issue: "VGGSoundDataset not found"
- Ensure `from datautils.dataset_vggsound import VGGSoundDataset` is in your imports
- Check `system/datautils/dataset_vggsound.py` exists

### Issue: "No samples loaded"
- Verify dataset path is correct: `/home/lpala/fedgfe/dataset/Audio/vggsound`
- Check that `train/audios/` and `test/audios/` folders exist and contain .wav files
- Ensure text embeddings file exists

### Issue: "Class name not found"
- Class names use underscores: `dog_barking` not `dog barking`
- Check available classes in `vggsound_class_names.txt`
- Text embeddings may use different naming (spaces vs underscores)

## References

- VGGSound Dataset: https://www.robots.ox.ac.uk/~vgg/data/vggsound/
- Paper: "VGGSound: A Large-scale Audio-Visual Dataset"
- Related: ESC-50, VEGAS datasets in same framework

---

**Last Updated**: 2026-01-10
**Contributors**: Implementation for federated Audio2Visual learning framework
