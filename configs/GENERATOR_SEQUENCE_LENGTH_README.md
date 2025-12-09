# Generator Sequence Length Configuration

## Overview

The generator now supports separate sequence lengths for training and generation/output, allowing efficient training with low memory usage while still generating full-length sequences when needed.

## Configuration Parameters

### `generator_training_sequence_length` (default: 4)
- **Purpose**: Sequence length used during generator training
- **Memory Impact**: Smaller values = less GPU memory usage
- **Recommended Values**: 4, 8, or 16
- **When to Adjust**:
  - Use **4** for minimal memory (recommended for most cases)
  - Use **8** if you have more GPU memory and want slightly better quality
  - Use **16** only if you have abundant GPU memory

### `generator_output_sequence_length` (default: None)
- **Purpose**: Target sequence length for generated samples
- **Upsampling**: Automatically upsampled from training length using interpolation
- **Recommended Values**:
  - `null` or omit - uses training sequence length (no upsampling)
  - `1214` - full AST output sequence length
  - Any other desired length

## How It Works

1. **During Training**:
   - Audio embeddings are reduced to `generator_training_sequence_length` (e.g., 4 tokens)
   - Generator is trained on these compact representations
   - Fits easily in GPU memory even with batch processing

2. **During Generation** (when using pre-trained generators):
   - Generator produces compact latent representations
   - Output is upsampled to `generator_output_sequence_length` (e.g., 1214 tokens)
   - Uses linear interpolation for smooth upsampling

## Example Configurations

### Minimal Memory Configuration (Training Mode)
```json
{
  "feda2v_config": {
    "generator_training_mode": true,
    "generator_type": "vae",
    "use_conditioned_vae": false,
    "generator_training_sequence_length": 4,
    "generator_output_sequence_length": null,
    "generator_training_epochs": 50
  }
}
```

### Full Sequence Generation (Inference Mode)
```json
{
  "feda2v_config": {
    "generator_only_mode": true,
    "generator_type": "vae",
    "use_conditioned_vae": false,
    "generator_training_sequence_length": 4,
    "generator_output_sequence_length": 1214,
    "pretrained_generator_path": "checkpoints/generator_round_10.pt",
    "synthetic_samples_per_class": 10
  }
}
```

### Balanced Configuration (Training with Higher Resolution)
```json
{
  "feda2v_config": {
    "generator_training_mode": true,
    "generator_type": "vae",
    "use_conditioned_vae": false,
    "generator_training_sequence_length": 8,
    "generator_output_sequence_length": null,
    "generator_training_epochs": 100
  }
}
```

## Memory Requirements (Approximate)

For VAEGenerator with `input_dim=768, hidden_dim=1024, latent_dim=256`:

| Training Seq Length | Model Parameters | Training Memory | Generation Memory |
|---------------------|------------------|-----------------|-------------------|
| 4                   | ~3.1M params     | ~2-3 GB         | ~1-2 GB          |
| 8                   | ~5.3M params     | ~4-6 GB         | ~2-3 GB          |
| 16                  | ~9.7M params     | ~8-12 GB        | ~3-4 GB          |
| 1214 (full)         | ~900M params     | **>40 GB** ❌    | **>20 GB** ❌     |

**Note**: The 1214 sequence length is **NOT recommended for training** - it requires massive GPU memory and doesn't provide significant quality benefits due to the VAE's latent compression.

## ⚠️ IMPORTANT: Information Loss Considerations

**Compression ratios matter!** Reducing from 1214 → 4 tokens means 303:1 compression, which causes **significant information loss** (~75-85%). This can hurt downstream task performance.

### Recommended Sequence Lengths

| Sequence Length | Compression | Info Preserved | Memory Required | Recommendation |
|-----------------|-------------|----------------|-----------------|----------------|
| **32** | 38:1 | ~70-80% | 6-8 GB | ✅ **RECOMMENDED** |
| 16 | 76:1 | ~50-60% | 4-5 GB | ⚠️ Moderate loss |
| 8 | 152:1 | ~30-40% | 3-4 GB | ⚠️ Significant loss |
| 4 | 303:1 | ~15-25% | 2-3 GB | ❌ Major loss (use only if necessary) |
| 64 | 19:1 | ~85-90% | 10-14 GB | ✅ High quality (high-end GPUs) |

See [GENERATOR_INFORMATION_PRESERVATION.md](../GENERATOR_INFORMATION_PRESERVATION.md) for detailed analysis.

## Best Practices

1. **For Most Use Cases** (RECOMMENDED):
   - Train with `generator_training_sequence_length: 32`
   - Generate with `generator_output_sequence_length: 1214`
   - This preserves ~70-80% of information with reasonable memory usage

2. **If You Have Memory Constraints** (<8 GB GPU):
   - Use `generator_training_sequence_length: 16` (50-60% info preserved)
   - Or `generator_training_sequence_length: 8` (30-40% info preserved)
   - Reduce `hidden_dim` to 512 if still OOM
   - Reduce `batch_size` for training
   - ⚠️ **Avoid seq_len=4** unless absolutely necessary

3. **If You Want Maximum Quality** (12-16+ GB GPU):
   - Use `generator_training_sequence_length: 64`
   - Enable `use_learned_upsampling: true`
   - Train for more epochs (100+)
   - Use data augmentation (`generator_augmentation: true`)

4. **Evaluation** (Critical!):
   - **Don't just measure reconstruction quality** - measure downstream task accuracy!
   - Compare classification accuracy with different sequence lengths
   - If accuracy drops with synthetic data, increase sequence length
   - Monitor both reconstruction and task performance

## Migration from Old Configuration

If you were using `generator_target_sequence_length`:

**Old (deprecated)**:
```json
{
  "generator_target_sequence_length": 4
}
```

**New (recommended)**:
```json
{
  "generator_training_sequence_length": 4,
  "generator_output_sequence_length": null
}
```

Or for full-length generation:
```json
{
  "generator_training_sequence_length": 4,
  "generator_output_sequence_length": 1214
}
```

## Technical Details

### Sequence Reduction
During training, sequences are reduced using `_reduce_sequence_adaptive()` which applies average pooling to maintain feature quality while reducing length.

### Sequence Upsampling
During generation, the `VAEGenerator.sample()` method uses `torch.nn.functional.interpolate()` with linear interpolation to upsample sequences smoothly.

### Architectural Consistency
The generator's encoder and decoder layers are sized based on `generator_training_sequence_length`, ensuring consistent dimensions throughout training:
- Encoder input: `[batch, training_seq_len * input_dim]`
- Latent space: `[batch, latent_dim]`
- Decoder output: `[batch, training_seq_len * input_dim]`
- Final output (after upsampling): `[batch, output_seq_len, input_dim]`
