# Generator Quality Evaluation Guide

This guide explains how to use the generator quality evaluation methods added to the `clientA2V` class.

## Overview

Three evaluation methods have been added to assess different aspects of generator performance:

1. **`evaluate_generator_quality()`** - Compares generated samples with real embeddings
2. **`evaluate_generator_diversity()`** - Measures diversity within generated samples
3. **`evaluate_generator_coverage()`** - Evaluates how well generated samples cover the real data distribution

## Methods

### 1. Quality Evaluation

Compares generated embeddings with real embeddings using various distance metrics.

```python
def evaluate_generator_quality(self, real_embeddings, generated_embeddings,
                               metrics=['l2_distance', 'cosine_similarity'])
```

**Available Metrics:**
- `l2_distance`: L2 (Euclidean) distance between mean embeddings (lower is better)
- `cosine_similarity`: Cosine similarity between mean embeddings (higher is better, range: -1 to 1)
- `mse`: Mean Squared Error (lower is better)
- `mae`: Mean Absolute Error (lower is better)
- `frechet_distance`: Simplified Fréchet distance (lower is better)

**Returns:**
```python
{
    'per_class': {
        'dog': {'l2_distance': 2.45, 'cosine_similarity': 0.92},
        'cat': {'l2_distance': 2.38, 'cosine_similarity': 0.94},
        ...
    },
    'overall': {
        'l2_distance': {'mean': 2.41, 'std': 0.15, 'min': 2.10, 'max': 2.80},
        'cosine_similarity': {'mean': 0.93, 'std': 0.02, 'min': 0.89, 'max': 0.96}
    },
    'metrics': ['l2_distance', 'cosine_similarity']
}
```

**Usage Example:**
```python
# During training or evaluation
real_audio_embeddings = {
    'dog': torch.randn(100, 4, 768),  # 100 real samples
    'cat': torch.randn(100, 4, 768)
}

generated_audio_embeddings = {
    'dog': torch.randn(50, 4, 768),   # 50 generated samples
    'cat': torch.randn(50, 4, 768)
}

quality_results = client.evaluate_generator_quality(
    real_embeddings=real_audio_embeddings,
    generated_embeddings=generated_audio_embeddings,
    metrics=['l2_distance', 'cosine_similarity', 'mse', 'mae']
)
```

---

### 2. Diversity Evaluation

Measures how diverse the generated samples are within each class.

```python
def evaluate_generator_diversity(self, generated_embeddings)
```

**Metrics Computed:**
- `std`: Standard deviation across samples (higher = more diverse)
- `avg_pairwise_distance`: Average L2 distance between all pairs of samples (higher = more diverse)

**Returns:**
```python
{
    'per_class': {
        'dog': {
            'std': 0.15,
            'avg_pairwise_distance': 3.42,
            'num_samples': 50
        },
        'cat': {
            'std': 0.18,
            'avg_pairwise_distance': 3.89,
            'num_samples': 50
        }
    },
    'overall': {
        'mean_std': 0.165,
        'mean_pairwise_distance': 3.65
    }
}
```

**Usage Example:**
```python
diversity_results = client.evaluate_generator_diversity(
    generated_embeddings=generated_audio_embeddings
)

# Check if diversity is sufficient
if diversity_results['overall']['mean_std'] < 0.1:
    print("Warning: Generated samples have low diversity!")
```

---

### 3. Coverage Evaluation

Evaluates how well generated samples cover the real data distribution.

```python
def evaluate_generator_coverage(self, real_embeddings, generated_embeddings,
                                threshold=0.5)
```

**Metrics:**
- `coverage`: Percentage of real samples that have at least one generated sample within threshold distance (higher is better)
- `precision`: Percentage of generated samples that are within threshold distance of at least one real sample (higher is better)

**Parameters:**
- `threshold`: Distance threshold for considering samples as "close" (adjust based on your embedding space)

**Returns:**
```python
{
    'per_class': {
        'dog': {
            'coverage': 0.85,      # 85% of real samples are covered
            'precision': 0.92,     # 92% of generated samples are realistic
            'num_real': 100,
            'num_generated': 50
        },
        'cat': {
            'coverage': 0.88,
            'precision': 0.90,
            'num_real': 100,
            'num_generated': 50
        }
    },
    'overall': {
        'mean_coverage': 0.865,
        'mean_precision': 0.91
    },
    'threshold': 0.5
}
```

**Usage Example:**
```python
# Adjust threshold based on your embedding space scale
coverage_results = client.evaluate_generator_coverage(
    real_embeddings=real_audio_embeddings,
    generated_embeddings=generated_audio_embeddings,
    threshold=0.5
)

# Check coverage quality
if coverage_results['overall']['mean_coverage'] < 0.7:
    print("Warning: Generated samples do not cover enough of the real distribution!")
if coverage_results['overall']['mean_precision'] < 0.8:
    print("Warning: Many generated samples are not realistic!")
```

---

## Complete Evaluation Workflow

Here's a complete example of how to evaluate generator quality during or after training:

```python
# 1. Collect real embeddings from training data
real_embeddings = client.collect_embeddings_for_generator_training()

# 2. Generate synthetic samples using trained generators
# Option A: Generate from collected CLIP embeddings
class_outputs = {}
for class_name in client.selected_classes:
    # Get CLIP embeddings for this class
    clip_emb = real_embeddings[class_name]['clip'][0]  # Take first sample
    class_outputs[class_name] = {'clip': clip_emb}

generated_embeddings = client.generate_synthetic_samples(class_outputs)

# Option B: Generate from generators directly (for unconditioned VAE)
generated_embeddings = {}
for class_name in client.selected_classes:
    generator = client.get_generator_for_class(class_name)
    if generator is not None:
        gen_samples = generator.sample(
            num_samples=50,
            device=client.device
        )
        generated_embeddings[class_name] = gen_samples

# 3. Extract only audio embeddings from real data
real_audio_embeddings = {
    class_name: embeddings['audio_embeddings']
    for class_name, embeddings in real_embeddings.items()
    if 'audio_embeddings' in embeddings
}

# 4. Evaluate quality
print("\n" + "="*60)
print("GENERATOR QUALITY EVALUATION")
print("="*60)

quality_results = client.evaluate_generator_quality(
    real_embeddings=real_audio_embeddings,
    generated_embeddings=generated_embeddings,
    metrics=['l2_distance', 'cosine_similarity', 'mse', 'mae']
)

# 5. Evaluate diversity
diversity_results = client.evaluate_generator_diversity(
    generated_embeddings=generated_embeddings
)

# 6. Evaluate coverage
coverage_results = client.evaluate_generator_coverage(
    real_embeddings=real_audio_embeddings,
    generated_embeddings=generated_embeddings,
    threshold=0.5
)

# 7. Log results to WandB (if enabled)
if not client.wandb_disabled and wandb.run is not None:
    wandb.log({
        # Quality metrics
        'gen_quality/l2_distance_mean': quality_results['overall']['l2_distance']['mean'],
        'gen_quality/cosine_sim_mean': quality_results['overall']['cosine_similarity']['mean'],
        'gen_quality/mse_mean': quality_results['overall']['mse']['mean'],

        # Diversity metrics
        'gen_diversity/std_mean': diversity_results['overall']['mean_std'],
        'gen_diversity/pairwise_dist_mean': diversity_results['overall']['mean_pairwise_distance'],

        # Coverage metrics
        'gen_coverage/coverage_mean': coverage_results['overall']['mean_coverage'],
        'gen_coverage/precision_mean': coverage_results['overall']['mean_precision'],

        # Per-class metrics (example for first class)
        **{f'gen_quality/{class_name}/l2_dist': metrics['l2_distance']
           for class_name, metrics in quality_results['per_class'].items()},
        **{f'gen_quality/{class_name}/cos_sim': metrics['cosine_similarity']
           for class_name, metrics in quality_results['per_class'].items()}
    })

print("\n✓ Evaluation complete!")
```

---

## Integration with Training Loop

You can integrate these evaluations into your training loop to monitor generator quality over time:

```python
# In serverA2V.py or clientA2V.py training loop

for round_num in range(num_rounds):
    # ... training code ...

    # Evaluate generator quality every N rounds
    if round_num % eval_frequency == 0:
        print(f"\n[Round {round_num}] Evaluating generator quality...")

        # Collect embeddings
        real_embs = client.collect_embeddings_for_generator_training()
        real_audio_embs = {
            cls: emb['audio_embeddings']
            for cls, emb in real_embs.items()
        }

        # Generate samples
        gen_embs = client.generate_synthetic_samples(real_embs)

        # Evaluate
        quality = client.evaluate_generator_quality(
            real_audio_embs, gen_embs,
            metrics=['l2_distance', 'cosine_similarity']
        )
        diversity = client.evaluate_generator_diversity(gen_embs)
        coverage = client.evaluate_generator_coverage(
            real_audio_embs, gen_embs, threshold=0.5
        )

        # Log to WandB
        wandb.log({
            'round': round_num,
            'gen_l2_dist': quality['overall']['l2_distance']['mean'],
            'gen_cos_sim': quality['overall']['cosine_similarity']['mean'],
            'gen_diversity': diversity['overall']['mean_std'],
            'gen_coverage': coverage['overall']['mean_coverage']
        })
```

---

## Interpreting Results

### Quality Metrics

**L2 Distance:**
- Lower values indicate generated samples are closer to real samples
- Typical range: 1.0 - 5.0 (depends on embedding space)
- **Good**: < 2.0
- **Acceptable**: 2.0 - 3.5
- **Poor**: > 3.5

**Cosine Similarity:**
- Higher values indicate better alignment with real samples
- Range: -1.0 to 1.0
- **Excellent**: > 0.95
- **Good**: 0.90 - 0.95
- **Acceptable**: 0.85 - 0.90
- **Poor**: < 0.85

### Diversity Metrics

**Standard Deviation:**
- Higher values indicate more diverse samples
- Too low → mode collapse (generator produces similar samples)
- Too high → unrealistic samples (generator is not stable)
- **Good range**: 0.1 - 0.3

**Pairwise Distance:**
- Average distance between all pairs of generated samples
- Higher values indicate more diverse samples
- **Good**: Similar to pairwise distances in real data

### Coverage Metrics

**Coverage:**
- Percentage of real distribution covered by generated samples
- **Excellent**: > 0.90
- **Good**: 0.80 - 0.90
- **Acceptable**: 0.70 - 0.80
- **Poor**: < 0.70

**Precision:**
- Percentage of generated samples that are realistic
- **Excellent**: > 0.90
- **Good**: 0.85 - 0.90
- **Acceptable**: 0.75 - 0.85
- **Poor**: < 0.75

---

## Tips and Best Practices

1. **Adjust Threshold for Coverage:**
   - Start with `threshold=0.5` and adjust based on results
   - Normalize embeddings if needed for consistent threshold values
   - Monitor both coverage and precision together

2. **Compare Across Rounds:**
   - Track metrics over training rounds to see improvement
   - Save best checkpoint based on combined metrics
   - Watch for overfitting or mode collapse

3. **Per-Class Analysis:**
   - Some classes may be harder to generate than others
   - Focus training on classes with poor metrics
   - Consider per-class generators if quality varies significantly

4. **Combine Multiple Metrics:**
   - Don't rely on a single metric
   - Create a composite score:
     ```python
     composite_score = (
         quality['overall']['cosine_similarity']['mean'] * 0.4 +
         (1.0 / (1.0 + quality['overall']['l2_distance']['mean'])) * 0.3 +
         diversity['overall']['mean_std'] * 0.15 +
         coverage['overall']['mean_coverage'] * 0.15
     )
     ```

5. **Save Evaluation Results:**
   ```python
   import json

   eval_results = {
       'round': round_num,
       'quality': quality_results,
       'diversity': diversity_results,
       'coverage': coverage_results
   }

   with open(f'eval_results_round_{round_num}.json', 'w') as f:
       json.dump(eval_results, f, indent=2)
   ```

---

## Troubleshooting

### Problem: Low Cosine Similarity

**Possible causes:**
- Generator not trained enough
- Learning rate too high/low
- Mismatch between conditioning and target

**Solutions:**
- Increase training epochs
- Adjust learning rate
- Check that conditioning embeddings match the target class

### Problem: Low Diversity (Mode Collapse)

**Possible causes:**
- KL divergence weight too high
- Training data too limited
- Generator architecture too simple

**Solutions:**
- Reduce KL weight in loss function
- Increase data augmentation
- Use larger latent dimension

### Problem: Low Coverage

**Possible causes:**
- Not enough generated samples
- Generator missing parts of the distribution
- Threshold too strict

**Solutions:**
- Generate more samples per class
- Use per-class or per-group generators
- Adjust coverage threshold

---

## Example Output

```
[Client 0] 📊 Evaluating generator quality with metrics: ['l2_distance', 'cosine_similarity']
  • dog: l2_distance: 2.1534, cosine_similarity: 0.9245
  • cat: l2_distance: 2.0821, cosine_similarity: 0.9312
  • bird: l2_distance: 2.2134, cosine_similarity: 0.9187

[Client 0] 📈 Overall Quality Summary:
  • l2_distance: mean=2.1496, std=0.0658, min=2.0821, max=2.2134
  • cosine_similarity: mean=0.9248, std=0.0063, min=0.9187, max=0.9312

[Client 0] 🎨 Evaluating generator diversity
  • dog: std=0.1523, avg_dist=3.4521 (50 samples)
  • cat: std=0.1687, avg_dist=3.7834 (50 samples)
  • bird: std=0.1432, avg_dist=3.2145 (50 samples)

[Client 0] 📊 Overall Diversity:
  • Mean std: 0.1547
  • Mean pairwise distance: 3.4833

[Client 0] 🎯 Evaluating generator coverage (threshold=0.5)
  • dog: coverage=87.00%, precision=92.00% (real=100, gen=50)
  • cat: coverage=89.00%, precision=94.00% (real=100, gen=50)
  • bird: coverage=84.00%, precision=90.00% (real=100, gen=50)

[Client 0] 📊 Overall Coverage:
  • Mean coverage: 86.67%
  • Mean precision: 92.00%
```

---

**Version:** 1.0
**Last Updated:** 2025-12-03
**Author:** FedA2V Team
