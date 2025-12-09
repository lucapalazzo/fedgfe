# Guida al Training del Generatore VAE nel Server Globale

## Panoramica

Questa implementazione permette di allenare un generatore VAE (Variational AutoEncoder) nel nodo globale del sistema di federated learning per generare prompt embeddings sintetici da utilizzare nel fine-tuning degli adapter globali.

## Architettura

### Componenti Principali

1. **VAE Generator** (`ConditionedVAEGenerator`):
   - Input: Audio embeddings (512D) + Visual condition (768D CLIP)
   - Latent space: 256D
   - Output: Synthetic audio-like embeddings (512D)

2. **Training Pipeline**:
   - Raccolta adapter outputs dai nodi client
   - Training VAE con data augmentation
   - Validazione qualità generativa
   - Fine-tuning adapter globali con dati reali + sintetici

3. **Checkpoint Management**:
   - Salvataggio automatico dei pesi del generatore
   - Caricamento da checkpoint esistenti
   - Supporto per resume training

## Configurazione

### Parametri Principali

Nel file JSON di configurazione, sezione `feda2v`:

```json
{
  "feda2v": {
    // Abilita il generatore VAE
    "use_generator": true,
    "generator_type": "vae",  // o "gan"

    // Checkpoint management
    "generator_save_checkpoint": true,
    "generator_load_checkpoint": false,
    "generator_checkpoint_path": "checkpoints/vae_generator.pt",
    "generator_checkpoint_frequency": 10,  // salva ogni 10 round

    // Training configuration
    "generator_training_epochs": 5,
    "generator_augmentation": true,
    "generator_augmentation_noise": 0.1,  // std dev del rumore gaussiano
    "synthetic_samples_per_class": 5,     // campioni sintetici per classe
    "generator_validation_frequency": 5,   // valida ogni 5 round

    // Adapter training
    "global_model_train_from_nodes_adapters": true,
    "adapter_aggregation_mode": "avg"
  }
}
```

### Esempio Completo

Vedi [configs/example_vae_generator_config.json](configs/example_vae_generator_config.json)

## Flusso di Lavoro

### 1. Inizializzazione

```python
# In serverA2V.py __init__
if self.use_generator:
    self.initialize_generators()

    # Carica checkpoint se configurato
    if self.generator_load_checkpoint:
        success = self.load_generator_checkpoint()
```

### 2. Training (ogni round)

Nel metodo `round_ending_hook()`:

```python
# Step 1: Raccogli adapter outputs dai nodi
all_class_prompts = {
    'class_name': [
        {
            'clip': tensor,           # CLIP embeddings
            't5': tensor,             # T5 embeddings (se Flux)
            'audio_embeddings': tensor # Audio embeddings originali
        },
        ...  # da tutti i nodi
    ]
}

# Step 2: Train VAE con augmentation
generator_loss = self.train_generator_from_class_prompts(all_class_prompts)

# Step 3: Valida (ogni N round)
if self.round % self.generator_validation_frequency == 0:
    validation_metrics = self.validate_generator(all_class_prompts)

# Step 4: Fine-tune adapter globali con prompt sintetici
adapter_loss = self.finetune_adapters_with_prompts(all_class_prompts)

# Step 5: Salva checkpoint (ogni N round)
if self.round % self.generator_checkpoint_frequency == 0:
    self.save_generator_checkpoint(round_num=self.round)
```

### 3. Generazione Prompt Sintetici

Durante il fine-tuning degli adapter:

```python
# Generate synthetic samples per ogni classe
with torch.no_grad():
    synthetic_audio_embs = self.prompt_generator.sample(
        num_samples=5,
        visual_condition=mean_clip_embedding,
        device=self.device
    )

# Combina reali + sintetici per training
combined = torch.cat([real_embeddings, synthetic_audio_embs], dim=0)
```

## Metriche e Logging

### Metriche Logged su WandB

1. **Training Metrics**:
   - `server/generator_loss`: Loss totale del VAE
   - `server/adapter_finetuning_loss`: Loss fine-tuning adapter

2. **Validation Metrics** (ogni N round):
   - `server/generator_validation_similarity`: Cosine similarity media
   - `server/generator_validation_mse`: MSE loss media
   - `server/generator_validation_l1`: L1 loss media

3. **Per-Epoch Metrics** (in logs):
   - Reconstruction loss
   - KL divergence loss
   - Similarity loss

### Esempio Output Log

```
=== Round 10: Training Generator and Global Adapters ===
Training generator for 5 epochs with augmentation=True
  Epoch 1 Class 'dog': VAE Loss=0.3421 (Recon=0.2134, KL=0.0987, Sim=0.0300)
  Epoch 2 Class 'dog': VAE Loss=0.2987 (Recon=0.1876, KL=0.0854, Sim=0.0257)
  ...
Generator training completed. Average loss: 0.2543

Validating generator at round 10
=== Generator Validation Metrics ===
Average Cosine Similarity: 0.8734
Average MSE Loss: 0.001234
Average L1 Loss: 0.023456
Classes validated: 15
  - dog: 0.8921
  - cat: 0.8543
  ...

Saving generator checkpoint at round 10
Saved generator checkpoint to checkpoints/vae_generator_round_10.pt
```

## Vantaggi del Sistema

### 1. Data Augmentation
- Genera variazioni plausibili degli audio embeddings
- Riduce overfitting su classi con pochi sample
- Migliora generalizzazione degli adapter globali

### 2. Privacy-Preserving
- Non condivide audio raw tra nodi
- Solo embeddings aggregati vanno al server
- Il VAE impara distribuzioni, non dati specifici

### 3. Scalabilità
- Genera infiniti sample sintetici
- Bilancia classi sbilanciate
- Supporta continual learning

### 4. Flessibilità
- Supporta VAE e GAN
- Configurabile via JSON
- Checkpoint per resume training

## Troubleshooting

### Problema: Generator loss non converge

**Soluzione**:
- Aumenta `generator_training_epochs` (prova 10-15)
- Riduci `generator_augmentation_noise` (prova 0.05)
- Verifica che gli adapter outputs siano corretti

### Problema: Validation similarity bassa

**Soluzione**:
- Il VAE sta ancora imparando, aspetta più round
- Controlla dimensioni latent space (256D default)
- Verifica che visual_condition sia corretta

### Problema: Out of memory durante training

**Soluzione**:
- Abilita `optimize_memory_usage: true`
- Riduci `synthetic_samples_per_class` (prova 3)
- Riduci batch size implicito (meno classi per nodo)

### Problema: Checkpoint non si carica

**Soluzione**:
- Verifica path: `generator_checkpoint_path`
- Controlla che `generator_type` matchi il checkpoint
- Usa `generator_load_checkpoint: true` in config

## Advanced Usage

### Custom Loss Function

Modifica `VAELoss` in [generators.py](system/flcore/trainmodel/generators.py):

```python
class VAELoss(nn.Module):
    def __init__(self, beta_schedule='constant', beta_max=1.0):
        self.beta_schedule = beta_schedule
        self.beta_max = beta_max

    def forward(self, recon_x, x, mu, logvar, epoch):
        # Custom reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL divergence con beta annealing
        beta = self._get_beta(epoch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Similarity loss (cosine)
        sim_loss = 1 - F.cosine_similarity(recon_x, x, dim=-1).mean()

        return recon_loss + beta * kl_loss + 0.1 * sim_loss
```

### Multi-Modal Generation

Per generare anche T5 embeddings (Flux):

```python
# Aggiungi secondo VAE per T5
self.prompt_generator_t5 = ConditionedVAEGenerator(
    input_dim=512,
    visual_dim=4096,  # T5 dimension
    hidden_dim=1024,
    latent_dim=512,
    sequence_length=17  # T5 sequence length
)
```

## File Modificati

- [system/flcore/servers/serverA2V.py](system/flcore/servers/serverA2V.py):
  - `save_generator_checkpoint()`: Salva checkpoint
  - `load_generator_checkpoint()`: Carica checkpoint
  - `validate_generator()`: Valida qualità
  - `train_generator_from_class_prompts()`: Training multi-epoca con augmentation
  - `finetune_adapters_with_prompts()`: Usa prompt sintetici
  - `round_ending_hook()`: Logging e checkpoint automatico

## Prossimi Step

Per testare il sistema completo:

1. **Configura l'esperimento**:
   ```bash
   cp configs/example_vae_generator_config.json configs/my_experiment.json
   # Modifica parametri come necessario
   ```

2. **Run training**:
   ```bash
   python main.py --config configs/my_experiment.json
   ```

3. **Monitor metrics su WandB**:
   - Generator loss trends
   - Validation similarity
   - Adapter fine-tuning effectiveness

4. **Analyze checkpoints**:
   ```python
   checkpoint = torch.load('checkpoints/vae_generator_round_50.pt')
   print(f"Round: {checkpoint['round']}")
   print(f"Generator type: {checkpoint['generator_type']}")
   ```

## Riferimenti

- Paper VAE originale: Kingma & Welling (2014)
- Conditioned VAE: Sohn et al. (2015)
- Federated Learning: McMahan et al. (2017)
