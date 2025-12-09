# Guida Completa: Generatori VAE nei Client per Federated Learning

## Indice
- [Panoramica](#panoramica)
- [Architettura](#architettura)
- [Workflow Completo](#workflow-completo)
- [Configurazione](#configurazione)
- [Run 1: Training Generatori](#run-1-training-generatori)
- [Run 2: Uso Generatori Pre-allenati](#run-2-uso-generatori-pre-allenati)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Panoramica

Il sistema implementa **generatori VAE (Variational AutoEncoder) sui client** per creare campioni sintetici in un setting di federated learning. Il processo si svolge in **due fasi distinte**:

### Fase 1: Training dei Generatori (Run 1)
Ogni client allena il proprio generatore VAE sui propri dati locali e salva un checkpoint.

### Fase 2: Generazione e Federazione (Run 2)
I client caricano i generatori pre-allenati, generano campioni sintetici, e partecipano al federated learning con dati augmented.

---

## Architettura

### Componenti Principali

```
┌─────────────────────────────────────────────────────────┐
│                    SERVER (FedA2V)                      │
│  • Aggrega adapter dai client                          │
│  • Aggrega campioni sintetici                          │
│  • Non ha generatori propri                            │
└─────────────────────────────────────────────────────────┘
                          ▲ │
                          │ │
              Aggregation │ │ Distribution
                          │ │
                          │ ▼
┌──────────────────┬──────────────────┬──────────────────┐
│   CLIENT 0       │   CLIENT 1       │   CLIENT 2       │
│                  │                  │                  │
│  ┌────────────┐  │  ┌────────────┐  │  ┌────────────┐  │
│  │ Local Data │  │  │ Local Data │  │  │ Local Data │  │
│  └──────┬─────┘  │  └──────┬─────┘  │  └──────┬─────┘  │
│         │        │         │        │         │        │
│         ▼        │         ▼        │         ▼        │
│  ┌────────────┐  │  ┌────────────┐  │  ┌────────────┐  │
│  │  Adapters  │  │  │  Adapters  │  │  │  Adapters  │  │
│  └──────┬─────┘  │  └──────┬─────┘  │  └──────┬─────┘  │
│         │        │         │        │         │        │
│         ▼        │         ▼        │         ▼        │
│  ┌────────────┐  │  ┌────────────┐  │  ┌────────────┐  │
│  │VAE Generator│ │  │VAE Generator│ │  │VAE Generator│ │
│  │(conditioned)│  │  │(conditioned)│  │  │(conditioned)│  │
│  └──────┬─────┘  │  └──────┬─────┘  │  └──────┬─────┘  │
│         │        │         │        │         │        │
│    Checkpoint    │    Checkpoint    │    Checkpoint    │
│    Save/Load     │    Save/Load     │    Save/Load     │
└──────────────────┴──────────────────┴──────────────────┘
```

### Generatore VAE Condizionato

```python
ConditionedVAEGenerator(
    input_dim=512,      # Audio embedding dimension
    hidden_dim=1024,    # Hidden layer size
    latent_dim=256,     # Latent space dimension
    visual_dim=768,     # CLIP embedding dimension (conditioning)
    sequence_length=4   # Sequence length
)
```

**Input**: Audio embeddings (512D) + CLIP visual condition (768D)
**Output**: Synthetic audio-like embeddings (512D, sequence of 4)

---

## Workflow Completo

### Run 1: Training dei Generatori

```
Round 1-30:
┌────────────────────────────────────────────────┐
│ 1. Client riceve global adapter                │
│ 2. Training locale adapter su dati reali       │
│ 3. Estrazione adapter outputs per classe       │
│ 4. Training VAE generator (10 epochs)          │
│ 5. Salvataggio checkpoint generatore           │
│ 6. Invio adapter al server (no generatori!)    │
└────────────────────────────────────────────────┘
         │
         ▼
Checkpoint salvati: checkpoints/generators/client_X_generator.pt
```

### Run 2: Uso Generatori Pre-allenati

```
Round 1-50:
┌────────────────────────────────────────────────┐
│ 1. Client carica generatore da checkpoint      │
│ 2. Training locale adapter su dati reali       │
│ 3. Generazione campioni sintetici (10/classe)  │
│ 4. Invio adapter + campioni sintetici al server│
│ 5. Server aggrega tutto                        │
│ 6. Global training con dati reali + sintetici  │
└────────────────────────────────────────────────┘
```

---

## Configurazione

### Parametri Principali (sezione `feda2v`)

#### Modalità di Run

```json
{
  "use_generator": true,              // Abilita generatori
  "generator_type": "vae",             // "vae" o "gan"
  "generator_training_mode": true,    // RUN 1: training generatori
  "use_pretrained_generators": false  // RUN 2: usa generatori pre-allenati
}
```

#### Checkpoint

```json
{
  "generator_checkpoint_dir": "checkpoints/generators"  // Directory checkpoint
}
```

#### Training Generatori (solo Run 1)

```json
{
  "generator_training_epochs": 10,     // Epochs per round
  "generator_augmentation": true,      // Data augmentation
  "generator_augmentation_noise": 0.1  // Std dev rumore gaussiano
}
```

#### Generazione Sintetici (solo Run 2)

```json
{
  "synthetic_samples_per_class": 10  // Campioni sintetici per classe
}
```

---

## Run 1: Training Generatori

### File di Configurazione

Usa: [`configs/a2v_generator_training_mode.json`](configs/a2v_generator_training_mode.json)

### Parametri Chiave

```json
{
  "feda2v": {
    "use_generator": true,
    "generator_type": "vae",
    "generator_training_mode": true,          // ✓ Training mode
    "use_pretrained_generators": false,       // ✗ No caricamento

    "generator_training_epochs": 10,
    "generator_augmentation": true,
    "generator_augmentation_noise": 0.1,

    "global_model_train": false,              // ✗ No global training
    "global_model_train_from_nodes_adapters": false
  }
}
```

### Esecuzione

```bash
python main.py --config configs/a2v_generator_training_mode.json
```

### Output Atteso

```
=== Round 1 ===
[Client 0] Initializing vae generator
[Client 0] Generator initialized successfully
Node 0 - Computing mean adapter outputs per class from training data
  - Class 'dog' clip: mean from 32 samples
  - Class 'rooster' clip: mean from 28 samples
  ...

[Client 0] Starting generator training mode
[Client 0] Training generator for 10 epochs
  [Client 0] Epoch 2/10: Loss=0.3421
  [Client 0] Epoch 4/10: Loss=0.2987
  ...
[Client 0] Generator training completed. Average loss: 0.2543
[Client 0] Saved generator checkpoint to checkpoints/generators/client_0_generator_round_5.pt
```

### Checkpoint Salvati

Ogni client salva periodicamente (ogni 5 round):

```
checkpoints/generators/
├── client_0_generator_round_5.pt
├── client_0_generator_round_10.pt
├── client_0_generator.pt           # Latest checkpoint
├── client_1_generator_round_5.pt
├── client_1_generator.pt
└── ...
```

### Struttura Checkpoint

```python
{
    'client_id': 0,
    'round': 10,
    'generator_type': 'vae',
    'diffusion_type': 'flux',
    'generator_state_dict': {...},  # Pesi del generatore
    'optimizer_state_dict': {...}   # Stato optimizer
}
```

---

## Run 2: Uso Generatori Pre-allenati

### File di Configurazione

Usa: [`configs/a2v_use_pretrained_generators.json`](configs/a2v_use_pretrained_generators.json)

### Parametri Chiave

```json
{
  "feda2v": {
    "use_generator": true,
    "generator_type": "vae",
    "generator_training_mode": false,         // ✗ No training
    "use_pretrained_generators": true,        // ✓ Carica checkpoint

    "synthetic_samples_per_class": 10,

    "global_model_train": true,               // ✓ Global training
    "global_model_train_epochs": 5,
    "global_model_train_from_nodes_adapters": true
  }
}
```

### Esecuzione

```bash
python main.py --config configs/a2v_use_pretrained_generators.json
```

### Output Atteso

```
=== Round 1 ===
[Client 0] Initializing vae generator
[Client 0] Loaded VAE generator from round 30
[Client 0] Generator checkpoint loaded successfully

Node 0 - Training adapters on real data...
[Client 0] Generated 5 synthetic sample sets
  - dog: synthetic samples (10, 4, 512)
  - rooster: synthetic samples (10, 4, 512)
  ...

[Server] Collected 5 synthetic sample sets from Client 0
[Server] Collected 5 synthetic sample sets from Client 1
[Server] Collected 5 synthetic sample sets from Client 2

[Server] Aggregating synthetic samples from 3 clients
[Server] Aggregated 150 synthetic samples across 15 classes
  - dog: 30 samples (10 per client × 3 clients)
  - rain: 30 samples
  ...
```

### Vantaggi

✅ **Data Augmentation**: Più dati per training adapters
✅ **Diversità**: Campioni da distribuzioni apprese
✅ **Privacy**: Nessun dato raw condiviso
✅ **Scalabilità**: Genera infiniti campioni sintetici

---

## API Reference

### Client Methods ([clientA2V.py](system/flcore/clients/clientA2V.py))

#### `initialize_generators()`
Inizializza il generatore VAE/GAN per questo client.

```python
self.initialize_generators()
```

#### `save_generator_checkpoint(round_num=None)`
Salva checkpoint del generatore.

```python
checkpoint_path = client.save_generator_checkpoint(round_num=10)
# Returns: "checkpoints/generators/client_0_generator_round_10.pt"
```

#### `load_generator_checkpoint(checkpoint_path=None)`
Carica checkpoint del generatore.

```python
success = client.load_generator_checkpoint()
# Returns: True if loaded successfully, False otherwise
```

#### `train_node_generator()`
Allena il generatore usando adapter outputs del training locale.

```python
generator_loss = client.train_node_generator()
# Called automatically in train_a2v() if generator_training_mode=True
```

#### `generate_synthetic_samples(class_outputs)`
Genera campioni sintetici usando il generatore pre-allenato.

```python
synthetic_samples = client.generate_synthetic_samples(class_outputs)
# Returns: {class_name: tensor(num_samples, seq_len, 512)}
```

### Server Methods ([serverA2V.py](system/flcore/servers/serverA2V.py))

#### `aggregate_synthetic_samples()`
Aggrega campioni sintetici da tutti i client.

```python
self.aggregate_synthetic_samples()
# Populates: self.aggregated_synthetic_samples
```

---

## Troubleshooting

### Problema: Generator checkpoint not found

**Sintomo**:
```
[Client 0] Warning: Generator checkpoint not found at checkpoints/generators/client_0_generator.pt
```

**Soluzione**:
1. Assicurati di aver eseguito **Run 1** prima di **Run 2**
2. Verifica che la directory `checkpoints/generators/` esista
3. Controlla che `generator_checkpoint_dir` sia lo stesso in entrambi i run

### Problema: Generator loss not converging

**Sintomo**:
```
[Client 0] Epoch 10/10: Loss=2.5432  # Loss troppo alta
```

**Soluzione**:
1. Aumenta `generator_training_epochs` (prova 15-20)
2. Riduci `generator_augmentation_noise` (prova 0.05)
3. Verifica che gli adapter outputs siano corretti
4. Aumenta `global_rounds` nel Run 1 (prova 50 rounds)

### Problema: Out of memory durante generazione

**Sintomo**:
```
RuntimeError: CUDA out of memory
```

**Soluzione**:
1. Riduci `synthetic_samples_per_class` (prova 5 invece di 10)
2. Abilita `optimize_memory_usage: true`
3. Riduci `batch_size` nel training

### Problema: Synthetic samples hanno scarsa qualità

**Sintomo**:
Low cosine similarity < 0.7 in validation metrics

**Soluzione**:
1. **Run 1**: Allena generatori per più rounds (50+ invece di 30)
2. Aumenta `generator_training_epochs` (15-20)
3. Verifica che `generator_augmentation: true`
4. Controlla che gli adapter siano ben allenati prima del Run 1

---

## Metriche e Logging

### Metriche WandB (Run 1)

```
train/node_0/generator_loss      # Loss del generatore per client
```

### Metriche WandB (Run 2)

```
server/synthetic_samples_total   # Totale campioni sintetici aggregati
server/synthetic_samples_classes # Numero classi con campioni sintetici
```

---

## Best Practices

### Run 1: Training Generatori

1. ✅ **Usa più rounds** (30-50) per allenare bene i generatori
2. ✅ **Monitora generator_loss** su WandB per verificare convergenza
3. ✅ **Salva checkpoint periodicamente** (ogni 5 rounds)
4. ✅ **Disabilita global training** per focus sui generatori

### Run 2: Uso Generatori

1. ✅ **Verifica che i checkpoint esistano** prima di iniziare
2. ✅ **Usa batch_size più grandi** (16-32) grazie ai dati sintetici
3. ✅ **Abilita global training** per sfruttare i dati aggregati
4. ✅ **Monitora quality metrics** della generazione

---

## Esempi Completi

### Esempio 1: Training Generatori (3 client, 30 rounds)

```bash
# Run 1
python main.py --config configs/a2v_generator_training_mode.json

# Output: checkpoints salvati in checkpoints/generators/
```

### Esempio 2: Federated Learning con Dati Sintetici

```bash
# Run 2 (dopo Run 1)
python main.py --config configs/a2v_use_pretrained_generators.json

# Output: Federated learning con real + synthetic data
```

### Esempio 3: Resume Training Generatori

Se il Run 1 si interrompe, puoi fare resume modificando la config:

```json
{
  "generator_training_mode": true,
  "use_pretrained_generators": true,  // Carica checkpoint esistenti
  "generator_checkpoint_dir": "checkpoints/generators"
}
```

---

## Riferimenti

- **Architettura VAE**: Kingma & Welling (2014) - "Auto-Encoding Variational Bayes"
- **Conditioned VAE**: Sohn et al. (2015) - "Learning Structured Output Representation"
- **Federated Learning**: McMahan et al. (2017) - "Communication-Efficient Learning"

---

## File Importanti

```
system/flcore/clients/clientA2V.py       # Client con generatori
system/flcore/servers/serverA2V.py       # Server con aggregazione
system/flcore/trainmodel/generators.py   # Architetture VAE/GAN
configs/a2v_generator_training_mode.json # Config Run 1
configs/a2v_use_pretrained_generators.json # Config Run 2
```

---

**Data**: 2025-01-28
**Versione**: 2.0 - Generatori sui Client
