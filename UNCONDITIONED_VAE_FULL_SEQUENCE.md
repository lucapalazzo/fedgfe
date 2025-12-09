# Unconditioned VAE - Generazione con Sequenza Completa AST

## Panoramica

Il VAE unconditioned è stato esteso per supportare la generazione di campioni sintetici con la forma completa dell'output di AST, invece della forma ridotta di default.

## Dimensioni degli Embeddings

### Output di AST
- **Forma completa**: `[batch, 1214, 768]`
  - 1214: numero di patch temporali (frame audio processati)
  - 768: dimensione dell'embedding per ogni patch

### VAE Generator (default)
- **Training**: Il VAE viene trainato su sequenze ridotte `[batch, 4, 768]`
  - La riduzione da 1214 a 4 viene effettuata usando `AdaptiveAvgPool1d`
  - Questo rende il training più efficiente

- **Generazione**: Per default, genera campioni `[batch, 4, 768]`

### Nuova Funzionalità: Generazione con Forma AST Completa

Ora è possibile generare campioni con la forma completa dell'output di AST `[batch, 1214, 768]` utilizzando upsampling con interpolazione lineare.

## Configurazione

Nel file di configurazione JSON, aggiungi il parametro `generator_target_sequence_length`:

```json
{
  "feda2v": {
    "generator_type": "vae",
    "use_conditioned_vae": false,
    "generator_granularity": "unified",
    "synthetic_samples_per_class": 5,

    // Nuova opzione: lunghezza sequenza target per la generazione
    "generator_target_sequence_length": 1214  // null o omesso = usa default (4)
  }
}
```

### Valori Supportati

- **`null` o omesso**: Usa la lunghezza default del generatore (4)
  - Output: `[batch, 4, 768]`

- **`1214`**: Genera con la forma completa dell'output di AST
  - Output: `[batch, 1214, 768]`
  - Il generatore produce comunque `[batch, 4, 768]` internamente, poi upsampla a 1214 usando interpolazione lineare

- **Altri valori**: Qualsiasi lunghezza di sequenza desiderata
  - Il generatore upsampla o downsampla alla lunghezza specificata

## Esempi di Configurazione

### Esempio 1: VAE Unconditioned con Forma Ridotta (Default)

```json
{
  "feda2v": {
    "generator_type": "vae",
    "use_conditioned_vae": false,
    "generator_granularity": "unified",
    "synthetic_samples_per_class": 10,
    "generator_training_epochs": 100
  }
}
```

**Output generato**: `[10, 4, 768]` per ogni classe

### Esempio 2: VAE Unconditioned con Forma AST Completa

```json
{
  "feda2v": {
    "generator_type": "vae",
    "use_conditioned_vae": false,
    "generator_granularity": "unified",
    "synthetic_samples_per_class": 10,
    "generator_training_epochs": 100,
    "generator_target_sequence_length": 1214
  }
}
```

**Output generato**: `[10, 1214, 768]` per ogni classe

### Esempio 3: VAE Unconditioned con Lunghezza Personalizzata

```json
{
  "feda2v": {
    "generator_type": "vae",
    "use_conditioned_vae": false,
    "generator_granularity": "unified",
    "synthetic_samples_per_class": 10,
    "generator_training_epochs": 100,
    "generator_target_sequence_length": 512
  }
}
```

**Output generato**: `[10, 512, 768]` per ogni classe

## Come Funziona

### Training del VAE

1. **Input**: Audio embeddings da AST `[batch, 1214, 768]`
2. **Riduzione**: Adaptive pooling riduce a `[batch, 4, 768]`
3. **VAE Forward**:
   - Encoder: `[batch, 4, 768]` → `[batch, latent_dim]`
   - Decoder: `[batch, latent_dim]` → `[batch, 4, 768]`

### Generazione di Campioni

#### Caso 1: Default (seq_len=4)

```python
synthetic_samples = vae_generator.sample(
    num_samples=10,
    device='cuda',
    target_sequence_length=None  # o omesso
)
# Output: [10, 4, 768]
```

#### Caso 2: Forma AST Completa (seq_len=1214)

```python
synthetic_samples = vae_generator.sample(
    num_samples=10,
    device='cuda',
    target_sequence_length=1214
)
# Output: [10, 1214, 768]
```

**Processo interno**:
1. Genera `z ~ N(0, I)` di dimensione `[10, latent_dim]`
2. Decodifica a `[10, 4, 768]`
3. Upsampla usando `torch.nn.functional.interpolate`:
   - Metodo: interpolazione lineare
   - Da: `[10, 768, 4]` (dopo transpose)
   - A: `[10, 768, 1214]`
   - Risultato finale: `[10, 1214, 768]` (dopo transpose back)

## Vantaggi e Svantaggi

### Forma Ridotta (4)

**Vantaggi**:
- Training più efficiente
- Meno parametri da apprendere
- Più veloce nella generazione

**Svantaggi**:
- Perdita di risoluzione temporale
- Potrebbe perdere dettagli fini

### Forma Completa (1214)

**Vantaggi**:
- Risoluzione temporale completa come l'output originale di AST
- Può catturare meglio le variazioni temporali fini
- Compatibile direttamente con pipeline che si aspettano `[batch, 1214, 768]`

**Svantaggi**:
- Richiede upsampling (interpolazione)
- L'interpolazione potrebbe introdurre artefatti
- Leggermente più lento nella generazione (overhead di interpolazione)

## Note Tecniche

### Interpolazione Lineare

L'upsampling usa `torch.nn.functional.interpolate` con:
- **Mode**: `'linear'` - interpolazione lineare 1D
- **Align corners**: `False` - per coerenza con le best practices di PyTorch

Questo metodo:
- È deterministico
- Preserva la continuità temporale
- È computazionalmente efficiente

### Compatibilità

Questa funzionalità è compatibile con:
- ✅ `VAEGenerator` (unconditioned)
- ❌ `ConditionedVAEGenerator` (non ancora implementato)
- ❌ `MultiModalVAEGenerator` (non ancora implementato)

Per i generatori condizionati, la funzionalità può essere aggiunta in futuro seguendo lo stesso pattern.

## Riferimenti nel Codice

### File Modificati

1. **`system/flcore/trainmodel/generators.py`**
   - Metodo `VAEGenerator.sample()` esteso con parametro `target_sequence_length`

2. **`system/flcore/clients/clientA2V.py`**
   - Aggiunto parametro di configurazione `generator_target_sequence_length`
   - Metodo `generate_synthetic_samples()` aggiornato per usare il nuovo parametro
   - Commenti chiariti per distinguere tra dimensioni AST e dimensioni CLIP

### Righe Chiave

- [generators.py:150-181](system/flcore/trainmodel/generators.py#L150-L181) - Implementazione del metodo `sample()` con upsampling
- [clientA2V.py:205](system/flcore/clients/clientA2V.py#L205) - Inizializzazione del parametro di configurazione
- [clientA2V.py:2754-2758](system/flcore/clients/clientA2V.py#L2754-L2758) - Uso del parametro nella generazione

## Testing

Per testare la nuova funzionalità:

```python
# Test con forma ridotta (default)
config1 = {
    "generator_target_sequence_length": None  # o omesso
}
# Output atteso: [batch, 4, 768]

# Test con forma completa AST
config2 = {
    "generator_target_sequence_length": 1214
}
# Output atteso: [batch, 1214, 768]

# Verifica forma
samples = client.generate_synthetic_samples(class_outputs)
for class_name, embs in samples.items():
    print(f"{class_name}: {embs.shape}")  # Dovrebbe mostrare [num_samples, seq_len, 768]
```
