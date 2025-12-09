# Generator Training Modes - Complete Guide

Questo documento descrive tutte le modalità di addestramento dei generatori disponibili nel sistema FedA2V.

## Indice
1. [Modalità di Training](#modalità-di-training)
2. [Granularità dei Generatori](#granularità-dei-generatori)
3. [Configurazione Checkpoint](#configurazione-checkpoint)
4. [Esempi di Configurazione](#esempi-di-configurazione)
5. [Validazione Metadati](#validazione-metadati)

---

## Modalità di Training

### 1. **Normal Mode** (Default)
Training normale degli adapters con possibilità di usare generatori pre-addestrati per inferenza.

```json
{
  "feda2v": {
    "use_generator": true,
    "use_pretrained_generators": true,
    "generator_training_mode": false,
    "generator_only_mode": false
  }
}
```

**Comportamento:**
- ✅ Train adapters (CLIP, T5)
- ✅ Use generators for inference (if pretrained)
- ❌ Train generators

---

### 2. **Generator Training Mode**
Training simultaneo di adapters e generatori.

```json
{
  "feda2v": {
    "use_generator": true,
    "generator_training_mode": true,
    "generator_only_mode": false
  }
}
```

**Comportamento:**
- ✅ Train adapters (CLIP, T5)
- ✅ Train generators usando gli output degli adapters
- ✅ Save generator checkpoints
- 📊 Adapters outputs sono memorizzati e usati per training generatori

**Workflow:**
1. Train adapters per alcune epoche
2. Raccolta output adapters per classe
3. Train generatori usando gli output raccolti
4. Salvataggio checkpoint generatori

---

### 3. **Generator-Only Mode** ⭐ NUOVO
Training SOLO dei generatori, senza training degli adapters.

```json
{
  "feda2v": {
    "use_generator": false,
    "generator_training_mode": true,
    "generator_only_mode": true,
    "use_pretrained_generators": false
  }
}
```

**Comportamento:**
- ❌ Skip adapter training
- ✅ Train ONLY generators
- ✅ Usa adapters frozen (dal global model o pre-trained)
- ⚡ Molto più veloce del training completo

**Use Case:**
- Fine-tuning dei generatori con nuovi dati
- Sperimentazione con diverse architetture di generatori
- Training generatori per nuove classi senza modificare adapters

---

### 4. **Server-Only Generator Training**
Training dei generatori solo sul server globale.

```json
{
  "feda2v": {
    "global_model_train": true,
    "global_model_train_from_generator": true,
    "generator_training_mode": false,
    "use_generator": false
  }
}
```

**Comportamento (Server):**
- ✅ Aggrega adapter outputs da tutti i client
- ✅ Train generator globale con dati aggregati
- ✅ Distribuisce generator ai client

**Comportamento (Client):**
- ✅ Train adapters normalmente
- ❌ Non trainano generatori localmente
- ✅ Ricevono generator dal server

---

## Granularità dei Generatori

### Option 1: **Unified Generator** (Default)
Un singolo generatore per tutte le classi.

```json
{
  "feda2v": {
    "generator_granularity": "unified"
  }
}
```

**Pro:**
- Più semplice da gestire
- Meno memoria richiesta
- Transfer learning tra classi

**Contro:**
- Potrebbe non catturare specificità di ogni classe

**File salvati:**
```
checkpoints/generators/
  ├── vae_generator_node0.pt
  ├── vae_generator_node1.pt
  └── vae_generator_node2.pt
```

---

### Option 2: **Per-Class Generators** ⭐ NUOVO
Un generatore separato per ogni classe.

```json
{
  "feda2v": {
    "generator_granularity": "per_class"
  }
}
```

**Pro:**
- Specializzazione massima per ogni classe
- Migliore qualità per classi molto diverse

**Contro:**
- Più memoria richiesta (N_classes × model_size)
- Training più lungo
- Potrebbe overfittare con pochi dati

**File salvati:**
```
checkpoints/generators/
  ├── vae_generator_node0_class_dog.pt
  ├── vae_generator_node0_class_rooster.pt
  ├── vae_generator_node0_class_pig.pt
  ├── vae_generator_node1_class_rain.pt
  └── ...
```

**Esempio Completo:**
```json
{
  "feda2v": {
    "generator_only_mode": true,
    "generator_training_mode": true,
    "generator_granularity": "per_class",
    "generator_checkpoint_base_name": "vae_per_class",
    "generator_training_epochs": 10
  },
  "nodes": {
    "0": {
      "selected_classes": ["dog", "cat", "bird"]
    }
  }
}
```

Risultato: 3 generatori per il nodo 0 (uno per dog, uno per cat, uno per bird)

---

### Option 3: **Per-Group Generators** ⭐ NUOVO
Un generatore per ogni gruppo di classi correlate.

```json
{
  "feda2v": {
    "generator_granularity": "per_group",
    "generator_class_groups": {
      "animals": ["dog", "rooster", "pig", "cow", "frog"],
      "nature": ["rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds"],
      "mechanical": ["helicopter", "chainsaw", "siren", "car_horn", "engine"]
    }
  }
}
```

**Pro:**
- Bilanciamento tra specializzazione e generalizzazione
- Transfer learning dentro ogni gruppo
- Meno memoria rispetto a per-class

**Contro:**
- Richiede definizione manuale dei gruppi
- Performance dipende dalla qualità del grouping

**File salvati:**
```
checkpoints/generators/
  ├── vae_generator_node0_group_animals.pt
  ├── vae_generator_node1_group_nature.pt
  └── vae_generator_node2_group_mechanical.pt
```

**Esempio Completo:**
```json
{
  "feda2v": {
    "generator_only_mode": true,
    "generator_training_mode": true,
    "generator_granularity": "per_group",
    "generator_class_groups": {
      "domestic_animals": ["dog", "cat", "cow", "pig"],
      "wild_animals": ["lion", "elephant", "wolf"],
      "weather": ["rain", "thunderstorm", "wind"],
      "water": ["sea_waves", "river", "waterfall"]
    },
    "generator_checkpoint_base_name": "vae_grouped"
  }
}
```

---

## Configurazione Checkpoint

### Sistema di Naming Flessibile

```json
{
  "feda2v": {
    "generator_checkpoint_dir": "checkpoints/generators",
    "generator_checkpoint_base_name": "vae_generator",
    "generator_save_checkpoint": true,
    "generator_load_checkpoint": false,
    "generator_checkpoint_frequency": 10
  }
}
```

### Naming Convention

**Unified:**
- Client: `{base_name}_node{id}[_round_{num}].pt`
- Server: `{base_name}[_round_{num}].pt`

**Per-Class:**
- Client: `{base_name}_node{id}_class_{class_name}[_round_{num}].pt`
- Server: `{base_name}_class_{class_name}[_round_{num}].pt`

**Per-Group:**
- Client: `{base_name}_node{id}_group_{group_name}[_round_{num}].pt`
- Server: `{base_name}_group_{group_name}[_round_{num}].pt`

### Esempi di Path Completi

```
# Unified
checkpoints/generators/vae_generator_node0.pt
checkpoints/generators/vae_generator_node0_round_10.pt

# Per-Class
checkpoints/generators/vae_per_class_node0_class_dog.pt
checkpoints/generators/vae_per_class_node0_class_dog_round_10.pt

# Per-Group
checkpoints/generators/vae_grouped_node0_group_animals.pt
checkpoints/generators/vae_grouped_node0_group_animals_round_10.pt
```

---

## Validazione Metadati

Ogni checkpoint include metadati completi per validazione:

### Metadati Client
```python
{
    'client_id': 0,
    'node_id': 0,
    'round': 10,
    'timestamp': '2025-11-28T14:30:00',
    'generator_type': 'vae',
    'generator_granularity': 'per_class',  # NEW
    'class_name': 'dog',                    # NEW (for per_class)
    'group_name': 'animals',                # NEW (for per_group)
    'diffusion_type': 'flux',
    'dataset_name': 'ESC50',
    'selected_classes': ['dog', 'rooster', 'pig', 'cow', 'frog'],
    'train_folds': [0, 1, 2, 3],
    'test_folds': [4],
    'num_train_samples': 400,
    'num_test_samples': 100,
    'generator_state_dict': {...},
    'optimizer_state_dict': {...}
}
```

### Metadati Server
```python
{
    'checkpoint_type': 'server',
    'is_global': True,
    'round': 10,
    'timestamp': '2025-11-28T14:30:00',
    'generator_type': 'vae',
    'generator_granularity': 'per_group',  # NEW
    'num_clients': 3,
    'client_metadata': [
        {'client_id': 0, 'dataset': 'ESC50', 'selected_classes': [...]},
        {'client_id': 1, 'dataset': 'ESC50', 'selected_classes': [...]},
        {'client_id': 2, 'dataset': 'ESC50', 'selected_classes': [...]}
    ],
    'federated_datasets': ['ESC50'],
    'federated_classes': ['dog', 'rain', 'helicopter', ...],
    'class_groups': {                      # NEW (for per_group)
        'animals': ['dog', 'rooster', ...],
        'nature': ['rain', 'sea_waves', ...],
        'mechanical': ['helicopter', 'chainsaw', ...]
    },
    'generator_state_dict': {...}
}
```

### Validazione al Caricamento

```python
# Strict validation (default)
success = client.load_generator_checkpoint(
    checkpoint_path="checkpoints/generators/vae_generator_node0.pt",
    strict_validation=True,
    warn_only=False
)

# Permissive mode
success = client.load_generator_checkpoint(
    checkpoint_path="checkpoints/generators/vae_generator_node1.pt",
    strict_validation=False,
    warn_only=True
)
```

**Validazione Errors (blocca caricamento):**
- ✗ Node ID mismatch
- ✗ Generator type mismatch
- ✗ Generator granularity mismatch
- ✗ Class name mismatch (per per_class)
- ✗ Group name mismatch (per per_group)

**Validazione Warnings (solo avviso):**
- ⚠ Dataset mismatch
- ⚠ Selected classes mismatch
- ⚠ Diffusion type mismatch

---

## Esempi di Configurazione

### Esempio 1: Training Generatori Unificati (Solo Generatori)

```json
{
  "experiment": {
    "name": "generator_only_unified"
  },
  "feda2v": {
    "generator_only_mode": true,
    "generator_training_mode": true,
    "use_generator": false,
    "generator_granularity": "unified",
    "generator_type": "vae",
    "generator_training_epochs": 10,
    "generator_save_checkpoint": true,
    "generator_checkpoint_dir": "checkpoints/unified",
    "generator_checkpoint_base_name": "vae_unified",
    "generator_checkpoint_frequency": 5
  },
  "nodes": {
    "0": {
      "dataset": "ESC50",
      "selected_classes": ["dog", "rooster", "pig"]
    }
  }
}
```

**Output:**
```
checkpoints/unified/
  └── vae_unified_node0.pt         (1 generatore per tutte le classi)
```

---

### Esempio 2: Generatori Per-Classe

```json
{
  "experiment": {
    "name": "generator_per_class"
  },
  "feda2v": {
    "generator_only_mode": true,
    "generator_training_mode": true,
    "generator_granularity": "per_class",
    "generator_checkpoint_base_name": "vae_class",
    "generator_training_epochs": 15
  },
  "nodes": {
    "0": {
      "dataset": "ESC50",
      "selected_classes": ["dog", "cat", "bird"]
    }
  }
}
```

**Output:**
```
checkpoints/generators/
  ├── vae_class_node0_class_dog.pt
  ├── vae_class_node0_class_cat.pt
  └── vae_class_node0_class_bird.pt
```

---

### Esempio 3: Generatori Per-Gruppo

```json
{
  "experiment": {
    "name": "generator_per_group"
  },
  "feda2v": {
    "generator_only_mode": true,
    "generator_training_mode": true,
    "generator_granularity": "per_group",
    "generator_class_groups": {
      "pets": ["dog", "cat"],
      "wildlife": ["lion", "elephant", "wolf"],
      "birds": ["rooster", "crow", "owl"]
    },
    "generator_checkpoint_base_name": "vae_group"
  },
  "nodes": {
    "0": {
      "selected_classes": ["dog", "cat", "rooster", "crow"]
    }
  }
}
```

**Output:**
```
checkpoints/generators/
  ├── vae_group_node0_group_pets.pt    (dog, cat)
  └── vae_group_node0_group_birds.pt   (rooster, crow)
```

---

### Esempio 4: Training Ibrido (Adapters + Generatori)

```json
{
  "experiment": {
    "name": "hybrid_training"
  },
  "feda2v": {
    "generator_only_mode": false,
    "generator_training_mode": true,
    "use_generator": true,
    "generator_granularity": "per_group",
    "generator_class_groups": {
      "animals": ["dog", "cat", "cow"],
      "nature": ["rain", "wind", "thunder"]
    },
    "adapters_learning_rate": 0.001,
    "generator_training_epochs": 5
  }
}
```

**Comportamento:**
1. Train adapters per 3 epoche locali
2. Raccolta adapter outputs
3. Train 2 generatori (uno per 'animals', uno per 'nature')
4. Save checkpoint di tutti i modelli

---

## Best Practices

### 1. **Scelta della Granularità**

**Usa `unified` quando:**
- ✅ Hai pochi dati per classe
- ✅ Le classi sono correlate (es. tutti animali)
- ✅ Vuoi massimizzare transfer learning
- ✅ Hai limiti di memoria

**Usa `per_class` quando:**
- ✅ Hai molti dati per ogni classe
- ✅ Le classi sono molto diverse (es. animali vs veicoli vs suoni naturali)
- ✅ Vuoi massima qualità generativa per ogni classe
- ✅ Hai memoria sufficiente

**Usa `per_group` quando:**
- ✅ Puoi raggruppare classi semanticamente (es. animali domestici, selvatici)
- ✅ Vuoi bilanciare qualità e efficienza
- ✅ Hai gruppi naturali nel tuo dataset

### 2. **Modalità di Training**

**Generator-Only Mode quando:**
- ✅ Hai già adapters pre-trained
- ✅ Vuoi solo migliorare la generazione
- ✅ Hai tempo limitato

**Generator Training Mode quando:**
- ✅ Stai partendo da zero
- ✅ Vuoi co-ottimizzare adapters e generators

**Server-Only quando:**
- ✅ Vuoi centralizzare il training dei generatori
- ✅ Hai risorse limitate sui client

### 3. **Checkpoint Management**

```json
{
  "generator_save_checkpoint": true,
  "generator_checkpoint_frequency": 10,
  "generator_checkpoint_dir": "checkpoints/experiment_name"
}
```

- Usa `checkpoint_frequency` appropriato (ogni 5-10 rounds)
- Organizza checkpoint per esperimento usando `checkpoint_dir`
- Usa naming descrittivo in `checkpoint_base_name`

---

## Troubleshooting

### Errore: "Generator type mismatch"
```
✗ Generator type mismatch: checkpoint=gan, current=vae
```
**Soluzione:** Assicurati che il checkpoint sia dello stesso tipo del generatore configurato.

### Errore: "Granularity mismatch"
```
✗ Generator granularity mismatch: checkpoint=per_class, current=unified
```
**Soluzione:** Il checkpoint è stato salvato con granularità diversa. Usa `strict_validation=False` per forzare il caricamento o riaddestra con la stessa granularità.

### Errore: "Class name not found in groups"
```
✗ Class 'dog' not found in any configured group
```
**Soluzione:** Verifica che tutte le classi dei nodi siano presenti in `generator_class_groups`.

---

## API Reference

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_generator` | bool | false | Use generators for inference |
| `generator_training_mode` | bool | false | Train generators during training |
| `generator_only_mode` | bool | false | Skip adapter training, only train generators |
| `generator_granularity` | str | "unified" | "unified", "per_class", or "per_group" |
| `generator_class_groups` | dict | null | Class grouping for per_group mode |
| `generator_type` | str | "vae" | "vae" or "gan" |
| `generator_checkpoint_dir` | str | "checkpoints/generators" | Directory for checkpoints |
| `generator_checkpoint_base_name` | str | "client_generator" | Base name for checkpoint files |
| `generator_checkpoint_frequency` | int | 10 | Save checkpoint every N rounds |
| `generator_training_epochs` | int | 5 | Training epochs per round |

---

**Versione:** 2.0
**Ultima modifica:** 2025-11-28
**Autore:** FedA2V Team
