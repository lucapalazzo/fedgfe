# Generator Training Configurations

Questa directory contiene configurazioni pronte all'uso per l'allenamento dei generatori VAE nel sistema FedA2V.

---

## 📁 File di Configurazione Disponibili

### 1. Training Configs (RUN 1 - Allena Generatori)

#### [`a2v_generator_training_mode.json`](a2v_generator_training_mode.json) - **Unified**
- **Granularità**: `unified` (1 generatore per tutte le classi)
- **Modalità**: `generator_only_mode: true` (solo training generatori, skip adapters)
- **Output**: 1 checkpoint per nodo
- **Memoria**: ~1 GB per nodo
- **Uso**: Setup più semplice e veloce

```bash
python main.py --config configs/a2v_generator_training_mode.json
```

**Output atteso**:
```
checkpoints/generators/
├── vae_unified_node0.pt
├── vae_unified_node1.pt
└── vae_unified_node2.pt
```

---

#### [`a2v_generator_training_per_class.json`](a2v_generator_training_per_class.json) - **Per-Class**
- **Granularità**: `per_class` (1 generatore per classe)
- **Modalità**: `generator_only_mode: true`
- **Output**: 5 checkpoint per nodo (per ESC50 subset)
- **Memoria**: ~5 GB per nodo
- **Uso**: Massima qualità, ogni generatore specializzato per una classe

```bash
python main.py --config configs/a2v_generator_training_per_class.json
```

**Output atteso** (Node 0):
```
checkpoints/generators_per_class/
├── vae_perclass_node0_class_dog.pt
├── vae_perclass_node0_class_rooster.pt
├── vae_perclass_node0_class_pig.pt
├── vae_perclass_node0_class_cow.pt
└── vae_perclass_node0_class_frog.pt
```

---

#### [`a2v_generator_training_per_group.json`](a2v_generator_training_per_group.json) - **Per-Group**
- **Granularità**: `per_group` (1 generatore per gruppo semantico)
- **Modalità**: `generator_only_mode: true`
- **Output**: 1 checkpoint per gruppo per nodo
- **Memoria**: ~1 GB per gruppo per nodo
- **Uso**: Bilanciamento ottimale qualità/memoria

**Gruppi definiti**:
- `animals`: dog, rooster, pig, cow, frog
- `nature`: rain, sea_waves, crackling_fire, crickets, chirping_birds
- `mechanical`: helicopter, chainsaw, siren, car_horn, engine

```bash
python main.py --config configs/a2v_generator_training_per_group.json
```

**Output atteso** (Node 0 - ha solo classi "animals"):
```
checkpoints/generators_per_group/
└── vae_group_node0_group_animals.pt
```

---

### 2. Inference Config (RUN 2 - Usa Generatori Pre-Allenati)

#### [`a2v_use_pretrained_generators.json`](a2v_use_pretrained_generators.json)
- **Granularità**: `unified` (deve corrispondere al training!)
- **Modalità**: `use_pretrained_generators: true`, `generator_training_mode: false`
- **Scopo**: Carica generatori pre-allenati e genera campioni sintetici durante FL

```bash
# Prima allena i generatori (RUN 1)
python main.py --config configs/a2v_generator_training_mode.json

# Poi usa i generatori allenati (RUN 2)
python main.py --config configs/a2v_use_pretrained_generators.json
```

---

## 🔄 Workflow Completo

### Opzione 1: Unified (Più Veloce)

```bash
# STEP 1: Allena 1 generatore unified per nodo
python main.py --config configs/a2v_generator_training_mode.json

# STEP 2: Usa i generatori per federated learning con data augmentation
python main.py --config configs/a2v_use_pretrained_generators.json
```

### Opzione 2: Per-Class (Massima Qualità)

```bash
# STEP 1: Allena 5 generatori per classe per nodo
python main.py --config configs/a2v_generator_training_per_class.json

# STEP 2: Modifica a2v_use_pretrained_generators.json
#   - Cambia "generator_granularity": "per_class"
#   - Cambia "generator_checkpoint_base_name": "vae_perclass"
#   - Cambia "generator_checkpoint_dir": "checkpoints/generators_per_class"

# STEP 3: Usa i generatori per federated learning
python main.py --config configs/a2v_use_pretrained_generators.json
```

### Opzione 3: Per-Group (Bilanciato)

```bash
# STEP 1: Allena generatori per gruppo semantico
python main.py --config configs/a2v_generator_training_per_group.json

# STEP 2: Modifica a2v_use_pretrained_generators.json
#   - Cambia "generator_granularity": "per_group"
#   - Cambia "generator_checkpoint_base_name": "vae_group"
#   - Cambia "generator_checkpoint_dir": "checkpoints/generators_per_group"
#   - Aggiungi "generator_class_groups": {...} (copia dal training config)

# STEP 3: Usa i generatori per federated learning
python main.py --config configs/a2v_use_pretrained_generators.json
```

---

## ⚙️ Parametri Chiave

### Granularità
```json
{
  "generator_granularity": "unified",  // "unified" | "per_class" | "per_group"
  "generator_class_groups": {          // Solo per per_group (dict con nomi)
    "animals": ["dog", "cat", "cow"],
    "nature": ["rain", "wind", "thunder"]
  }
}
```

### Modalità Training
```json
{
  "generator_training_mode": true,     // Train generators?
  "generator_only_mode": true,         // ONLY generators (skip adapters)?
  "use_pretrained_generators": false   // Load pretrained generators?
}
```

### Checkpoint
```json
{
  "generator_checkpoint_dir": "checkpoints/generators",
  "generator_checkpoint_base_name": "vae_unified",
  "generator_save_checkpoint": true,
  "generator_load_checkpoint": false,
  "generator_checkpoint_frequency": 5  // Salva ogni 5 rounds
}
```

---

## 📊 Confronto Granularità

| Granularità | Generatori/Nodo | Memoria/Nodo | Qualità | Velocità Training |
|-------------|-----------------|--------------|---------|-------------------|
| **Unified** | 1 | ~1 GB | Buona | ⚡⚡⚡ Veloce |
| **Per-Class** | 5 | ~5 GB | Ottima | ⚡ Lenta |
| **Per-Group** | 1-3 | ~1-3 GB | Molto Buona | ⚡⚡ Media |

---

## 🔍 Verifica Checkpoint

### Unified
```bash
ls -lh checkpoints/generators/
# Output:
# vae_unified_node0.pt
# vae_unified_node0_round_5.pt
# vae_unified_node0_round_10.pt
```

### Per-Class
```bash
ls -lh checkpoints/generators_per_class/
# Output:
# vae_perclass_node0_class_dog.pt
# vae_perclass_node0_class_cat.pt
# vae_perclass_node0_class_dog_round_5.pt
```

### Per-Group
```bash
ls -lh checkpoints/generators_per_group/
# Output:
# vae_group_node0_group_animals.pt
# vae_group_node0_group_nature.pt
# vae_group_node0_group_animals_round_5.pt
```

---

## ⚠️ Note Importanti

### 1. Compatibilità Granularità
**La granularità deve essere la stessa tra training e inference!**

```bash
# ❌ ERRATO - Mismatch
python main.py --config a2v_generator_training_per_class.json  # per_class
python main.py --config a2v_use_pretrained_generators.json     # unified (default)

# ✅ CORRETTO
python main.py --config a2v_generator_training_per_class.json
# Poi modifica a2v_use_pretrained_generators.json per usare per_class
python main.py --config a2v_use_pretrained_generators.json
```

### 2. Checkpoint Naming
Il parametro `generator_checkpoint_base_name` deve essere coerente:
- Unified: `"vae_unified"`
- Per-Class: `"vae_perclass"`
- Per-Group: `"vae_group"`

### 3. Directory Separate
Usa directory diverse per evitare conflitti:
- Unified: `checkpoints/generators`
- Per-Class: `checkpoints/generators_per_class`
- Per-Group: `checkpoints/generators_per_group`

---

## 📚 Documentazione Completa

Per maggiori dettagli:
- **[GENERATOR_TRAINING_MODES.md](../GENERATOR_TRAINING_MODES.md)**: Guida completa (400+ linee)
- **[IMPLEMENTATION_COMPLETE.md](../IMPLEMENTATION_COMPLETE.md)**: Status implementazione
- **[IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)**: Checklist e testing

---

## 🐛 Troubleshooting

### Errore: "Generator checkpoint not found"
```bash
# Verifica che i checkpoint esistano
ls checkpoints/generators/

# Verifica che il base_name corrisponda
grep "generator_checkpoint_base_name" configs/a2v_use_pretrained_generators.json
```

### Errore: "Granularity mismatch"
```bash
# Il sistema rileva automaticamente mismatch nei metadati
# Soluzione: usa la stessa granularità tra training e inference
```

### Out of Memory con Per-Class
```bash
# Riduci batch_size o usa per_group invece di per_class
# Oppure usa generator_granularity: "unified"
```

---

**Versione**: 2.0
**Data**: 2025-11-28
**Status**: ✅ Production Ready
