# Configurazioni Mixed ESC50-VEGAS

## Overview

Queste configurazioni permettono di usare **nodi federati con dataset diversi** nello stesso esperimento: alcuni nodi usano ESC50 e altri VEGAS.

## File Creati

### 1. a2v_mixed_esc50_vegas_2n.json
**Descrizione**: Configurazione con 2 nodi, uno ESC50 e uno VEGAS

**Nodi**:
- **Node 0**: ESC50 dataset, classe `dog`
- **Node 1**: VEGAS dataset, classe `baby_cry`

**Uso**:
```bash
python main.py --config configs/a2v_mixed_esc50_vegas_2n.json
```

### 2. a2v_mixed_esc50_vegas_2n_alternative.json
**Descrizione**: Configurazione alternativa con classi diverse

**Nodi**:
- **Node 0**: ESC50 dataset, classe `rooster`
- **Node 1**: VEGAS dataset, classe `chainsaw`

**Uso**:
```bash
python main.py --config configs/a2v_mixed_esc50_vegas_2n_alternative.json
```

## Struttura della Configurazione

### Nodes Section

```json
"nodes": {
  "0": {
    "dataset": "ESC50",              // Dataset per node 0
    "selected_classes": ["dog"],     // Classe ESC50
    "diffusion_type": "flux",
    "_comment": "Node 0: ESC50 dataset with dog class"
  },
  "1": {
    "dataset": "VEGAS",              // Dataset per node 1
    "selected_classes": ["baby_cry"], // Classe VEGAS
    "diffusion_type": "flux",
    "_comment": "Node 1: VEGAS dataset with baby_cry class"
  }
}
```

### Dataset Section

```json
"dataset": {
  "name": "MIXED",  // Indica configurazione mista
  "path": "./data",
  "_comment": "Mixed configuration uses both ESC50 and VEGAS datasets"
}
```

## Classi Disponibili

### ESC50 Classes (50 totali)
Esempi comuni:
- `dog`, `rooster`, `pig`, `cow`, `frog`
- `cat`, `hen`, `insects`, `sheep`, `crow`
- `rain`, `sea_waves`, `crackling_fire`, `crickets`, `chirping_birds`
- `water_drops`, `wind`, `pouring_water`, `toilet_flush`, `thunderstorm`
- `crying_baby`, `sneezing`, `clapping`, `breathing`, `coughing`
- `footsteps`, `laughing`, `brushing_teeth`, `snoring`, `drinking_sipping`
- `door_wood_knock`, `mouse_click`, `keyboard_typing`, `door_wood_creaks`, `can_opening`
- `washing_machine`, `vacuum_cleaner`, `clock_alarm`, `clock_tick`, `glass_breaking`
- `helicopter`, `chainsaw`, `siren`, `car_horn`, `engine`
- `train`, `church_bells`, `airplane`, `fireworks`, `hand_saw`

### VEGAS Classes (10 totali)
- `baby_cry`
- `chainsaw`
- `dog`
- `drum`
- `fireworks`
- `helicopter`
- `printer`
- `rail_transport`
- `snoring`
- `water_flowing`

## Classi Comuni tra ESC50 e VEGAS

Queste classi esistono in entrambi i dataset e possono essere confrontate:
- `dog` (ESC50) ↔ `dog` (VEGAS)
- `chainsaw` (ESC50) ↔ `chainsaw` (VEGAS)
- `helicopter` (ESC50) ↔ `helicopter` (VEGAS)
- `fireworks` (ESC50) ↔ `fireworks` (VEGAS)
- `snoring` (ESC50) ↔ `snoring` (VEGAS)

## Esempi di Configurazioni Utili

### Esempio 1: Classi Simili tra Dataset
Confrontare la stessa classe su dataset diversi:

```json
"nodes": {
  "0": {
    "dataset": "ESC50",
    "selected_classes": ["dog"]
  },
  "1": {
    "dataset": "VEGAS",
    "selected_classes": ["dog"]
  }
}
```

### Esempio 2: Classi Diverse, Dataset Diversi
```json
"nodes": {
  "0": {
    "dataset": "ESC50",
    "selected_classes": ["airplane"]
  },
  "1": {
    "dataset": "VEGAS",
    "selected_classes": ["baby_cry"]
  }
}
```

### Esempio 3: Multi-Classe per Nodo
```json
"nodes": {
  "0": {
    "dataset": "ESC50",
    "selected_classes": ["dog", "cat", "rooster"]
  },
  "1": {
    "dataset": "VEGAS",
    "selected_classes": ["baby_cry", "dog"]
  }
}
```

## Parametri Federati

### federation section
```json
"federation": {
  "algorithm": "FedA2V",
  "num_clients": 2,           // Numero di nodi
  "global_rounds": 20,        // Round globali
  "local_epochs": 1,          // Epoche locali per nodo
  "eval_gap": 10,            // Frequenza valutazione
  "diffusion_type": "flux",
  "model_aggregation": "none",
  "use_saved_audio_embeddings": false
}
```

### feda2v section
```json
"feda2v": {
  "generate_nodes_images_frequency": 1,  // Genera immagini ogni round
  "generate_global_images_frequency": 0, // Non genera immagini globali
  "generate_low_memomy_footprint": true,
  "adapter_aggregation_mode": "avg",     // Aggregazione: avg, weighted_avg
  "global_model_train": true,
  "global_model_train_from_nodes_adapters": false,
  "global_model_train_from_generator": true,
  "global_model_train_from_nodes_audio_embeddings": false,
  "global_model_train_epochs": 5,
  "use_generator": true,
  "generator_type": "vae",               // vae, gan, diffusion
  "store_audio_embeddings": false,
  "audio_embedding_file_name": null,
  "images_output_dir": "mixed-esc50-vegas-2n"
}
```

## Split Management con le Nuove Feature

Grazie alle nuove feature implementate in VEGAS (v2.0.0), puoi anche specificare split personalizzati:

### Configurazione con Split Personalizzati
```json
"nodes": {
  "0": {
    "dataset": "ESC50",
    "selected_classes": ["dog"],
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "stratify": true,
    "_comment": "ESC50 con split 70-15-15"
  },
  "1": {
    "dataset": "VEGAS",
    "selected_classes": ["baby_cry"],
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "stratify": true,
    "_comment": "VEGAS con split 60-20-20"
  }
}
```

### Configurazione con Fold (ESC50)
```json
"nodes": {
  "0": {
    "dataset": "ESC50",
    "selected_classes": ["dog"],
    "use_folds": true,
    "train_folds": [0, 1, 2],
    "val_folds": [3],
    "test_folds": [4],
    "_comment": "ESC50 con fold ufficiali"
  },
  "1": {
    "dataset": "VEGAS",
    "selected_classes": ["baby_cry"],
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "_comment": "VEGAS con split standard"
  }
}
```

## Best Practices

### 1. Bilanciamento dei Dati
Assicurati che i nodi abbiano quantità simili di dati:
```python
# ESC50: ~40 samples per classe
# VEGAS: varia per classe (10-100+ samples)
```

### 2. Classi Compatibili
Scegli classi semanticamente compatibili per risultati migliori:
- ✅ Buono: `dog` (ESC50) + `baby_cry` (VEGAS)
- ✅ Buono: `helicopter` (ESC50) + `helicopter` (VEGAS)
- ⚠️ Attenzione: Mix troppo eterogenei potrebbero dare risultati inconsistenti

### 3. Stratificazione
Usa sempre `stratify: true` per mantenere la distribuzione delle classi:
```json
"stratify": true
```

### 4. Seed per Riproducibilità
```json
"experiment": {
  "seed": 42  // Stesso seed per risultati riproducibili
}
```

### 5. Batch Size
Adatta il batch size al numero di campioni:
```json
"training": {
  "batch_size": 32  // Riduci se hai pochi campioni
}
```

## Troubleshooting

### Errore: "Class not found in dataset"
**Soluzione**: Verifica che il nome della classe sia corretto
- ESC50: usa underscore (es. `brushing_teeth`)
- VEGAS: usa underscore (es. `baby_cry`)

### Errore: "Insufficient samples"
**Soluzione**: Riduci train_ratio o batch_size
```json
"train_ratio": 0.6,  // Ridotto da 0.7
"batch_size": 16     // Ridotto da 32
```

### Warning: "Ratios normalized"
**Soluzione**: Assicurati che train_ratio + val_ratio + test_ratio = 1.0

### Errore: "Dataset path not found"
**Soluzione**: Verifica i path dei dataset:
- ESC50: solitamente in `/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full`
- VEGAS: solitamente in `/home/lpala/fedgfe/dataset/Audio/VEGAS`

## Testing

### Test Rapido
```bash
# Test con 1 round e poche epoche
python main.py --config configs/a2v_mixed_esc50_vegas_2n.json \
  --global_rounds 1 \
  --local_epochs 1
```

### Verifica Configurazione
```python
# Script per verificare la configurazione
import json

with open('configs/a2v_mixed_esc50_vegas_2n.json', 'r') as f:
    config = json.load(f)

# Verifica nodi
for node_id, node_config in config['nodes'].items():
    print(f"Node {node_id}:")
    print(f"  Dataset: {node_config['dataset']}")
    print(f"  Classes: {node_config['selected_classes']}")
```

## Riferimenti

- **VEGAS Documentation**: [VEGAS_README.md](../VEGAS_README.md)
- **VEGAS Quick Start**: [VEGAS_QUICK_START.txt](../VEGAS_QUICK_START.txt)
- **ESC50 Dataset**: Dataset ESC-50 documentation

## Changelog

### 2024 - Initial Release
- ✅ Created mixed ESC50-VEGAS configurations
- ✅ Support for VEGAS v2.0.0 features
- ✅ Documentation and examples
