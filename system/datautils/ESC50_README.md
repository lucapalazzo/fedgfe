# ESC-50 Dataset Support

## Overview

Il supporto per il dataset ESC-50 è stato implementato in `dataset_esc50.py`. ESC-50 (Environmental Sound Classification - 50 classes) è un dataset di classificazione di suoni ambientali contenente 2000 registrazioni audio di 5 secondi organizzate in 50 classi semantiche.

## Struttura del Dataset

```
esc50-v2.0.0-full/
├── audio/
│   ├── fold00/
│   ├── fold01/
│   ├── fold02/
│   ├── fold03/
│   └── fold04/
├── image/
│   └── cat_*.png (200 immagini, 4 per classe)
├── class_labels.json
├── captions.json
├── fold00.json
├── fold01.json
├── fold02.json
├── fold03.json
└── fold04.json
```

## Classi Disponibili (50 classi)

Le 50 classi sono organizzate in 5 macro-categorie:

### Animals (10 classi)
- dog (0), rooster (1), pig (2), cow (3), frog (4), cat (5), hen (6), insects (7), sheep (8), crow (9)

### Natural soundscapes & water sounds (10 classi)
- rain (10), sea_waves (11), crackling_fire (12), crickets (13), chirping_birds (14), water_drops (15), wind (16), pouring_water (17), toilet_flush (18), thunderstorm (19)

### Human, non-speech sounds (10 classi)
- crying_baby (20), sneezing (21), clapping (22), breathing (23), coughing (24), footsteps (25), laughing (26), brushing_teeth (27), snoring (28), drinking_sipping (29)

### Interior/domestic sounds (10 classi)
- door_wood_knock (30), mouse_click (31), keyboard_typing (32), door_wood_creaks (33), can_opening (34), washing_machine (35), vacuum_cleaner (36), clock_alarm (37), clock_tick (38), glass_breaking (39)

### Exterior/urban noises (10 classi)
- helicopter (40), chainsaw (41), siren (42), car_horn (43), engine (44), train (45), church_bells (46), airplane (47), fireworks (48), hand_saw (49)

## Utilizzo

### Esempio Base

```python
from datautils.dataset_esc50 import ESC50Dataset

# Crea dataset con classi specifiche
dataset = ESC50Dataset(
    selected_classes=['dog', 'cat', 'rooster'],
    split='train',
    use_folds=True,
    train_folds=[0, 1, 2, 3],
    test_folds=[4]
)

print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.get_class_names()}")
print(f"Samples per class: {dataset.get_samples_per_class()}")

# Carica un sample
sample = dataset[0]
print(f"Audio shape: {sample['audio'].shape}")  # (80000,) per 5 sec @ 16kHz
print(f"Image shape: {sample['image'].shape}")   # (3, 224, 224)
print(f"Label: {sample['label']} - {sample['class_name']}")
```

### Configurazione per Federated Learning

```json
{
  "nodes": {
    "0": {
      "dataset": "ESC50",
      "selected_classes": ["dog", "cat", "rooster"],
      "use_folds": true,
      "train_folds": [0, 1, 2],
      "test_folds": [3]
    },
    "1": {
      "dataset": "ESC50",
      "selected_classes": ["chainsaw", "helicopter", "airplane"],
      "use_folds": true,
      "train_folds": [0, 1, 2],
      "test_folds": [3]
    }
  }
}
```

## Parametri del Dataset

### Parametri Principali

- `root_dir`: Directory root del dataset (default: `/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full`)
- `selected_classes`: Lista di classi da includere (nomi o indici)
- `excluded_classes`: Lista di classi da escludere
- `split`: 'train', 'test', o 'all'
- `split_ratio`: Rapporto train/test se `use_folds=False` (default: 0.8)

### Parametri Folds (ESC-50 Specific)

- `use_folds`: Se True, usa i fold ufficiali ESC-50 (default: True)
- `train_folds`: Lista di fold per training (default: [0, 1, 2, 3])
- `test_folds`: Lista di fold per testing (default: [4])

### Parametri Audio/Image

- `audio_sample_rate`: Sample rate target (default: 16000 Hz)
- `audio_duration`: Durata audio in secondi (default: 5.0)
- `image_size`: Dimensione immagini (default: (224, 224))

### Parametri Embedding

- `embedding_file`: Path al file di text embeddings (opzionale)
- `audio_embedding_file`: Path al file di audio embeddings (opzionale)

### Parametri Cache

- `enable_cache`: Abilita caching (default: False)
- `cache_dir`: Directory cache (default: '/tmp/esc50_cache')

## Features

### 1. Fold-based Splitting

ESC-50 fornisce 5 fold ufficiali per cross-validation. Puoi:
- Usare i fold ufficiali: `use_folds=True`
- Usare split casuale: `use_folds=False` + `split_ratio=0.8`

### 2. Class Filtering

Supporta selezione/esclusione classi:
```python
# Per nome
dataset = ESC50Dataset(selected_classes=['dog', 'cat', 'bird'])

# Per indice
dataset = ESC50Dataset(selected_classes=[0, 5, 14])

# Esclusione
dataset = ESC50Dataset(excluded_classes=['dog', 'cat'])
```

### 3. Caching

Per velocizzare il caricamento:
```python
dataset = ESC50Dataset(
    enable_cache=True,
    cache_dir='/tmp/esc50_cache'
)
```

### 4. Multimodal Data

Ogni sample contiene:
- `audio`: Waveform audio (torch.Tensor)
- `image`: Immagine rappresentativa (torch.Tensor)
- `label`: Label della classe (torch.Tensor)
- `text_emb`: Text embedding se disponibile
- `audio_emb`: Audio embedding se disponibile
- `caption`: Caption testuale se disponibile
- Metadata: file_id, class_name, fold, etc.

## Integrazione con FedA2V

Il dataset ESC-50 è completamente integrato con il server FedA2V:

```python
# In serverA2V.py
if node_config.dataset == "ESC50":
    node_dataset = ESC50Dataset(
        selected_classes=node_config.selected_classes,
        excluded_classes=node_config.excluded_classes,
        split=node_config.dataset_split,
        use_folds=node_config.use_folds,
        train_folds=node_config.train_folds,
        test_folds=node_config.test_folds,
        node_id=int(node_id)
    )
```

## Note

1. **Audio Duration**: ESC-50 ha clip di 5 secondi (vs 10 secondi di VEGAS)
2. **Images**: ESC-50 ha 4 immagini per classe (200 totali) condivise tra tutti i sample della stessa classe
3. **Folds**: I 5 fold contengono 40 sample ciascuno (400 totali per fold)
4. **Captions**: Disponibili per molti (ma non tutti) i sample

## Testing

Per testare il dataset:

```bash
cd /home/lpala/fedgfe/system/datautils
python dataset_esc50.py
```

## References

- ESC-50 Paper: Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification.
- Dataset URL: https://github.com/karolpiczak/ESC-50
