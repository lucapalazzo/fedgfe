# ESC-50 Dataset Implementation Summary

## 📋 Riepilogo delle Modifiche

Implementato il supporto completo per il dataset ESC-50 (Environmental Sound Classification - 50 classes) nel framework FedA2V per Audio2Visual federated learning.

## 🆕 File Creati

### 1. Dataset Implementation
**File**: `system/datautils/dataset_esc50.py`

Implementazione completa del dataset ESC-50 con le seguenti features:
- ✅ Caricamento audio e immagini
- ✅ Supporto fold-based splitting (5 fold ufficiali)
- ✅ Class filtering (selected/excluded classes)
- ✅ Caching per performance
- ✅ Support per text/audio embeddings
- ✅ Integrazione con DataLoader PyTorch
- ✅ Metadata completi (captions, fold info, etc.)

**Caratteristiche Principali**:
```python
class ESC50Dataset(Dataset):
    - 50 classi di suoni ambientali
    - 2000 samples totali (40 per classe)
    - 5 fold ufficiali per cross-validation
    - Audio: 5 secondi @ 16kHz
    - Images: 224x224 (4 per classe)
```

### 2. Documentation
**File**: `system/datautils/ESC50_README.md`

Documentazione completa che include:
- Panoramica del dataset
- Lista completa delle 50 classi
- Esempi di utilizzo
- Parametri di configurazione
- Integrazione con FedA2V
- Testing instructions

### 3. Configuration Examples
**File**: `configs/esc50_example.json`

Configurazione di esempio per esperimenti con ESC-50:
```json
{
  "nodes": {
    "0": {
      "dataset": "ESC50",
      "selected_classes": ["dog", "cat", "rooster"],
      "use_folds": true,
      "train_folds": [0, 1, 2],
      "test_folds": [3]
    }
  }
}
```

### 4. Test Script
**File**: `test_esc50_dataset.py`

Script di test completo che verifica:
- ✅ Caricamento class labels
- ✅ Creazione dataset con classi specifiche
- ✅ Caricamento samples
- ✅ Funzionamento con tutte le 50 classi
- ✅ Fold-based splitting
- ✅ Class exclusion

## 🔧 File Modificati

### 1. Server Audio2Visual
**File**: `system/flcore/servers/serverA2V.py`

**Modifiche**:
```python
# Aggiunto import
from datautils.dataset_esc50 import ESC50Dataset

# Aggiunto supporto in create_clients()
elif node_config.dataset == "ESC50":
    selected_classes = getattr(node_config, 'selected_classes', None)
    excluded_classes = getattr(node_config, 'excluded_classes', None)
    split = getattr(node_config, 'dataset_split', 'train')
    use_folds = getattr(node_config, 'use_folds', True)
    train_folds = getattr(node_config, 'train_folds', [0, 1, 2, 3])
    test_folds = getattr(node_config, 'test_folds', [4])

    node_dataset = ESC50Dataset(
        selected_classes=selected_classes,
        excluded_classes=excluded_classes,
        split=split,
        use_folds=use_folds,
        train_folds=train_folds,
        test_folds=test_folds,
        node_id=int(node_id)
    )
```

### 2. Config Loader Defaults
**File**: `system/utils/config_loader.py`

**Modifiche**:
```python
'node': {
    'dataset_split': 'train',
    'pretext_tasks': [],
    'task_type': 'classification',
    'selected_classes': None,
    'excluded_classes': None,
    # ... altri parametri ...
    # ESC-50 specific parameters
    'use_folds': True,
    'train_folds': [0, 1, 2, 3],
    'test_folds': [4]
}
```

## 📊 Struttura del Dataset ESC-50

```
/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full/
├── audio/
│   ├── fold00/        # 400 files
│   ├── fold01/        # 400 files
│   ├── fold02/        # 400 files
│   ├── fold03/        # 400 files
│   └── fold04/        # 400 files
├── image/
│   └── cat_*.png      # 200 images (4 per class)
├── class_labels.json  # Mapping classe → indice
├── captions.json      # Captions testuali
├── fold00.json        # Metadata fold 0
├── fold01.json        # Metadata fold 1
├── fold02.json        # Metadata fold 2
├── fold03.json        # Metadata fold 3
└── fold04.json        # Metadata fold 4
```

## 🎯 Le 50 Classi ESC-50

### Animals (10 classi)
`dog`, `rooster`, `pig`, `cow`, `frog`, `cat`, `hen`, `insects`, `sheep`, `crow`

### Natural Soundscapes (10 classi)
`rain`, `sea_waves`, `crackling_fire`, `crickets`, `chirping_birds`, `water_drops`, `wind`, `pouring_water`, `toilet_flush`, `thunderstorm`

### Human Non-Speech (10 classi)
`crying_baby`, `sneezing`, `clapping`, `breathing`, `coughing`, `footsteps`, `laughing`, `brushing_teeth`, `snoring`, `drinking_sipping`

### Interior/Domestic (10 classi)
`door_wood_knock`, `mouse_click`, `keyboard_typing`, `door_wood_creaks`, `can_opening`, `washing_machine`, `vacuum_cleaner`, `clock_alarm`, `clock_tick`, `glass_breaking`

### Exterior/Urban (10 classi)
`helicopter`, `chainsaw`, `siren`, `car_horn`, `engine`, `train`, `church_bells`, `airplane`, `fireworks`, `hand_saw`

## ✅ Testing Results

```
ESC-50 Dataset Test Results:
✓ Class labels loading: 50 classes loaded
✓ Dataset creation: 72 samples (3 classes)
✓ Sample loading: Audio [80000], Image [3, 224, 224]
✓ Full dataset: 2000 samples, 50 classes
✓ Fold splitting: 64 train + 16 test = 80 total
✓ Class exclusion: Working correctly
```

## 🚀 Utilizzo

### Esempio Base Python

```python
from datautils.dataset_esc50 import ESC50Dataset

# Crea dataset
dataset = ESC50Dataset(
    selected_classes=['dog', 'cat', 'rooster'],
    split='train',
    use_folds=True,
    train_folds=[0, 1, 2, 3],
    test_folds=[4]
)

# Accedi ai dati
sample = dataset[0]
print(f"Audio: {sample['audio'].shape}")
print(f"Image: {sample['image'].shape}")
print(f"Label: {sample['label']} - {sample['class_name']}")
```

### Configurazione JSON per Federated Learning

```json
{
  "experiment": {
    "goal": "ESC50-FedA2V-Experiment"
  },
  "federation": {
    "algorithm": "FedA2V",
    "num_clients": 3
  },
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
    },
    "2": {
      "dataset": "ESC50",
      "selected_classes": ["fireworks", "rain", "thunderstorm"],
      "use_folds": true,
      "train_folds": [0, 1, 2],
      "test_folds": [3]
    }
  }
}
```

### Esecuzione Test

```bash
# Test dataset
python test_esc50_dataset.py

# Test con configurazione
python system/main.py --config configs/esc50_example.json
```

## 📝 Features Implementate

### 1. Fold-Based Splitting ✅
- Supporto per i 5 fold ufficiali ESC-50
- Configurabile train/test folds
- Alternative: split casuale con ratio

### 2. Class Filtering ✅
- Selezione classi per nome o indice
- Esclusione classi
- Remapping automatico delle label

### 3. Multimodal Data ✅
- Audio: 5 sec @ 16kHz (80,000 samples)
- Image: 224x224 RGB
- Text embeddings (opzionale)
- Audio embeddings (opzionale)
- Captions (quando disponibili)

### 4. Performance Optimization ✅
- Caching per sample metadata
- Lazy loading di audio/image
- DataLoader ottimizzato con collate_fn

### 5. Federated Learning Ready ✅
- Node-specific configurations
- Reproducible splits con node_id seed
- Balanced class distribution

## 🎓 Differenze VEGAS vs ESC-50

| Feature | VEGAS | ESC-50 |
|---------|-------|--------|
| Classi | 10 | 50 |
| Samples totali | ~variabile | 2000 |
| Samples per classe | ~variabile | 40 |
| Audio duration | 10 sec | 5 sec |
| Fold ufficiali | No | Sì (5 fold) |
| Images | Per sample | Per classe (4 shared) |
| Captions | Sì | Sì (parziali) |
| Video | Opzionale | No |

## 📚 References

- **ESC-50 Paper**: Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification.
- **Dataset Repository**: https://github.com/karolpiczak/ESC-50
- **Implementation**: Based on VEGASDataset structure

## ✨ Next Steps

Possibili miglioramenti futuri:
1. Generazione text embeddings per tutte le 50 classi
2. Generazione audio embeddings pre-computed
3. Supporto per augmentation specifiche audio
4. Integration con altri modelli audio (AST, Wav2Vec2, etc.)
5. Metrics specifiche per sound classification

## 🏆 Conclusioni

Implementazione completa e testata del dataset ESC-50 per federated learning Audio2Visual. Il dataset è ora completamente integrato nel framework e pronto per esperimenti.

**Test Status**: ✅ All tests passed
**Integration Status**: ✅ Fully integrated
**Documentation Status**: ✅ Complete
**Ready for Production**: ✅ Yes
