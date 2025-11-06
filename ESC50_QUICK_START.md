# ESC-50 Quick Start Guide

## ✅ Implementazione Completata

Il dataset ESC-50 è stato completamente implementato e integrato nel framework FedA2V.

## 🚀 Quick Start

### 1. Verifica Test

```bash
cd /home/lpala/fedgfe
python test_esc50_dataset.py
```

**Output atteso**: ✅ All tests passed

### 2. Esempio Base

```python
from system.datautils.dataset_esc50 import ESC50Dataset

# Crea dataset
dataset = ESC50Dataset(
    selected_classes=['dog', 'cat', 'rooster'],
    split='train',
    use_folds=True,
    train_folds=[0, 1, 2, 3],
    test_folds=[4]
)

# Carica sample
sample = dataset[0]
print(f"Audio: {sample['audio'].shape}")        # (80000,)
print(f"Image: {sample['image'].shape}")        # (3, 224, 224)
print(f"Label: {sample['label']}")              # tensor(X)
print(f"Class: {sample['class_name']}")         # 'dog'
```

### 3. Configurazione Federated Learning

**File**: `configs/esc50_example.json`

```json
{
  "experiment": {
    "goal": "ESC50-Test"
  },
  "federation": {
    "algorithm": "FedA2V",
    "num_clients": 3
  },
  "nodes": {
    "0": {
      "dataset": "ESC50",
      "selected_classes": ["dog", "cat", "rooster"]
    },
    "1": {
      "dataset": "ESC50",
      "selected_classes": ["chainsaw", "helicopter", "airplane"]
    },
    "2": {
      "dataset": "ESC50",
      "selected_classes": ["fireworks", "rain", "thunderstorm"]
    }
  }
}
```

**Esecuzione**:
```bash
python system/main.py --config configs/esc50_example.json
```

## 📁 File Implementati

### Codice
- ✅ `system/datautils/dataset_esc50.py` - Dataset implementation
- ✅ `system/flcore/servers/serverA2V.py` - Server integration
- ✅ `system/utils/config_loader.py` - Config defaults

### Documentazione
- ✅ `system/datautils/ESC50_README.md` - Documentazione completa
- ✅ `ESC50_IMPLEMENTATION_SUMMARY.md` - Riepilogo implementazione
- ✅ `ESC50_QUICK_START.md` - Questa guida

### Configurazioni
- ✅ `configs/esc50_example.json` - Esempio configurazione
- ✅ `test_esc50_dataset.py` - Script di test

## 🎯 Le 50 Classi

### Animals (10)
dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow

### Natural Sounds (10)
rain, sea_waves, crackling_fire, crickets, chirping_birds, water_drops, wind, pouring_water, toilet_flush, thunderstorm

### Human Sounds (10)
crying_baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing_teeth, snoring, drinking_sipping

### Interior Sounds (10)
door_wood_knock, mouse_click, keyboard_typing, door_wood_creaks, can_opening, washing_machine, vacuum_cleaner, clock_alarm, clock_tick, glass_breaking

### Exterior Sounds (10)
helicopter, chainsaw, siren, car_horn, engine, train, church_bells, airplane, fireworks, hand_saw

## 📊 Parametri Chiave

### Fold Configuration (consigliato)
```json
{
  "use_folds": true,
  "train_folds": [0, 1, 2, 3],
  "test_folds": [4]
}
```

### Alternative: Split Ratio
```json
{
  "use_folds": false,
  "split_ratio": 0.8
}
```

### Class Selection
```json
{
  "selected_classes": ["dog", "cat", "rooster"],
  "excluded_classes": ["pig", "cow"]
}
```

## ✨ Features

- ✅ **50 classi** di suoni ambientali
- ✅ **2000 samples** totali (40 per classe)
- ✅ **5 fold ufficiali** per cross-validation
- ✅ **Audio**: 5 sec @ 16kHz (80,000 samples)
- ✅ **Images**: 224x224 RGB (4 per classe)
- ✅ **Class filtering**: selected/excluded
- ✅ **Caching**: performance optimization
- ✅ **Fed Learning**: node-specific configs

## 🔍 Verifica Status

```bash
# Test rapido
python -c "
import sys
sys.path.insert(0, 'system')
from datautils.dataset_esc50 import ESC50Dataset
dataset = ESC50Dataset(selected_classes=['dog', 'cat'], split='train')
print(f'✅ ESC-50 loaded: {len(dataset)} samples')
print(f'✅ Classes: {dataset.get_class_names()}')
sample = dataset[0]
print(f'✅ Audio shape: {sample[\"audio\"].shape}')
print(f'✅ Image shape: {sample[\"image\"].shape}')
print('✅ ESC-50 is fully functional!')
"
```

## 📚 Documentazione Completa

Per dettagli completi, consulta:
- `system/datautils/ESC50_README.md` - Guida completa
- `ESC50_IMPLEMENTATION_SUMMARY.md` - Dettagli implementazione

## 🎉 Pronto all'Uso!

L'implementazione ESC-50 è **completa, testata e pronta per la produzione**.

**Test Status**: ✅ All tests passed
**Integration**: ✅ FedA2V fully integrated
**Documentation**: ✅ Complete
