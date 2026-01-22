# VEGAS Dataset: Funzionalità num_samples

## Descrizione

È stata aggiunta la possibilità di limitare il numero di campioni per classe nel dataset VEGAS tramite il parametro `num_samples`.

## Caratteristiche

- **Limitazione per classe**: Il parametro `num_samples` limita il numero di campioni **per ogni classe** selezionata
- **Compatibile con selezione classi**: Funziona insieme a `selected_classes` per filtrare prima le classi e poi limitare i campioni
- **Applicabile a tutte le modalità**: Funziona con split train/val/test e stratificazione

## Utilizzo

### 1. Nel file JSON di configurazione

Aggiungi il parametro `num_samples` nella configurazione del nodo:

```json
{
  "nodes": {
    "0": {
      "dataset": "VEGAS",
      "selected_classes": ["baby_cry"],
      "num_samples": 50
    }
  }
}
```

### 2. Con più classi

Quando si selezionano più classi, `num_samples` viene applicato a ciascuna classe:

```json
{
  "nodes": {
    "0": {
      "dataset": "VEGAS",
      "selected_classes": ["baby_cry", "dog", "chainsaw"],
      "num_samples": 30
    }
  }
}
```

In questo esempio, verranno caricati **30 campioni per classe** (90 campioni totali).

### 3. Uso programmatico in Python

```python
from system.datautils.dataset_vegas import VEGASDataset

# Carica una classe con limite di campioni
dataset = VEGASDataset(
    selected_classes=['baby_cry'],
    num_samples=50,
    split='train'
)

# Carica più classi con limite di campioni
dataset = VEGASDataset(
    selected_classes=['baby_cry', 'dog', 'chainsaw'],
    num_samples=30,
    split='train'
)

# Senza limite (tutti i campioni disponibili)
dataset = VEGASDataset(
    selected_classes=['baby_cry'],
    split='train'
)
```

## Esempi di configurazione

### Esempio 1: Una classe con 50 campioni
File: `configs/a2v_vegas_1n_1c_num_samples.json`

```json
{
  "nodes": {
    "0": {
      "dataset": "VEGAS",
      "selected_classes": ["baby_cry"],
      "num_samples": 50
    }
  }
}
```

### Esempio 2: Tre classi con 30 campioni ciascuna
File: `configs/a2v_vegas_1n_multi_class_num_samples.json`

```json
{
  "nodes": {
    "0": {
      "dataset": "VEGAS",
      "selected_classes": ["baby_cry", "dog", "chainsaw"],
      "num_samples": 30
    }
  }
}
```

## Comportamento

1. **Ordine di applicazione**:
   - Prima vengono filtrate le classi (se `selected_classes` è specificato)
   - Poi vengono caricati tutti i campioni di ciascuna classe
   - Infine viene applicato il limite `num_samples` per classe

2. **Selezione dei campioni**:
   - Vengono selezionati i primi `num_samples` campioni per ogni classe
   - L'ordine è determinato dall'ordine nel file `video_info.csv` o dalla posizione nel filesystem

3. **Cache**:
   - La cache tiene conto del parametro `num_samples`
   - Modificando `num_samples` verrà generata una nuova cache

4. **Split train/val/test**:
   - Il limite viene applicato **prima** della divisione in split
   - Gli split vengono quindi calcolati sui campioni limitati

## Note importanti

- Se `num_samples` è `None` (valore di default), vengono caricati **tutti** i campioni disponibili
- Se una classe ha meno campioni di `num_samples`, vengono caricati tutti i campioni disponibili per quella classe
- Il parametro è opzionale: se non specificato, il comportamento è invariato (tutti i campioni)

## Classi disponibili in VEGAS

```python
CLASS_LABELS = {
    'baby_cry': 0,
    'chainsaw': 1,
    'dog': 2,
    'drum': 3,
    'fireworks': 4,
    'helicopter': 5,
    'printer': 6,
    'rail_transport': 7,
    'snoring': 8,
    'water_flowing': 9
}
```

## Compatibilità

Questa funzionalità è compatibile con:
- ✅ Selezione classi (`selected_classes`)
- ✅ Esclusione classi (`excluded_classes`)
- ✅ Split train/val/test
- ✅ Stratificazione
- ✅ Cache del dataset
- ✅ Folds (per ESC-50)
- ✅ Tutti i tipi di split (`split='train'`, `split='val'`, `split='test'`, `split='all'`, `split=None`)
