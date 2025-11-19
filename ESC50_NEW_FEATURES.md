# ESC50Dataset - Nuove Funzionalità

## Overview

Il dataset ESC50 è stato aggiornato con tre nuove funzionalità principali:

1. **Auto-creazione Split** (NEW!): Quando `split=None`, crea automaticamente train/val/test
2. **Ratio Personalizzabili**: Split train/val/test con ratio personalizzabili (default 70-20-10)
3. **Caricamento Selettivo**: Possibilità di caricare solo split specifici combinandoli

---

## 1. Auto-creazione Split (split=None) - NUOVO!

### Comportamento Default

Quando istanzi `ESC50Dataset` **senza specificare split** (o con `split=None`), il dataset crea automaticamente tutti e tre gli split accessibili come attributi `.train`, `.val`, `.test`.

### Esempio Base

```python
from system.datautils.dataset_esc50 import ESC50Dataset

# Auto-crea tutti e tre gli split
dataset = ESC50Dataset(
    root_dir="/path/to/esc50",
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2
)

# Accedi ai tre split
print(f"Train: {len(dataset.train)} samples")
print(f"Val:   {len(dataset.val)} samples")
print(f"Test:  {len(dataset.test)} samples")

# Usa i dataset separatamente
train_loader = DataLoader(dataset.train, batch_size=32)
val_loader = DataLoader(dataset.val, batch_size=32)
test_loader = DataLoader(dataset.test, batch_size=32)
```

Output:
```
Creating train split...
Creating validation split...
Creating test split...
Auto-split created: train=1400, val=200, test=400
Train: 1400 samples
Val:   200 samples
Test:  400 samples
```

### Vantaggi

✅ **Una sola chiamata** invece di tre
✅ **Parametri condivisi** automaticamente tra gli split
✅ **Meno verboso** e più leggibile
✅ **Backwards compatible** - codice esistente continua a funzionare

### Confronto: Prima vs Dopo

#### Prima (3 chiamate separate)
```python
train_ds = ESC50Dataset(root_dir="/path", split='train', train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
val_ds = ESC50Dataset(root_dir="/path", split='val', train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
test_ds = ESC50Dataset(root_dir="/path", split='test', train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
```

#### Dopo (1 chiamata)
```python
dataset = ESC50Dataset(root_dir="/path", train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
train_ds = dataset.train
val_ds = dataset.val
test_ds = dataset.test
```

### Singolo Split (comportamento esplicito)

Se hai bisogno di **un solo split**, specifica esplicitamente:

```python
# Solo training set
train_only = ESC50Dataset(
    root_dir="/path/to/esc50",
    split='train',  # Esplicito
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2
)
```

---

## 2. Ratio Personalizzabili (train_ratio, val_ratio, test_ratio)

### Parametri Nuovi

- `train_ratio` (float, default=0.7): Percentuale per training set (70%)
- `val_ratio` (float, default=0.1): Percentuale per validation set (10%)
- `test_ratio` (float, default=0.2): Percentuale per test set (20%)

### Esempi di Utilizzo

#### Split 70-20-10 (Default)

```python
from system.datautils.dataset_esc50 import ESC50Dataset

# Training set (70%)
train_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    split='train',
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True
)

# Validation set (10%)
val_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    split='val',
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True
)

# Test set (20%)
test_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    split='test',
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True
)

# Output:
# Train: ~1400 samples (70%)
# Val:   ~200 samples (10%)
# Test:  ~400 samples (20%)
```

#### Split 80-10-10

```python
train_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    split='train',
    use_folds=False,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify=True
)

val_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    split='val',
    use_folds=False,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify=True
)

test_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    split='test',
    use_folds=False,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify=True
)

# Output:
# Train: ~1600 samples (80%)
# Val:   ~200 samples (10%)
# Test:  ~200 samples (10%)
```

#### Split 60-20-20

```python
train_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    split='train',
    use_folds=False,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    stratify=True
)

# Output:
# Train: ~1200 samples (60%)
# Val:   ~400 samples (20%)
# Test:  ~400 samples (20%)
```

### Note

- I ratio devono sommare a 1.0 (con tolleranza ±0.01)
- Se la somma non è esattamente 1.0, vengono normalizzati automaticamente
- La stratificazione garantisce distribuzione bilanciata delle classi

---

## 2. Caricamento Selettivo (splits_to_load)

### Parametro Nuovo

- `splits_to_load` (Optional[List[str]], default=None): Lista degli split da caricare e combinare
  - Valori validi: `['train']`, `['val']`, `['test']`, `['train', 'val']`, `['train', 'val', 'test']`, `['all']`

### Esempi di Utilizzo

#### Caricare solo Train + Val

```python
# Utile per training con validation on-the-fly
train_val_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    split='all',  # Ignorato quando splits_to_load è specificato
    splits_to_load=['train', 'val'],
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True
)

# Output: ~1600 samples (70% + 10% = 80% del totale)
print(f"Combined train+val: {len(train_val_ds)} samples")
```

#### Caricare solo Train

```python
train_only_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    splits_to_load=['train'],
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True
)

# Equivalente a:
train_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    split='train',
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True
)
```

#### Caricare tutti gli split

```python
all_data_ds = ESC50Dataset(
    root_dir="/path/to/esc50",
    splits_to_load=['train', 'val', 'test'],
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True
)

# Output: 2000 samples (100% del dataset)
```

### Use Cases

1. **Training con Validation Set Integrato**
   ```python
   # Carica train+val per fare training con validation on-the-fly
   combined_ds = ESC50Dataset(
       splits_to_load=['train', 'val'],
       train_ratio=0.7,
       val_ratio=0.1,
       test_ratio=0.2
   )
   ```

2. **Data Augmentation**
   ```python
   # Usa train+val per augmentation, tieni test separato
   augment_ds = ESC50Dataset(
       splits_to_load=['train', 'val'],
       train_ratio=0.7,
       val_ratio=0.1,
       test_ratio=0.2
   )
   ```

3. **Cross-Validation Personalizzata**
   ```python
   # Combina split per implementare CV personalizzato
   fold1_ds = ESC50Dataset(
       splits_to_load=['train'],
       train_ratio=0.8,
       val_ratio=0.0,
       test_ratio=0.2
   )
   ```

---

## Compatibilità con Versione Precedente

### Parametro Deprecato: `split_ratio`

Il parametro `split_ratio` è stato deprecato ma è ancora supportato per compatibilità:

```python
# Vecchia API (ancora funzionante ma genera warning)
train_ds = ESC50Dataset(
    split='train',
    split_ratio=0.8,  # DEPRECATED: usa train_ratio invece
    val_ratio=0.1,
    use_folds=False
)

# Output: Warning: split_ratio is deprecated. Use train_ratio, val_ratio, test_ratio instead.
```

Conversione automatica:
- `split_ratio=0.8` → `train_ratio=0.7`, `val_ratio=0.1`, `test_ratio=0.2`
- `split_ratio` definiva (train+val) vs test
- Nuova API è più esplicita e flessibile

---

## Validazione e Controlli

### Validazione Ratio

```python
# Ratio che non sommano a 1.0 vengono normalizzati automaticamente
ds = ESC50Dataset(
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.3,  # Somma = 1.1
    use_folds=False
)

# Output Warning: train_ratio + val_ratio + test_ratio = 1.100, normalizing to 1.0
# Ratios normalizzati: 0.545, 0.182, 0.273
```

### Validazione splits_to_load

```python
# Split invalidi generano errore
try:
    ds = ESC50Dataset(
        splits_to_load=['training', 'validation']  # ERRORE!
    )
except ValueError as e:
    print(e)  # Invalid split in splits_to_load: training
```

---

## Aggiornamento Notebook Test

Il notebook `test_esc50_dataset_functionality.ipynb` è stato aggiornato per usare la nuova API:

```python
# Cella 3: Create Datasets con 70-20-10 split
train_dataset = ESC50Dataset(
    root_dir=root_dir,
    text_embedding_file=text_embedding_file,
    split='train',
    use_folds=False,
    train_ratio=0.7,  # 70%
    val_ratio=0.1,    # 10%
    test_ratio=0.2,   # 20%
    stratify=True,
    node_id=42
)
```

Sezione 9b aggiunta: **DataLoader Consistency Verification** per verificare che i DataLoader restituiscano il numero corretto di campioni.

---

## Migration Guide

### Da Vecchia a Nuova API

#### Prima (usando fold fissi)
```python
train_ds = ESC50Dataset(
    split='train',
    use_folds=True,
    train_folds=[0, 1, 2],
    val_folds=[3],
    test_folds=[4]
)
```

#### Dopo (usando ratio personalizzabili)
```python
train_ds = ESC50Dataset(
    split='train',
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True,
    node_id=42
)
```

### Da split_ratio a train/val/test ratio

#### Prima
```python
train_ds = ESC50Dataset(
    split='train',
    split_ratio=0.8,  # 80% train+val, 20% test
    val_ratio=0.1,
    use_folds=False
)
```

#### Dopo
```python
train_ds = ESC50Dataset(
    split='train',
    train_ratio=0.7,  # Esplicito: 70% train
    val_ratio=0.1,    # 10% val
    test_ratio=0.2,   # 20% test
    use_folds=False
)
```

---

## Testing

Eseguire il test con:

```bash
cd /home/lpala/fedgfe
python test_esc50_custom_splits.py
```

Test inclusi:
1. Test ratio 70-20-10
2. Test ratio 80-10-10
3. Test splits_to_load (train+val)
4. Test single split via splits_to_load
5. Test backwards compatibility

---

## Esempi Completi

### Esempio 1: Esperimento con 10 Classi e Split 70-20-10

```python
from system.datautils.dataset_esc50 import ESC50Dataset

root_dir = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full"
selected_classes = ["dog", "rooster", "pig", "cow", "frog",
                   "cat", "hen", "insects", "sheep", "crow"]

# Training set
train_ds = ESC50Dataset(
    root_dir=root_dir,
    selected_classes=selected_classes,
    split='train',
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True,
    node_id=42
)

# Validation set
val_ds = ESC50Dataset(
    root_dir=root_dir,
    selected_classes=selected_classes,
    split='val',
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True,
    node_id=42
)

# Test set
test_ds = ESC50Dataset(
    root_dir=root_dir,
    selected_classes=selected_classes,
    split='test',
    use_folds=False,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True,
    node_id=42
)

print(f"Train: {len(train_ds)} samples")
print(f"Val:   {len(val_ds)} samples")
print(f"Test:  {len(test_ds)} samples")

# Output:
# Train: 280 samples (70% di 400)
# Val:   40 samples (10% di 400)
# Test:  80 samples (20% di 400)
```

### Esempio 2: Caricamento Combinato per Federated Learning

```python
# Nodo 1: carica train+val per training
node1_ds = ESC50Dataset(
    root_dir=root_dir,
    selected_classes=["dog", "cat", "rooster"],
    splits_to_load=['train', 'val'],
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True,
    node_id=1
)

# Nodo 2: carica train+val per training
node2_ds = ESC50Dataset(
    root_dir=root_dir,
    selected_classes=["pig", "cow", "sheep"],
    splits_to_load=['train', 'val'],
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True,
    node_id=2
)

# Server: carica test per evaluation globale
server_test_ds = ESC50Dataset(
    root_dir=root_dir,
    splits_to_load=['test'],
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True,
    node_id=0
)

print(f"Node 1 (train+val): {len(node1_ds)} samples")
print(f"Node 2 (train+val): {len(node2_ds)} samples")
print(f"Server (test only): {len(server_test_ds)} samples")
```

---

## Conclusioni

Le nuove funzionalità permettono:

1. ✅ **Flessibilità**: Ratio personalizzabili per qualsiasi split
2. ✅ **Efficienza**: Caricamento selettivo solo degli split necessari
3. ✅ **Federated Learning**: Combinazione flessibile di split per scenari FL
4. ✅ **Compatibilità**: Supporto per codice legacy via parametri deprecated
5. ✅ **Riproducibilità**: Seed consistente via `node_id` parameter

Per domande o issue, consultare il codice sorgente in:
`/home/lpala/fedgfe/system/datautils/dataset_esc50.py`
