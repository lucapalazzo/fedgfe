# ESC-50 Dataset - Stratified Train/Val/Test Split

## Panoramica

Il dataset ESC-50 ora supporta lo split stratificato train/validation/test utilizzando `train_test_split` di scikit-learn. La stratificazione garantisce che la distribuzione delle classi sia mantenuta uniforme tra i vari split.

## Caratteristiche Principali

✅ **Split Stratificato**: Usa `sklearn.model_selection.train_test_split` per mantenere la distribuzione delle classi
✅ **Validation Split**: Supporto per split train/val/test (non solo train/test)
✅ **Fold-based o Ratio-based**: Funziona sia con fold ufficiali ESC-50 che con split percentuali
✅ **Verifiche Integrate**: Metodi per verificare la stratificazione tra split
✅ **Statistiche Dettagliate**: Visualizzazione della distribuzione delle classi per split

## Parametri Nuovi

### `val_ratio` (float, default=0.1)
Percentuale del dataset da usare per validation. Viene estratta dal set totale.

**Esempio**:
- `split_ratio=0.7` → 70% train+val, 30% test
- `val_ratio=0.15` → 15% validation dal totale
- Risultato: 55% train, 15% val, 30% test

### `val_folds` (Optional[List[int]], default=None)
Lista di fold da usare per validation quando `use_folds=True`.

**Esempio**:
```python
train_folds=[0, 1, 2]  # 60% train
val_folds=[3]          # 20% validation
test_folds=[4]         # 20% test
```

### `stratify` (bool, default=True)
Abilita la stratificazione per mantenere la distribuzione delle classi uniforme.

## Esempi di Utilizzo

### 1. Split Stratificato Percentuale (Raccomandato)

```python
from datautils.dataset_esc50 import ESC50Dataset

# Train dataset (55% dei dati)
train_dataset = ESC50Dataset(
    selected_classes=["dog", "rooster", "pig"],
    split='train',
    split_ratio=0.7,   # 70% per train+val combinato
    val_ratio=0.15,    # 15% del totale per validation
    stratify=True,     # Abilita stratificazione
    use_folds=False
)

# Validation dataset (15% dei dati)
val_dataset = ESC50Dataset(
    selected_classes=["dog", "rooster", "pig"],
    split='val',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True,
    use_folds=False
)

# Test dataset (30% dei dati rimanenti)
test_dataset = ESC50Dataset(
    selected_classes=["dog", "rooster", "pig"],
    split='test',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True,
    use_folds=False
)

# Verifica stratificazione
train_dataset.print_split_statistics()
val_dataset.print_split_statistics()
test_dataset.print_split_statistics()

# Verifica che le distribuzioni siano simili
train_dataset.verify_stratification(val_dataset, tolerance=0.05)
```

### 2. Split con Fold Ufficiali ESC-50

```python
# Train: folds 0, 1, 2 (60%)
train_dataset = ESC50Dataset(
    split='train',
    use_folds=True,
    train_folds=[0, 1, 2]
)

# Validation: fold 3 (20%)
val_dataset = ESC50Dataset(
    split='val',
    use_folds=True,
    val_folds=[3]  # Specifica esplicitamente il fold per validation
)

# Test: fold 4 (20%)
test_dataset = ESC50Dataset(
    split='test',
    use_folds=True,
    test_folds=[4]
)
```

### 3. Split Senza Validation

```python
# Solo train/test (come prima)
train_dataset = ESC50Dataset(
    split='train',
    split_ratio=0.8,
    val_ratio=0.0,  # Nessuna validation
    stratify=True
)

test_dataset = ESC50Dataset(
    split='test',
    split_ratio=0.8,
    val_ratio=0.0,
    stratify=True
)
```

## Nuovi Metodi

### `get_class_distribution() -> Dict[str, float]`
Restituisce la distribuzione percentuale delle classi nel dataset.

```python
distribution = train_dataset.get_class_distribution()
# {'dog': 33.3, 'rooster': 33.3, 'pig': 33.4}
```

### `print_split_statistics()`
Stampa statistiche dettagliate sul dataset corrente.

```python
train_dataset.print_split_statistics()

# Output:
# === ESC-50 Dataset Statistics (train split) ===
# Total samples: 110
# Number of classes: 3
# Stratified: True
#
# Class distribution:
#   dog: 37 samples (33.64%)
#   pig: 36 samples (32.73%)
#   rooster: 37 samples (33.64%)
```

### `verify_stratification(other_dataset, tolerance=0.05) -> bool`
Verifica che due dataset abbiano distribuzioni di classi simili.

```python
is_stratified = train_dataset.verify_stratification(val_dataset, tolerance=0.05)

# Output:
# === Stratification Verification ===
# ✓ dog: 33.64% vs 33.33% (diff: 0.31%)
# ✓ pig: 32.73% vs 33.33% (diff: 0.61%)
# ✓ rooster: 33.64% vs 33.33% (diff: 0.31%)
#
# Stratification PASSED (tolerance: 5.0%)
```

## Come Funziona la Stratificazione

### Metodo sklearn.train_test_split

1. **Primo Split**: Divide i dati in (train+val) e test
   - `train_test_split(samples, test_size=1.0-split_ratio, stratify=labels)`
   - Mantiene la proporzione delle classi

2. **Secondo Split**: Divide (train+val) in train e val
   - `train_test_split(train_val_samples, test_size=val_ratio/split_ratio, stratify=labels)`
   - Mantiene la proporzione delle classi anche nel validation

### Esempio con Numeri

Dataset: 100 samples per classe (3 classi = 300 totali)

**Configurazione**:
- `split_ratio=0.7` → 70% train+val = 210 samples
- `val_ratio=0.15` → 15% validation = 45 samples
- Test = 30% = 90 samples

**Risultato con Stratificazione**:

| Classe | Train (55%) | Val (15%) | Test (30%) | Totale |
|--------|-------------|-----------|------------|--------|
| dog    | 55          | 15        | 30         | 100    |
| pig    | 55          | 15        | 30         | 100    |
| rooster| 55          | 15        | 30         | 100    |
| **TOTALE** | **165** | **45**    | **90**     | **300** |

Ogni classe ha la stessa proporzione in tutti gli split!

## Test della Stratificazione

Esegui lo script di test:

```bash
cd /home/lpala/fedgfe
python test_esc50_stratification.py
```

**Output atteso**:
```
================================================================================
Testing ESC-50 Stratified Split
================================================================================

=== ESC-50 Dataset Statistics (train split) ===
Total samples: 110
Number of classes: 5
Stratified: True

Class distribution:
  cow: 22 samples (20.00%)
  dog: 22 samples (20.00%)
  frog: 22 samples (20.00%)
  pig: 22 samples (20.00%)
  rooster: 22 samples (20.00%)

...

=== Stratification Verification ===
✓ cow: 20.00% vs 20.00% (diff: 0.00%)
✓ dog: 20.00% vs 20.00% (diff: 0.00%)
✓ frog: 20.00% vs 20.00% (diff: 0.00%)
✓ pig: 20.00% vs 20.00% (diff: 0.00%)
✓ rooster: 20.00% vs 20.00% (diff: 0.00%)

Stratification PASSED (tolerance: 10.0%)
```

## Federated Learning

Per federated learning con validation:

```python
# Node 0: Train e validation locali
node0_train = ESC50Dataset(
    selected_classes=["airplane", "helicopter"],
    split='train',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True,
    node_id=0  # Per split consistente
)

node0_val = ESC50Dataset(
    selected_classes=["airplane", "helicopter"],
    split='val',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True,
    node_id=0  # Stesso node_id per split consistente
)

# Node 1: Classi diverse
node1_train = ESC50Dataset(
    selected_classes=["dog", "cat"],
    split='train',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True,
    node_id=1
)

node1_val = ESC50Dataset(
    selected_classes=["dog", "cat"],
    split='val',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True,
    node_id=1
)
```

## Confronto: Prima vs Dopo

### Prima (Senza Stratificazione)
```python
train_dataset = ESC50Dataset(split='train', split_ratio=0.8)
test_dataset = ESC50Dataset(split='test', split_ratio=0.8)
# NO validation split
# NO verifica della distribuzione
```

### Dopo (Con Stratificazione)
```python
# Tre split invece di due
train_dataset = ESC50Dataset(
    split='train',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True
)
val_dataset = ESC50Dataset(
    split='val',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True
)
test_dataset = ESC50Dataset(
    split='test',
    split_ratio=0.7,
    val_ratio=0.15,
    stratify=True
)

# Verifica stratificazione
train_dataset.verify_stratification(val_dataset)
val_dataset.verify_stratification(test_dataset)
```

## Best Practices

1. **Usa sempre `stratify=True`** per dataset sbilanciati
2. **Verifica la stratificazione** con `verify_stratification()`
3. **Stampa le statistiche** con `print_split_statistics()` per debug
4. **Usa lo stesso `node_id`** per split consistenti in federated learning
5. **Validation set**: 10-20% del totale è generalmente sufficiente
6. **Test set**: 20-30% del totale per valutazione robusta

## Troubleshooting

### Errore: "The least populated class in y has only 1 member"

**Causa**: Alcune classi hanno troppo pochi sample per la stratificazione.

**Soluzione**:
```python
# Opzione 1: Disabilita stratificazione
dataset = ESC50Dataset(stratify=False)

# Opzione 2: Aumenta il numero di classi selezionate
dataset = ESC50Dataset(selected_classes=more_classes)
```

### Validazione fallita (diff > tolerance)

**Causa**: Split non perfettamente stratificato (normale con piccoli dataset).

**Soluzione**:
```python
# Aumenta la tolleranza
train_dataset.verify_stratification(val_dataset, tolerance=0.10)  # 10% invece di 5%
```

## File Modificati

- `system/datautils/dataset_esc50.py`: Implementazione stratificazione
- `test_esc50_stratification.py`: Script di test
- `ESC50_STRATIFICATION.md`: Questa documentazione

## Dipendenze

```bash
pip install scikit-learn
```

Già incluso in: `requirements.txt`
