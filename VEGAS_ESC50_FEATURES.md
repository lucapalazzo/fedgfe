# Implementazione Feature ESC50 in VEGAS Dataset

## Sommario

Questo documento descrive le feature implementate in `VEGASDataset` per renderlo compatibile con le capacità di gestione degli split di `ESC50Dataset`.

## Feature Implementate

### 1. Nuovi Parametri di Split Ratio

**Parametri aggiunti:**
- `train_ratio` (default: 0.7): Percentuale di dati per il training (70%)
- `val_ratio` (default: 0.1): Percentuale di dati per la validazione (10%)
- `test_ratio` (default: 0.2): Percentuale di dati per il test (20%)

**Retrocompatibilità:**
- Il parametro legacy `split_ratio` è ancora supportato ma deprecato
- Viene emesso un warning quando si usa `split_ratio`
- I nuovi parametri vengono normalizzati automaticamente se non sommano a 1.0

**Esempio:**
```python
dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry'],
    split='train',
    train_ratio=0.6,   # 60% training
    val_ratio=0.2,     # 20% validation
    test_ratio=0.2,    # 20% test
    stratify=True
)
```

### 2. Auto-creazione Split con `split=None`

**Funzionalità:**
Quando `split=None`, il dataset crea automaticamente tre split accessibili come attributi:
- `dataset.train` - Split di training
- `dataset.val` - Split di validazione
- `dataset.test` - Split di test

**Metodo aggiunto:**
- `_create_all_splits(**kwargs)`: Crea automaticamente i tre split

**Esempio:**
```python
# Crea automaticamente train/val/test
dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split=None,  # Auto-crea gli split
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Accedi agli split
print(f"Train: {len(dataset.train)} samples")
print(f"Val: {len(dataset.val)} samples")
print(f"Test: {len(dataset.test)} samples")
```

### 3. Parametro `splits_to_load`

**Funzionalità:**
Permette di combinare dati da più split in un unico dataset.

**Valori validi:** `['train', 'val', 'test', 'all']`

**Esempio:**
```python
# Carica train + val combinati (utile per il fine-tuning)
combined_dataset = VEGASDataset(
    selected_classes=['dog', 'baby_cry'],
    splits_to_load=['train', 'val'],
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### 4. Supporto per Fold

**Parametri aggiunti:**
- `use_folds` (bool): Abilita l'uso di fold personalizzati
- `train_folds` (List[int]): Indici dei fold per training
- `val_folds` (List[int]): Indici dei fold per validation
- `test_folds` (List[int]): Indici dei fold per test

**Nota:** VEGAS non ha fold predefiniti come ESC50, ma l'infrastruttura è pronta per supportarli quando necessario.

**Esempio (quando saranno disponibili i fold):**
```python
dataset_train = VEGASDataset(
    selected_classes=['dog', 'baby_cry'],
    split='train',
    use_folds=True,
    train_folds=[0, 1, 2],
    test_folds=[3]
)
```

### 5. Miglioramenti a `_apply_split`

**Modifiche:**
- Calcolo più robusto dei rapporti di split
- Supporto per i nuovi parametri train_ratio/val_ratio/test_ratio
- Gestione corretta dei casi edge (es. val_ratio=0)
- Compatibilità con use_folds

**Caratteristiche:**
- Stratificazione migliorata usando sklearn.train_test_split
- Seed riproducibile basato su node_id per federated learning
- Normalizzazione automatica dei ratio se non sommano a 1.0

### 6. Metodo `_load_samples_from_disk`

**Nuovo metodo:**
Estratto da `_load_samples` per separare la logica di caricamento da disco dalla logica di splitting.

**Benefici:**
- Codice più modulare e manutenibile
- Facilita il supporto di `splits_to_load`
- Migliore riusabilità del codice

## Metodi Utility Esistenti (Mantenuti)

### Verifica Stratificazione
```python
train_dataset.verify_stratification(val_dataset, tolerance=0.05)
```

### Statistiche Split
```python
dataset.print_split_statistics()
```

### Distribuzione Classi
```python
distribution = dataset.get_class_distribution()
samples_per_class = dataset.get_samples_per_class()
```

## Pattern d'Uso Consigliati

### 1. Training Standard con Validation
```python
# Approccio 1: Creare split separati
train_ds = VEGASDataset(split='train', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
val_ds = VEGASDataset(split='val', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
test_ds = VEGASDataset(split='test', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# Approccio 2: Auto-creazione con split=None (CONSIGLIATO)
dataset = VEGASDataset(split=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train_loader = DataLoader(dataset.train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset.val, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset.test, batch_size=32, shuffle=False)
```

### 2. Fine-tuning con Train+Val
```python
# Combina train e val per il fine-tuning
finetune_ds = VEGASDataset(
    splits_to_load=['train', 'val'],
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
test_ds = VEGASDataset(split='test', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### 3. Federated Learning con Node ID
```python
# Ogni nodo ottiene gli stessi split riproducibili
node0_train = VEGASDataset(split='train', node_id=0, stratify=True)
node1_train = VEGASDataset(split='train', node_id=1, stratify=True)
# node_id influenza il seed casuale per split riproducibili ma diversi
```

## Compatibilità

### Backward Compatibility
- Tutti i parametri legacy sono ancora supportati
- `split_ratio` funziona ancora ma emette un warning
- Default dei parametri mantenuti per non rompere codice esistente

### Forward Compatibility
- L'infrastruttura per i fold è pronta anche se VEGAS non li usa ancora
- Facile estensione futura per nuove feature

## Test

Eseguire i test con:
```bash
python tests/test_vegas_esc50_features.py
```

Test coperti:
1. Auto-creazione split con split=None
2. Custom ratios (es. 60-20-20)
3. Parametro splits_to_load
4. Compatibilità legacy con split_ratio
5. Verifica stratificazione
6. Distribuzione classi
7. Edge cases (val_ratio=0, ratios non normalizzati)

## Differenze con ESC50

### Similitudini
- Stessa API per train_ratio/val_ratio/test_ratio
- Stesso meccanismo di auto-split con split=None
- Stessa logica di stratificazione
- Stessi metodi utility (verify_stratification, print_split_statistics, etc.)

### Differenze
- ESC50 ha fold predefiniti (fold00.json - fold04.json)
- VEGAS carica da struttura directory (class_name/audios, class_name/img, etc.)
- VEGAS supporta video oltre ad audio e immagini
- VEGAS usa video_id invece di file_id come identificatore

## Note di Implementazione

### Cache
La cache considera i nuovi parametri nel calcolo della cache key:
```python
config_str = f"{self.root_dir}_{self.active_classes}_{self.split}_{self.split_ratio}_{self.node_id}"
```

### Logging
- Warning per parametri deprecati (split_ratio)
- Info per normalizzazione automatica dei ratio
- Info per creazione degli split automatici

### Validazione
- Verifica che splits_to_load contenga valori validi
- Normalizzazione automatica se train_ratio+val_ratio+test_ratio ≠ 1.0
- Gestione corretta di val_ratio=0 (solo train/test split)

## Conclusioni

Il dataset VEGAS ora supporta tutte le principali feature di gestione degli split di ESC50, mantenendo la retrocompatibilità e aggiungendo flessibilità per scenari di training avanzati.
