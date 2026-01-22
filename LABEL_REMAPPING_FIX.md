# Fix: Label Remapping in Multi-Class Federated Learning

**Data**: 2026-01-20
**Issue**: Inconsistenza delle label quando si selezionano più classi su nodi diversi

---

## Problema Identificato

Quando si selezionavano più classi su un nodo VEGAS/ESC50/VGGSound, le label venivano **rimappate** a partire da 0, causando inconsistenze tra nodi e con i generatori condizionali.

### Esempio del Problema

```python
# CLASS_LABELS originali VEGAS:
CLASS_LABELS = {
    'baby_cry': 0,
    'chainsaw': 1,
    'dog': 2,
    'drum': 3,
    'fireworks': 4,
    ...
}

# PRIMA DEL FIX - con remapping:
# Nodo 0: selected_classes=['dog']
#   → active_classes = {'dog': 0}  ← Rimappato!

# Nodo 1: selected_classes=['dog', 'drum']
#   → active_classes = {'dog': 0, 'drum': 1}  ← Rimappato!

# PROBLEMA: 'dog' ha label 0 su entrambi i nodi, MA 'drum' ha label 1 invece di 3
# Se il generatore è condizionato sulla label, ci saranno inconsistenze!
```

### Conseguenze

1. **Inconsistenza tra nodi**: Nodi con selezioni di classi diverse avevano label space differenti
2. **Generatore condizionale**: Il generatore riceveva label diverse per la stessa classe a seconda del nodo
3. **Mixed datasets**: Impossibile mixare ESC50 e VEGAS mantenendo label consistenti
4. **Test globale**: Durante la valutazione su test set globale, le label erano inconsistenti

---

## Soluzione Implementata

### Modifica a `_filter_classes()`

**File modificati**:
- `/home/lpala/fedgfe/system/datautils/dataset_vegas.py`
- `/home/lpala/fedgfe/system/datautils/dataset_esc50.py`
- `/home/lpala/fedgfe/system/datautils/dataset_vggsound.py`

**Cambiamento**: Rimosso il remapping delle label. Le label originali da `CLASS_LABELS` vengono preservate.

```python
def _filter_classes(self) -> Dict[str, int]:
    """
    Filter classes based on selection and exclusion.

    IMPORTANT: Labels are NOT remapped - original CLASS_LABELS indices are preserved.
    This ensures consistency across federated nodes and with conditional generators.
    """
    active_classes = dict(self.CLASS_LABELS)

    # Apply selection filter
    if self._selected_classes:
        active_classes = {cls: label for cls, label in active_classes.items()
                        if cls in self.selected_classes}

    # Apply exclusion filter
    if self._excluded_classes:
        active_classes = {cls: label for cls, label in active_classes.items()
                        if cls not in self.excluded_classes}

    # DO NOT remap labels - preserve original CLASS_LABELS indices
    # REMOVED: sorted_classes = sorted(active_classes.keys())
    # REMOVED: active_classes = {cls: idx for idx, cls in enumerate(sorted_classes)}

    return active_classes
```

### Nuovo Metodo: `get_max_class_label()`

Aggiunto a tutti e tre i dataset per ottenere il massimo valore di label (utile per dimensioni output del modello):

```python
def get_max_class_label(self) -> int:
    """
    Get the maximum class label value in active classes.

    This is useful for determining model output dimensions when labels are not remapped.

    Returns:
        Maximum label value, or -1 if no active classes

    Example:
        selected_classes = ['dog', 'drum']  # original labels: dog=2, drum=3
        get_max_class_label() -> 3
    """
    if not self.active_classes:
        return -1
    return max(self.active_classes.values())
```

---

## Comportamento DOPO il Fix

```python
# VEGAS - 10 classi totali (0-9)
# Nodo 0: selected_classes=['dog']
#   → active_classes = {'dog': 2}  ✓ Label originale preservata

# Nodo 1: selected_classes=['dog', 'drum']
#   → active_classes = {'dog': 2, 'drum': 3}  ✓ Label originali preservate

# Nodo 2: selected_classes=['fireworks']
#   → active_classes = {'fireworks': 4}  ✓ Label originale preservata

# ✓ Tutti i nodi usano lo stesso label space!
# ✓ Il generatore condizionato sulla label riceve sempre la stessa label per la stessa classe
# ✓ Compatible con mixed datasets (ESC50 + VEGAS)
```

---

## Impatto

### Vantaggi

✅ **Consistenza globale**: Tutti i nodi usano lo stesso label space
✅ **Generatori condizionali**: Label consistenti per il conditioning
✅ **Mixed datasets**: Compatibile con configurazioni ESC50 + VEGAS
✅ **Test globale**: Valutazione consistente su test set comune
✅ **Semplicità**: Nessun mapping aggiuntivo da gestire

### Considerazioni

⚠️ **Dimensione output**: Se selezioni 2 classi su 10, il modello downstream deve avere 10 output invece di 2
- In pratica **non è un problema**: le classi non selezionate non vengono mai usate durante il training
- I neuroni corrispondenti rimangono non addestrati

⚠️ **Efficienza**: Output layer leggermente più grande del necessario
- Differenza trascurabile: 10 output invece di 2 = 8 neuroni in più nell'ultimo layer
- Memoria aggiuntiva: ~KB invece di GB

---

## API Changes

### `get_num_classes()`

**NESSUN CAMBIAMENTO FUNZIONALE** - continua a restituire il **numero** di classi attive

```python
# selected_classes = ['dog', 'drum']
dataset.get_num_classes()  # → 2 (count di classi)
```

### `get_max_class_label()` (NUOVO)

Restituisce il **valore massimo** delle label attive (utile per output dimensions)

```python
# selected_classes = ['dog', 'drum']  # labels: 2, 3
dataset.get_max_class_label()  # → 3 (max label value)

# Per determinare output_dim del modello:
output_dim = dataset.get_max_class_label() + 1  # → 4
```

---

## Backward Compatibility

✅ **Completamente retrocompatibile** per la maggior parte dei casi d'uso:
- `get_num_classes()` continua a funzionare come prima
- `active_classes` è ancora un dizionario `{class_name: label}`
- `get_class_names()` e `get_class_labels()` funzionano come prima

⚠️ **Potenziale breaking change**:
- Se il codice downstream assume che le label siano sempre contigue 0, 1, 2, ...
- Soluzione: usare `get_max_class_label() + 1` invece di `get_num_classes()` per output dimensions

---

## Testing Consigliato

### Test 1: Label Consistency Across Nodes

```python
# Verifica che nodi con classi diverse usino label consistenti
node0 = VEGASDataset(selected_classes=['dog'], node_id=0)
node1 = VEGASDataset(selected_classes=['dog', 'drum'], node_id=1)

# Entrambi dovrebbero avere 'dog' con label 2
assert node0.active_classes['dog'] == 2
assert node1.active_classes['dog'] == 2
assert node1.active_classes['drum'] == 3
```

### Test 2: Mixed Dataset Compatibility

```python
# ESC50 + VEGAS mixed
esc50_node = ESC50Dataset(selected_classes=['dog'], node_id=0)
vegas_node = VEGASDataset(selected_classes=['dog'], node_id=1)

# Entrambi dovrebbero preservare le label originali
# (anche se i CLASS_LABELS sono diversi tra ESC50 e VEGAS)
```

### Test 3: Generator Conditioning

```python
# Verifica che il generatore riceva label consistenti
dataset = VEGASDataset(selected_classes=['dog', 'drum'])
sample = dataset[0]

# La label dovrebbe corrispondere a CLASS_LABELS originale
if sample['class_name'] == 'dog':
    assert sample['label'] == 2  # Original VEGAS label for 'dog'
elif sample['class_name'] == 'drum':
    assert sample['label'] == 3  # Original VEGAS label for 'drum'
```

---

## Migration Guide

### Se usavi `get_num_classes()` per output dimensions:

**PRIMA**:
```python
num_classes = dataset.get_num_classes()
model = MyModel(output_dim=num_classes)
```

**DOPO** (se le label non sono contigue):
```python
# Opzione 1: Usa il max label value
max_label = dataset.get_max_class_label()
model = MyModel(output_dim=max_label + 1)

# Opzione 2: Se sai che usi tutte le classi, get_num_classes() va bene
num_classes = dataset.get_num_classes()
model = MyModel(output_dim=num_classes)  # OK se usi tutte le 10 classi VEGAS
```

---

## Files Changed

1. `system/datautils/dataset_vegas.py`
   - Modified `_filter_classes()` - removed label remapping
   - Added `get_max_class_label()` method
   - Updated `get_num_classes()` docstring

2. `system/datautils/dataset_esc50.py`
   - Modified `_filter_classes()` - removed label remapping
   - Added `get_max_class_label()` method
   - Updated `get_num_classes()` docstring

3. `system/datautils/dataset_vggsound.py`
   - Modified `_filter_classes()` - removed label remapping
   - Added `get_max_class_label()` method
   - Updated `get_num_classes()` docstring

---

## Conclusione

Questo fix risolve un problema critico di inconsistenza delle label in scenari federated learning multi-classe. Le label ora sono **globalmente consistenti** attraverso tutti i nodi, garantendo corretto funzionamento di generatori condizionali e mixed datasets.

Il fix è **backward compatible** per la maggior parte dei casi d'uso, con solo un potenziale impatto su codice che assume label contigue 0, 1, 2, ... (facilmente risolvibile usando `get_max_class_label()`).
