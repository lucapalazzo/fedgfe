# Per-Class Outputs - Usage Guide

Questo documento spiega come utilizzare i nuovi metodi per conservare e calcolare gli output per classe nel client Audio2Visual.

## Panoramica

Il client `clientA2V` ora supporta la conservazione di **tutti** gli output (non solo la media) per ogni classe, sia per le feature AST che per gli output degli adapter.

**IMPORTANTE**: Durante il training, gli output degli adapter vengono **automaticamente** conservati batch per batch e le medie vengono calcolate alla fine di ogni round di training.

## Struttura dei Dati

### 1. Feature AST per classe
```python
# self.per_class_outputs contiene TUTTI gli output AST per classe
# Struttura: {class_name: [list of tensors]}
# Esempio:
{
    'dog': [tensor1, tensor2, tensor3, ...],  # N campioni di 'dog'
    'cat': [tensor1, tensor2, ...],           # M campioni di 'cat'
    ...
}
```

### 2. Output degli Adapter per classe
```python
# Struttura: {adapter_name: {class_name: [list of tensors]}}
# Esempio:
{
    'clip_adapter': {
        'dog': [output1, output2, output3, ...],
        'cat': [output1, output2, ...],
    },
    't5_adapter': {
        'dog': [output1, output2, output3, ...],
        'cat': [output1, output2, ...],
    }
}
```

## Metodi Disponibili

### 1. `compute_class_mean_features()`
Calcola le feature AST e le conserva **tutte** per ogni classe (non solo la media).

```python
# Compute features from training data
ast_features = client.compute_class_mean_features(
    dataloader=None,      # None = usa il dataloader del nodo
    use_train=True,       # True = train data, False = test data
    device=device
)

# Output: {class_name: [list of all AST features]}
```

### 2. `compute_per_class_adapter_outputs()`
Passa le feature AST attraverso tutti gli adapter e conserva **tutti** gli output per ogni classe.

```python
# Compute adapter outputs for all classes
adapter_outputs = client.compute_per_class_adapter_outputs(
    dataloader=None,
    use_train=True,
    device=device
)

# Output: {adapter_name: {class_name: [list of all outputs]}}
```

### 3. `compute_adapter_mean_outputs()`
Calcola le medie degli output degli adapter a partire da tutti gli output conservati.

```python
# First, get all adapter outputs
adapter_outputs = client.compute_per_class_adapter_outputs(use_train=True, device=device)

# Then compute means
adapter_means = client.compute_adapter_mean_outputs(adapter_outputs)

# Output: {adapter_name: {class_name: mean_tensor}}
```

### 4. `update_per_class_mean_output()`
Metodo esistente, ora aggiornato per conservare tutti gli output in `self.per_class_outputs` e calcolare le medie in `self.per_class_outputs_mean`.

```python
# Automatically called at the end of each training round
client.update_per_class_mean_output(use_train=True, device=device)

# Conserva tutti gli output in: client.per_class_outputs
# Conserva le medie in: client.per_class_outputs_mean
```

## Esempio d'Uso Completo

### Metodo 1: Usare gli output raccolti durante il training (RACCOMANDATO)

Gli output degli adapter vengono raccolti **automaticamente** durante il training:

```python
# Durante il training
client.train(client_device=device)
# Gli output degli adapter vengono conservati automaticamente durante il training!

# Dopo il training, accedi agli output raccolti
# 1. Tutti gli output degli adapter raccolti durante training
all_training_outputs = client.get_training_adapter_outputs()
# Struttura: {adapter_name: {class_name: [list of outputs]}}

# 2. Medie degli adapter calcolate automaticamente
all_training_means = client.get_training_adapter_means()
# Struttura: {adapter_name: {class_name: mean_tensor}}

# 3. Accedi agli output per un adapter specifico
clip_outputs_all_classes = client.get_training_adapter_outputs('clip_adapter')
# Struttura: {class_name: [list of outputs]}

# 4. Accedi agli output per un adapter e classe specifici
dog_clip_outputs = client.get_training_adapter_outputs('clip_adapter', 'dog')
# Output: [tensor1, tensor2, tensor3, ...] - lista di tutti gli output per 'dog'

# 5. Accedi alla media per un adapter e classe specifici
dog_clip_mean = client.get_training_adapter_means('clip_adapter', 'dog')
# Output: tensor - media di tutti gli output per 'dog'

print(f"CLIP adapter outputs for 'dog': {len(dog_clip_outputs)} samples")
print(f"CLIP adapter mean for 'dog': {dog_clip_mean.shape}")
```

### Metodo 2: Calcolare gli output dopo il training

Puoi anche calcolare gli output dopo il training (utile se vuoi riprocessare):

```python
# Calcola gli output passando di nuovo attraverso gli adapter
adapter_outputs = client.compute_per_class_adapter_outputs(
    use_train=True,
    device=device
)

# Calcola le medie
adapter_means = client.compute_adapter_mean_outputs(adapter_outputs)

# Accedi agli output
dog_clip_outputs = adapter_outputs['clip_adapter']['dog']
dog_clip_mean = adapter_means['clip_adapter']['dog']
```

### Metodo 3: Lavorare con le feature AST

```python
# Durante il training, viene chiamato automaticamente update_per_class_mean_output()
# che popola:
# - client.per_class_outputs (tutti gli output AST)
# - client.per_class_outputs_mean (medie degli output AST)

# 1. Tutti gli output AST per classe
all_ast_outputs = client.per_class_outputs
print(f"Classi: {list(all_ast_outputs.keys())}")
for class_name, outputs_list in all_ast_outputs.items():
    print(f"  {class_name}: {len(outputs_list)} campioni")

# 2. Medie degli output AST per classe
ast_means = client.per_class_outputs_mean
for class_name, mean_output in ast_means.items():
    print(f"  {class_name}: {mean_output.shape}")
```

## Caso d'Uso: Generazione di Immagini

### Opzione 1: Usando le medie raccolte durante training (RACCOMANDATO)

```python
# Dopo il training, le medie sono già disponibili!
adapter_means = client.get_training_adapter_means()

# Per ogni classe, usa la media per generare un'immagine
for class_name in adapter_means['clip_adapter'].keys():
    # Prendi la media del clip_adapter per questa classe
    clip_mean = adapter_means['clip_adapter'][class_name]

    # Usa questa media per la generazione
    # (il codice esatto dipende dal tuo pipeline di generazione)
    generated_image = generate_image_from_embedding(clip_mean)

    print(f"Generated image for class: {class_name}")
```

### Opzione 2: Calcolare le medie dopo il training

```python
# 1. Calcola le medie degli adapter
adapter_outputs_all = client.compute_per_class_adapter_outputs(use_train=True, device=device)
adapter_means = client.compute_adapter_mean_outputs(adapter_outputs_all)

# 2. Per ogni classe, usa la media per generare un'immagine
for class_name in adapter_means['clip_adapter'].keys():
    # Prendi la media del clip_adapter per questa classe
    clip_mean = adapter_means['clip_adapter'][class_name]

    # Usa questa media per la generazione
    generated_image = generate_image_from_embedding(clip_mean)

    print(f"Generated image for class: {class_name}")
```

## Note Importanti

1. **Memoria**: Conservare tutti gli output richiede più memoria. Se hai molti campioni, considera di:
   - Calcolare le medie batch per batch invece di conservare tutto
   - Liberare memoria dopo aver calcolato le medie

2. **Device**: Tutti i tensori vengono conservati sul device specificato. Assicurati di avere memoria GPU sufficiente o sposta su CPU dopo il calcolo.

3. **Performance**: Il metodo `compute_per_class_adapter_outputs()` processa tutti i campioni attraverso gli adapter. Per dataset grandi, questo può richiedere tempo.

## Struttura Interna

```
clientA2V
├── per_class_outputs                    # dict: {class_name: [all AST features]}
├── per_class_outputs_mean               # dict: {class_name: mean AST feature}
├── training_adapter_outputs_all         # dict: {adapter_name: {class_name: [all outputs]}}
│                                        # Raccolti AUTOMATICAMENTE durante training!
├── training_adapter_outputs_mean        # dict: {adapter_name: {class_name: mean_tensor}}
│                                        # Calcolati AUTOMATICAMENTE a fine training!
└── Methods:
    ├── train_a2v()                              # AUTOMATICAMENTE raccoglie output durante training
    ├── get_training_adapter_outputs()           # Accedi agli output raccolti durante training
    ├── get_training_adapter_means()             # Accedi alle medie calcolate durante training
    ├── compute_class_mean_features()            # Calcola e conserva tutte le AST features
    ├── compute_per_class_adapter_outputs()      # Passa attraverso adapter, conserva tutto
    ├── compute_adapter_mean_outputs()           # Calcola medie dagli output adapter
    └── update_per_class_mean_output()           # Aggiorna per_class_outputs e medie
```

## Esempio Minimo

### Scenario 1: Durante/dopo il training (RACCOMANDATO)

```python
# Step 1: Esegui il training (gli output vengono raccolti automaticamente!)
client.train(client_device=device)

# Step 2: Accedi alle medie già calcolate
clip_means = client.get_training_adapter_means('clip_adapter')

# Step 3: Usa le medie
for class_name, mean_output in clip_means.items():
    print(f"{class_name}: {mean_output.shape}")
```

### Scenario 2: Calcolare medie dopo il training

```python
# Step 1: Calcola tutti gli output degli adapter
adapter_outputs = client.compute_per_class_adapter_outputs(use_train=True)

# Step 2: Calcola le medie
adapter_means = client.compute_adapter_mean_outputs(adapter_outputs)

# Step 3: Accedi alle medie CLIP
clip_means = adapter_means['clip_adapter']
for class_name, mean_output in clip_means.items():
    print(f"{class_name}: {mean_output.shape}")
```

## Riepilogo Rapido

```python
# DURANTE TRAINING (automatico):
# - Gli output degli adapter vengono conservati in training_adapter_outputs_all
# - Le medie vengono calcolate in training_adapter_outputs_mean

# DOPO TRAINING (accesso facile):
all_outputs = client.get_training_adapter_outputs()              # Tutti gli output
all_means = client.get_training_adapter_means()                  # Tutte le medie
clip_outputs = client.get_training_adapter_outputs('clip_adapter')  # Output CLIP
clip_dog = client.get_training_adapter_means('clip_adapter', 'dog')  # Media CLIP per 'dog'
```
