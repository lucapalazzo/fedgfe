# VEGAS Dataset - Node Splitting per Federated Learning

Questa documentazione descrive come utilizzare la funzionalità di splitting per nodi nel dataset VEGAS, che permette di dividere il dataset in parti riproducibili per scenari di Federated Learning.

## Caratteristiche Principali

- **Split bilanciati per classe**: I campioni sono distribuiti mantenendo la proporzione delle classi
- **Riproducibilità**: Utilizzando lo stesso seed, gli split sono sempre identici
- **Due modalità di splitting**:
  1. **num_nodes**: Divide tutti i campioni equamente tra N nodi
  2. **samples_per_node**: Ogni nodo riceve esattamente K campioni per classe

## Modalità 1: Divisione Equa tra N Nodi (`num_nodes`)

### Descrizione
Divide automaticamente tutti i campioni disponibili in N parti uguali, mantenendo il bilanciamento delle classi.

### Parametri
- `node_split_id`: ID del nodo (0-indexed, deve essere < num_nodes)
- `num_nodes`: Numero totale di nodi
- `node_split_seed`: Seed per la riproducibilità (default: 42)

### Esempio Python

```python
from datautils.dataset_vegas import VEGASDataset

# Configurazione per 3 nodi
num_nodes = 3
seed = 42
classes = ['dog', 'baby_cry', 'chainsaw']

# Crea dataset per il nodo 0
node0_dataset = VEGASDataset(
    root_dir="/path/to/VEGAS",
    selected_classes=classes,
    split='train',
    node_id=0,
    num_nodes=num_nodes,
    node_split_seed=seed
)

# Crea dataset per il nodo 1
node1_dataset = VEGASDataset(
    root_dir="/path/to/VEGAS",
    selected_classes=classes,
    split='train',
    node_id=1,
    num_nodes=num_nodes,
    node_split_seed=seed
)

# Crea dataset per il nodo 2
node2_dataset = VEGASDataset(
    root_dir="/path/to/VEGAS",
    selected_classes=classes,
    split='train',
    node_id=2,
    num_nodes=num_nodes,
    node_split_seed=seed
)

print(f"Nodo 0: {len(node0_dataset)} campioni")
print(f"Nodo 1: {len(node1_dataset)} campioni")
print(f"Nodo 2: {len(node2_dataset)} campioni")
```

### Esempio JSON Config

```json
{
  "federation": {
    "num_clients": 3
  },
  "nodes": {
    "0": {
      "dataset": "vegas",
      "node_split_id": 0,
      "num_nodes": 3,
      "node_split_seed": 42,
      "selected_classes": ["dog", "baby_cry", "chainsaw"]
    },
    "1": {
      "dataset": "vegas",
      "node_split_id": 1,
      "num_nodes": 3,
      "node_split_seed": 42,
      "selected_classes": ["dog", "baby_cry", "chainsaw"]
    },
    "2": {
      "dataset": "vegas",
      "node_split_id": 2,
      "num_nodes": 3,
      "node_split_seed": 42,
      "selected_classes": ["dog", "baby_cry", "chainsaw"]
    }
  }
}
```

### Come Funziona

Se una classe ha 100 campioni e `num_nodes=3`:
- Nodo 0: riceve campioni 0-34 (35 campioni)
- Nodo 1: riceve campioni 35-67 (33 campioni)
- Nodo 2: riceve campioni 68-99 (32 campioni)

I campioni rimanenti (100 % 3 = 1) vengono distribuiti ai primi nodi.

## Modalità 2: Numero Fisso di Campioni (`samples_per_node`)

### Descrizione
Ogni nodo riceve esattamente K campioni per ogni classe. Utile quando si vuole controllare la dimensione del dataset per nodo.

### Parametri
- `node_split_id`: ID del nodo (0-indexed)
- `samples_per_node`: Numero di campioni per classe che ogni nodo deve ricevere
- `node_split_seed`: Seed per la riproducibilità (default: 42)

### Esempio Python

```python
from datautils.dataset_vegas import VEGASDataset

# Configurazione: ogni nodo riceve 20 campioni per classe
samples_per_node = 20
seed = 42
classes = ['dog', 'baby_cry', 'chainsaw', 'drum']

# Crea dataset per il nodo 0 (primi 20 campioni per classe)
node0_dataset = VEGASDataset(
    root_dir="/path/to/VEGAS",
    selected_classes=classes,
    split='train',
    node_id=0,
    samples_per_node=samples_per_node,
    node_split_seed=seed
)

# Crea dataset per il nodo 1 (successivi 20 campioni per classe)
node1_dataset = VEGASDataset(
    root_dir="/path/to/VEGAS",
    selected_classes=classes,
    split='train',
    node_id=1,
    samples_per_node=samples_per_node,
    node_split_seed=seed
)

# Ogni nodo avrà esattamente 20 * 4 = 80 campioni
print(f"Nodo 0: {len(node0_dataset)} campioni")
print(f"Nodo 1: {len(node1_dataset)} campioni")
```

### Esempio JSON Config

```json
{
  "federation": {
    "num_clients": 5
  },
  "nodes": {
    "0": {
      "dataset": "vegas",
      "node_split_id": 0,
      "samples_per_node": 20,
      "node_split_seed": 42,
      "selected_classes": ["dog", "baby_cry", "chainsaw"]
    },
    "1": {
      "dataset": "vegas",
      "node_split_id": 1,
      "samples_per_node": 20,
      "node_split_seed": 42,
      "selected_classes": ["dog", "baby_cry", "chainsaw"]
    }
  }
}
```

### Come Funziona

Con `samples_per_node=20`:
- Nodo 0: riceve campioni 0-19 per ogni classe (20 campioni/classe)
- Nodo 1: riceve campioni 20-39 per ogni classe (20 campioni/classe)
- Nodo 2: riceve campioni 40-59 per ogni classe (20 campioni/classe)
- etc.

## Scenari di Utilizzo Avanzati

### Scenario 1: FL Omogeneo con Classi Condivise

Tutti i nodi hanno le stesse classi ma dati diversi:

```json
{
  "nodes": {
    "0": {
      "node_split_id": 0,
      "num_nodes": 5,
      "selected_classes": ["dog", "baby_cry", "chainsaw", "drum", "fireworks"]
    },
    "1": {
      "node_split_id": 1,
      "num_nodes": 5,
      "selected_classes": ["dog", "baby_cry", "chainsaw", "drum", "fireworks"]
    },
    "2": {
      "node_split_id": 2,
      "num_nodes": 5,
      "selected_classes": ["dog", "baby_cry", "chainsaw", "drum", "fireworks"]
    }
  }
}
```

### Scenario 2: FL Eterogeneo con Classi Diverse

Ogni nodo ha classi completamente diverse:

```json
{
  "nodes": {
    "0": {
      "node_split_id": 0,
      "samples_per_node": 30,
      "selected_classes": ["dog", "baby_cry"]
    },
    "1": {
      "node_split_id": 0,
      "samples_per_node": 30,
      "selected_classes": ["chainsaw", "drum"]
    },
    "2": {
      "node_split_id": 0,
      "samples_per_node": 30,
      "selected_classes": ["fireworks", "helicopter"]
    }
  }
}
```

### Scenario 3: FL Semi-Eterogeneo con Overlapping

Alcuni nodi condividono alcune classi:

```json
{
  "nodes": {
    "0": {
      "node_split_id": 0,
      "num_nodes": 2,
      "selected_classes": ["dog", "baby_cry", "chainsaw"]
    },
    "1": {
      "node_split_id": 1,
      "num_nodes": 2,
      "selected_classes": ["dog", "baby_cry", "chainsaw"]
    },
    "2": {
      "node_split_id": 0,
      "num_nodes": 2,
      "selected_classes": ["chainsaw", "drum", "fireworks"]
    },
    "3": {
      "node_split_id": 1,
      "num_nodes": 2,
      "selected_classes": ["chainsaw", "drum", "fireworks"]
    }
  }
}
```

## Riproducibilità

### Garantire Split Identici

Per garantire che gli split siano sempre identici tra diverse esecuzioni:

1. **Usa lo stesso `node_split_seed`** per tutti i nodi
2. **Mantieni l'ordine delle classi** in `selected_classes`
3. **Usa gli stessi parametri di split** (`num_nodes` o `samples_per_node`)

```python
# Questi due dataset saranno identici
dataset1 = VEGASDataset(
    selected_classes=['dog', 'baby_cry'],
    node_id=0,
    num_nodes=3,
    node_split_seed=42
)

dataset2 = VEGASDataset(
    selected_classes=['dog', 'baby_cry'],
    node_id=0,
    num_nodes=3,
    node_split_seed=42
)

assert len(dataset1) == len(dataset2)
```

### Cambiare Seed per Esperimenti Diversi

```python
# Esperimento 1
exp1_node0 = VEGASDataset(node_id=0, num_nodes=3, node_split_seed=42)
exp1_node1 = VEGASDataset(node_id=1, num_nodes=3, node_split_seed=42)

# Esperimento 2 con split diversi
exp2_node0 = VEGASDataset(node_id=0, num_nodes=3, node_split_seed=123)
exp2_node1 = VEGASDataset(node_id=1, num_nodes=3, node_split_seed=123)
```

## Integrazione con Train/Val/Test Split

Il node splitting avviene **dopo** il train/val/test split:

```python
# Prima viene applicato il train/val/test split
# Poi viene applicato il node split

# Nodo 0 - Training set
train_node0 = VEGASDataset(
    split='train',
    node_id=0,
    num_nodes=3
)

# Nodo 0 - Validation set
val_node0 = VEGASDataset(
    split='val',
    node_id=0,
    num_nodes=3
)

# Nodo 0 - Test set (solitamente condiviso tra tutti i nodi)
test_node0 = VEGASDataset(
    split='test',
    node_id=0,
    num_nodes=3
)
```

### Uso con Auto-Split (split=None)

```python
# Crea automaticamente train/val/test con node splitting
dataset = VEGASDataset(
    split=None,  # Auto-crea train/val/test
    node_id=0,
    num_nodes=5,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2
)

# Accedi agli split
train_loader = DataLoader(dataset.train, batch_size=32)
val_loader = DataLoader(dataset.val, batch_size=32)
test_loader = DataLoader(dataset.test, batch_size=32)
```

## Validazione e Debugging

### Verificare la Distribuzione dei Campioni

```python
def print_node_distribution(dataset):
    """Stampa la distribuzione delle classi per un nodo."""
    samples_per_class = dataset.get_samples_per_class()
    distribution = dataset.get_class_distribution()

    print(f"\nNodo {dataset.node_id}:")
    print(f"Totale campioni: {len(dataset)}")
    print("\nDistribuzione per classe:")
    for class_name in sorted(samples_per_class.keys()):
        count = samples_per_class[class_name]
        percent = distribution[class_name]
        print(f"  {class_name}: {count} campioni ({percent:.2f}%)")

# Esempio
node0 = VEGASDataset(node_id=0, num_nodes=3, selected_classes=['dog', 'baby_cry'])
node1 = VEGASDataset(node_id=1, num_nodes=3, selected_classes=['dog', 'baby_cry'])
node2 = VEGASDataset(node_id=2, num_nodes=3, selected_classes=['dog', 'baby_cry'])

print_node_distribution(node0)
print_node_distribution(node1)
print_node_distribution(node2)
```

### Verificare che Non Ci Siano Sovrapposizioni

```python
def check_no_overlap(datasets):
    """Verifica che non ci siano campioni duplicati tra i nodi."""
    all_file_ids = []

    for i, dataset in enumerate(datasets):
        file_ids = set([sample['file_id'] for sample in dataset.samples])

        # Controlla sovrapposizioni
        overlap = set(all_file_ids).intersection(file_ids)
        if overlap:
            print(f"ERRORE: Nodo {i} ha {len(overlap)} campioni sovrapposti!")
            return False

        all_file_ids.extend(file_ids)

    print(f"OK: Nessuna sovrapposizione tra {len(datasets)} nodi")
    print(f"Totale campioni unici: {len(all_file_ids)}")
    return True

# Esempio
nodes = [
    VEGASDataset(node_id=0, num_nodes=3, selected_classes=['dog']),
    VEGASDataset(node_id=1, num_nodes=3, selected_classes=['dog']),
    VEGASDataset(node_id=2, num_nodes=3, selected_classes=['dog'])
]

check_no_overlap(nodes)
```

## Best Practices

1. **Usa sempre lo stesso seed** per tutti i nodi in un esperimento
2. **Documenta i parametri di split** nel nome dell'esperimento
3. **Verifica la distribuzione** prima di avviare training lunghi
4. **Per test globali**, usa lo stesso test set per tutti i nodi (node_id=0 o senza node splitting)
5. **Salva le configurazioni** in file JSON per riproducibilità

## Limitazioni e Note

- **Mutua esclusività**: Non è possibile usare `num_nodes` e `samples_per_node` contemporaneamente
- **Campioni insufficienti**: Con `samples_per_node`, se una classe ha meno campioni del richiesto, verrà mostrato un warning
- **Ordine delle classi**: L'ordine in `selected_classes` influenza l'ordinamento interno ma non la distribuzione
- **Cache**: Il caching considera i parametri di node splitting, quindi split diversi avranno cache separate

## Esempi Completi

Vedi i file di configurazione di esempio:
- `configs/vegas_node_splitting_example.json` - Esempio con num_nodes
- `configs/vegas_samples_per_node_example.json` - Esempio con samples_per_node

## Supporto

Per problemi o domande sulla funzionalità di node splitting, consulta:
- Codice sorgente: `system/datautils/dataset_vegas.py`
- Config loader: `system/utils/config_loader.py`
- Test: `tests/test_vegas_node_splitting.py`
