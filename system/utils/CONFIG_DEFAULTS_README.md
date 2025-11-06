# ConfigLoader Default Values Documentation

## Overview

Il `ConfigLoader` ora supporta valori di default automatici per tutti i parametri di configurazione, inclusi i parametri a livello di nodo della federazione.

## Funzionalità Principali

### 1. Valori di Default Automatici

Quando carichi un file di configurazione, tutti i parametri mancanti vengono automaticamente riempiti con valori di default sensati.

```python
from system.utils.config_loader import ConfigLoader

# Caricamento con defaults abilitati (comportamento di default)
loader = ConfigLoader('my_config.json')
config = loader.load_config()

# Caricamento senza defaults
loader = ConfigLoader('my_config.json', apply_defaults=False)
config = loader.load_config()
```

### 2. Defaults per Nodi della Federazione

I valori di default vengono applicati anche ai singoli nodi nella sezione `nodes` o `nodes_tasks`:

```json
{
  "nodes": {
    "0": {
      "dataset": "VEGAS"
    }
  }
}
```

Viene automaticamente espanso a:

```json
{
  "nodes": {
    "0": {
      "dataset": "VEGAS",
      "dataset_split": "0",
      "pretext_tasks": [],
      "task_type": "classification",
      "balance_classes": false
    }
  }
}
```

### 3. Defaults Disponibili per Nodi

I seguenti parametri hanno valori di default per i nodi:

- `dataset_split`: `"0"` - Split del dataset da utilizzare
- `pretext_tasks`: `[]` - Lista di task pretext
- `task_type`: `"classification"` - Tipo di task (classification, segmentation, etc.)
- `balance_classes`: `false` - Bilanciamento delle classi
- `selected_classes`: `null` - Classi specifiche da selezionare (opzionale)
- `excluded_classes`: `null` - Classi da escludere (opzionale)
- `class_labels`: `null` - Mapping personalizzato label→indice (opzionale)
- `class_remapping`: `null` - Rimappatura degli indici delle classi (opzionale)
- `limit_samples`: `null` - Limite numero di campioni (opzionale)

**Nota**: I parametri con valore `null` sono opzionali e vengono applicati solo se specificati nella configurazione.

## Metodi Utility

### Ottenere un Singolo Default

```python
# Ottieni il valore di default per un parametro specifico
lr = ConfigLoader.get_default('training', 'learning_rate', fallback=0.01)
# Ritorna: 0.005 (il valore di default)
```

### Ottenere Tutti i Defaults di una Sezione

```python
# Ottieni tutti i defaults per una sezione
training_defaults = ConfigLoader.get_section_defaults('training')
# Ritorna: {'optimizer': 'SGD', 'learning_rate': 0.005, ...}
```

### Ottenere i Defaults dei Nodi

```python
# Ottieni tutti i defaults per la configurazione dei nodi
node_defaults = ConfigLoader.get_node_defaults()
# Ritorna: {'dataset_split': '0', 'pretext_tasks': [], ...}
```

### Esportare Template di Configurazione

```python
# Esporta template base (senza esempi di nodi)
ConfigLoader.export_default_config('template.json')

# Esporta template con esempi di nodi (senza parametri opzionali)
ConfigLoader.export_default_config(
    'template_with_nodes.json',
    include_nodes_example=True,
    include_optional=False
)

# Esporta template completo (con tutti i parametri, anche opzionali)
ConfigLoader.export_default_config(
    'template_full.json',
    include_nodes_example=True,
    include_optional=True
)
```

## Sezioni con Defaults

Le seguenti sezioni supportano defaults automatici:

1. **experiment**: Parametri dell'esperimento (goal, device, seed, etc.)
2. **federation**: Parametri della federazione (algorithm, num_clients, global_rounds, etc.)
3. **model**: Parametri del modello (backbone, embedding_size, patch_size, etc.)
4. **training**: Parametri di training (optimizer, learning_rate, batch_size, etc.)
5. **dataset**: Parametri del dataset (name, partition, dir_alpha, etc.)
6. **rewind**: Parametri di rewind (epochs, ratio, strategy, etc.)
7. **fedgfe**: Parametri specifici FedGFE
8. **feda2v**: Parametri specifici FedA2V/Audio2Visual
9. **wandb**: Parametri Weights & Biases
10. **node**: Defaults per i singoli nodi della federazione

## Esempi Pratici

### Esempio 1: Configurazione Minimale

File `minimal_config.json`:
```json
{
  "experiment": {
    "goal": "my_experiment"
  },
  "nodes": {
    "0": {
      "dataset": "cifar10",
      "selected_classes": [0, 1, 2, 3, 4]
    }
  }
}
```

Quando viene caricato, tutti gli altri parametri vengono riempiti con defaults.

### Esempio 2: Configurazione Multi-Nodo

```json
{
  "experiment": {
    "goal": "federated_learning"
  },
  "federation": {
    "num_clients": 3
  },
  "nodes": {
    "0": {
      "dataset": "cifar10",
      "selected_classes": [0, 1, 2]
    },
    "1": {
      "dataset": "mnist",
      "task_type": "classification"
    },
    "2": {
      "dataset": "VEGAS",
      "pretext_tasks": ["rotation", "masking"]
    }
  }
}
```

Ogni nodo eredita i defaults appropriati per i parametri non specificati.

### Esempio 3: Override Selettivi

```json
{
  "training": {
    "learning_rate": 0.001
  },
  "nodes": {
    "0": {
      "dataset": "cifar10",
      "balance_classes": true
    }
  }
}
```

Solo i parametri specificati sovrascrivono i defaults, tutti gli altri usano i valori predefiniti.

## Note Importanti

1. **Precedenza CLI**: Gli argomenti da command-line hanno sempre precedenza sui valori JSON e sui defaults
2. **Logging**: Ogni default applicato viene stampato a console per trasparenza
3. **Backwards Compatibility**: Le configurazioni esistenti continuano a funzionare senza modifiche
4. **Parametri Opzionali**: Parametri con valore `null` nei defaults non vengono applicati a meno che non siano esplicitamente richiesti

## Script di Esempio

Esegui `example_config_defaults.py` per vedere tutti gli esempi in azione:

```bash
python example_config_defaults.py
```

## Accesso agli Attributi Opzionali

Il `DotDict` ora supporta l'accesso sicuro agli attributi opzionali. Quando si accede a un attributo che non esiste, viene restituito `None` invece di sollevare un'eccezione `AttributeError`.

```python
# Configurazione con nodo senza attributi opzionali
node_config = DotDict({'dataset': 'VEGAS', 'dataset_split': '0'})

# Accesso sicuro agli attributi opzionali
print(node_config.selected_classes)  # None (non solleva eccezione)
print(node_config.excluded_classes)  # None

# Pattern consigliati per controllare attributi opzionali
if node_config.selected_classes is not None:
    # Usa selected_classes
    pass

# O con 'in' operator
if 'selected_classes' in node_config:
    # L'attributo esiste
    pass

# O con get() method
classes = node_config.get('selected_classes', [])
```

Questo rende il codice più robusto e riduce la necessità di gestire eccezioni quando si lavora con configurazioni che possono avere attributi opzionali.

## Conclusioni

Il sistema di defaults rende le configurazioni più concise e facili da mantenere, permettendoti di specificare solo i parametri che differiscono dai valori standard. Insieme al supporto per attributi opzionali, fornisce un modo flessibile e sicuro per gestire configurazioni complesse.
