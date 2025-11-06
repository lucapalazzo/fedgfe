# nodes_tasks Configuration System

## Panoramica

Il sistema `nodes_tasks` permette di configurare compiti specifici per ogni nodo nel federated learning. La configurazione viene preservata in **due formati** in `args`:

### 🔹 Formato Originale (Nuovo)
- **`args.nodes_tasks`** - Dizionario Python con la configurazione originale per nodo

### 🔹 Formato CLI (Esistente)
- **`args.nodes_datasets`** - String con dataset convertiti
- **`args.nodes_pretext_tasks`** - String con pretext task aggregati
- **`args.nodes_downstream_tasks`** - String con downstream task aggregati
- **`args.num_clients`** - Numero di nodi aggiornato automaticamente

## Utilizzo

### 📝 Via CLI (Comando)

```bash
# JSON string diretto
python main.py --nodes_tasks_config '{"0": {"task_type": "classification", "pretext_tasks": ["image_rotation"]}, "1": {"task_type": "segmentation", "pretext_tasks": ["patch_masking"]}}'

# File JSON
python main.py --nodes_tasks_config configs/my_nodes.json
```

### 📝 Via File di Configurazione JSON

```json
{
  "federation": {"algorithm": "FedGFE"},
  "nodes_tasks": {
    "0": {
      "task_type": "classification",
      "pretext_tasks": ["image_rotation", "byod"],
      "dataset": "JSRT-1C-ClaSSeg",
      "dataset_split": "0"
    },
    "1": {
      "task_type": "segmentation",
      "pretext_tasks": ["patch_masking"],
      "dataset": "JSRT-1C-ClaSSeg",
      "dataset_split": "1"
    }
  }
}
```

```bash
python main.py --config configs/experiment.json
```

## Struttura di `args.nodes_tasks`

```python
args.nodes_tasks = {
    "0": {
        "task_type": "classification",          # Tipo di task downstream
        "pretext_tasks": ["image_rotation"],    # Lista pretext task per questo nodo
        "dataset": "JSRT-1C-ClaSSeg",          # Dataset per questo nodo
        "dataset_split": "0"                    # Split specifico del dataset
    },
    "1": {
        "task_type": "segmentation",
        "pretext_tasks": ["patch_masking", "byod"],
        "dataset": "JSRT-1C-ClaSSeg",
        "dataset_split": "1"
    }
}
```

## Conversione Automatica

Il sistema converte automaticamente da `nodes_tasks` al formato CLI esistente:

### Input `nodes_tasks`:
```json
{
  "0": {"task_type": "classification", "pretext_tasks": ["image_rotation"]},
  "1": {"task_type": "segmentation", "pretext_tasks": ["patch_masking"]}
}
```

### Output in `args`:
```python
args.nodes_tasks = {...}  # Configurazione originale preservata

# Formato CLI generato automaticamente:
args.nodes_downstream_tasks = "classification,segmentation"
args.nodes_pretext_tasks = "image_rotation,patch_masking"
args.num_clients = 2
```

## Priorità CLI vs JSON

**CLI ha sempre priorità su JSON:**

```bash
# JSON ha nodes_tasks con 1 nodo, ma CLI override con 2 nodi
python main.py --config base.json --nodes_tasks_config '{"0": {...}, "1": {...}}'
# Risultato: CLI vince, 2 nodi utilizzati
```

## Validazione

Il sistema include validazione automatica:

### ✅ Task Types Validi
- `classification`
- `segmentation`
- `regression`

### ✅ Pretext Tasks Validi
- `image_rotation`
- `patch_masking`
- `byod`
- `simclr`

### ❌ Errori Comuni
```python
# Errore: task_type non valido
{"0": {"task_type": "invalid_task"}}

# Errore: pretext_tasks deve essere lista
{"0": {"pretext_tasks": "not_a_list"}}

# Errore: JSON syntax non valido
--nodes_tasks_config '{invalid: json}'
```

## Esempi Pratici

### Scenario 1: Multi-Task Heterogeneous
```python
args.nodes_tasks = {
    "0": {"task_type": "classification", "pretext_tasks": ["image_rotation"]},
    "1": {"task_type": "segmentation", "pretext_tasks": ["patch_masking"]},
    "2": {"task_type": "regression", "pretext_tasks": ["byod"]}
}
```

### Scenario 2: Same Task, Different Pretext
```python
args.nodes_tasks = {
    "0": {"task_type": "classification", "pretext_tasks": ["image_rotation"]},
    "1": {"task_type": "classification", "pretext_tasks": ["simclr"]},
    "2": {"task_type": "classification", "pretext_tasks": ["byod"]}
}
```

### Scenario 3: Multi-Dataset Setup
```python
args.nodes_tasks = {
    "0": {"task_type": "classification", "dataset": "JSRT-1C-ClaSSeg", "dataset_split": "0"},
    "1": {"task_type": "classification", "dataset": "CIFAR10-4C", "dataset_split": "0"},
    "2": {"task_type": "segmentation", "dataset": "JSRT-1C-ClaSSeg", "dataset_split": "1"}
}
```

## Accesso nel Codice

### In `servergfe.py` - Creazione Client
```python
def set_clients(self, clientObj):
    # Controllo se abbiamo configurazione nodes_tasks
    if hasattr(self.args, 'nodes_tasks') and self.args.nodes_tasks is not None:
        print("Using nodes_tasks configuration for client creation")
        self._set_clients_from_nodes_tasks(clientObj)
    else:
        print("Using legacy dataset configuration for client creation")
        self._set_clients_legacy(clientObj)

def _set_clients_from_nodes_tasks(self, clientObj):
    # Ordinamento per consistenza
    sorted_node_ids = sorted(self.args.nodes_tasks.keys(), key=lambda x: int(x))

    for node_id in sorted_node_ids:
        node_config = self.args.nodes_tasks[node_id]
        client_id = int(node_id)

        # Configurazione specifica per nodo
        task_type = node_config.get('task_type', 'classification')
        pretext_tasks = node_config.get('pretext_tasks', [])
        dataset = node_config.get('dataset', self.args.dataset)
        dataset_split = int(node_config.get('dataset_split', client_id))

        # Creazione client con configurazione specifica
        client = self.create_clients(clientObj, client_id, dataset, dataset_split)
        client.pretext_tasks = pretext_tasks  # Specifico per nodo
        client.downstream_task_name = task_type
```

### In altre funzioni
```python
def your_function(args):
    # Accesso alla configurazione originale
    if hasattr(args, 'nodes_tasks') and args.nodes_tasks:
        for node_id, node_config in args.nodes_tasks.items():
            task_type = node_config['task_type']
            pretext_tasks = node_config.get('pretext_tasks', [])
            dataset = node_config.get('dataset', None)

            print(f"Node {node_id}: {task_type} with {pretext_tasks}")

    # Accesso al formato CLI (compatibilità)
    downstream_tasks = args.nodes_downstream_tasks.split(',')
    pretext_tasks = args.nodes_pretext_tasks.split(',')
    num_clients = args.num_clients
```

## Esempi di Comando Completi

### Esempio 1: Via JSON Configuration
```bash
# Usando il file di configurazione completo
python main.py --config configs/nodes_tasks_example.json

# Override specifici parametri
python main.py --config configs/nodes_tasks_example.json --global_rounds 100 --local_learning_rate 0.002
```

### Esempio 2: Via CLI nodes_tasks_config
```bash
# JSON string inline
python main.py --nodes_tasks_config '{"0": {"task_type": "classification", "pretext_tasks": ["image_rotation"]}, "1": {"task_type": "segmentation", "pretext_tasks": ["patch_masking"]}}'

# File JSON dedicato
python main.py --nodes_tasks_config configs/my_nodes.json --algorithm FedGFE --global_rounds 50
```

### Esempio 3: Multi-Dataset Heterogeneous
```bash
python main.py --nodes_tasks_config '{
  "0": {"task_type": "classification", "pretext_tasks": ["image_rotation"], "dataset": "JSRT-1C-ClaSSeg", "dataset_split": "0"},
  "1": {"task_type": "segmentation", "pretext_tasks": ["patch_masking"], "dataset": "JSRT-1C-ClaSSeg", "dataset_split": "1"},
  "2": {"task_type": "classification", "pretext_tasks": ["byod"], "dataset": "CIFAR10-4C", "dataset_split": "0"}
}'
```

## Output Esempio

Quando si usa `nodes_tasks`, l'output mostrerà:

```
Using nodes_tasks configuration for client creation
Creating 3 clients from nodes_tasks configuration:
  Node 0: classification on JSRT-1C-ClaSSeg:0 with pretext_tasks ['image_rotation', 'byod']
  Node 1: segmentation on JSRT-1C-ClaSSeg:1 with pretext_tasks ['patch_masking']
  Node 2: classification on JSRT-2C-ClaSSeg:0 with pretext_tasks ['simclr']
Successfully created 3 clients from nodes_tasks configuration

*** Client 0 dataset JSRT-1C-ClaSSeg id 0
*** Client 1 dataset JSRT-1C-ClaSSeg id 1
*** Client 2 dataset JSRT-2C-ClaSSeg id 0
```

## Compatibilità

- ✅ **Backward Compatible**: Il formato CLI esistente continua a funzionare
- ✅ **Forward Compatible**: `args.nodes_tasks` fornisce accesso completo ai dettagli
- ✅ **Dual Access**: Entrambi i formati disponibili simultaneamente
- ✅ **Graceful Fallback**: Errori gestiti senza crash del sistema
- ✅ **Server Integration**: `servergfe.py` usa automaticamente `nodes_tasks` quando disponibile