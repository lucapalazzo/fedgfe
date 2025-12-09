# Node-Specific Generator Configuration

## Overview

Questa feature permette a ogni nodo di personalizzare completamente la configurazione del generator:
1. **Override granularity**: Ogni nodo può scegliere la propria granularità (`unified`, `per_class`, `per_group`)
2. **Subset di classi**: Ogni nodo può specificare un sottoinsieme delle sue classi per il training del generator
3. **Gruppo specifico**: Per modalità `per_group`, ogni nodo può specificare quale gruppo usare

## Configurazione

### 1. Configurazione Globale

Nel config JSON, sotto `feda2v`, definisci i default globali:

```json
"feda2v": {
  "generator_granularity": "per_group",  // Default granularity (può essere overriden per nodo)
  "generator_class_groups": {
    "animals": ["dog", "rooster", "pig", "cow", "frog"],
    "nature": ["rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds"],
    "mechanical": ["helicopter", "chainsaw", "siren", "car_horn", "engine"]
  }
}
```

### 2. Configurazione Per-Nodo

Ogni nodo può personalizzare la propria configurazione con questi campi opzionali:

#### 2.1 Override Granularity (`generator_granularity`)

Ogni nodo può scegliere la propria granularità indipendentemente dal default globale:

```json
"nodes": {
  "0": {
    "selected_classes": ["dog", "rooster", "pig"],
    "generator_granularity": "per_class"  // Override: usa per_class invece del default
  },
  "1": {
    "selected_classes": ["rain", "sea_waves"],
    "generator_granularity": "unified"    // Override: usa unified invece del default
  }
}
```

#### 2.2 Subset di Classi (`generator_train_classes`)

Ogni nodo può specificare un sottoinsieme delle sue `selected_classes` per il training del generator:

```json
"nodes": {
  "0": {
    "selected_classes": ["dog", "rooster", "pig", "cow", "frog"],  // 5 classi totali
    "generator_train_classes": ["dog", "rooster", "pig"]           // Generator allena solo su 3
  }
}
```

**IMPORTANTE**: `generator_train_classes` DEVE essere un sottoinsieme di `selected_classes`, altrimenti le classi invalide vengono ignorate.

#### 2.3 Gruppo Specifico (`generator_class_group`)

Per modalità `per_group`, specifica quale gruppo usare:

```json
"nodes": {
  "0": {
    "selected_classes": ["dog", "rooster", "pig", "cow", "frog"],
    "generator_class_group": "animals"
  }
}
```

## Comportamento

### Priorità delle Configurazioni

Il sistema segue questa priorità per determinare la configurazione del generator:

1. **Granularity**: `node.generator_granularity` > `feda2v.generator_granularity` (global default)
2. **Classi training**: `node.generator_train_classes` > `node.selected_classes` (tutte le classi)
3. **Gruppo**: `node.generator_class_group` > auto-detect da `feda2v.generator_class_groups`

### Esempi di Comportamento

#### Esempio 1: Override Granularity + Subset Classi

```json
"feda2v": {
  "generator_granularity": "per_group"  // Default globale
},
"nodes": {
  "0": {
    "selected_classes": ["dog", "rooster", "pig", "cow", "frog"],
    "generator_granularity": "per_class",        // Override → usa per_class
    "generator_train_classes": ["dog", "pig"]    // Solo 2 classi
  }
}
```

**Risultato**:
- Vengono creati 2 generators: uno per "dog" e uno per "pig"
- Le classi "rooster", "cow", "frog" non hanno generator

#### Esempio 2: Per-Group con Gruppo Specifico

```json
"nodes": {
  "0": {
    "selected_classes": ["dog", "rooster", "pig", "cow", "frog"],
    "generator_class_group": "animals"
  }
}
```

**Risultato**:
- Un solo generator chiamato "animals"
- Allena su tutte e 5 le classi
- Checkpoint: `vae_group_animals_node0.pth`

#### Esempio 3: Auto-Detect Gruppi

```json
"nodes": {
  "0": {
    "selected_classes": ["dog", "rain", "helicopter"]
    // Nessun override
  }
}
```

**Risultato** (con granularity=per_group):
- 3 generators creati automaticamente:
  - "animals" (per "dog")
  - "nature" (per "rain")
  - "mechanical" (per "helicopter")

## Casi d'uso

### 1. Un gruppo per nodo (consigliato per efficienza)

Ogni nodo allena un generator su un gruppo semantico specifico:

```json
"nodes": {
  "0": {"selected_classes": ["dog", "cow"], "generator_class_group": "animals"},
  "1": {"selected_classes": ["rain", "sea_waves"], "generator_class_group": "nature"},
  "2": {"selected_classes": ["helicopter", "chainsaw"], "generator_class_group": "mechanical"}
}
```

**Vantaggi**:
- Un solo generator per nodo (efficiente)
- Specializzazione semantica chiara
- Nome checkpoint descrittivo (es: `vae_group_animals_node0.pth`)

### 2. Per-class su subset (massima flessibilità)

Nodo che vuole generators separati solo per alcune classi specifiche:

```json
"nodes": {
  "0": {
    "selected_classes": ["dog", "rooster", "pig", "cow", "frog"],
    "generator_granularity": "per_class",
    "generator_train_classes": ["dog", "rooster"]
  }
}
```

**Risultato**: Solo "dog" e "rooster" hanno generator dedicati

### 3. Mix di strategie tra nodi

Ogni nodo usa una strategia diversa:

```json
"nodes": {
  "0": {
    "selected_classes": ["dog", "rooster", "pig"],
    "generator_granularity": "per_class"  // 3 generators separati
  },
  "1": {
    "selected_classes": ["rain", "sea_waves", "crackling_fire"],
    "generator_granularity": "unified"    // 1 generator unificato
  },
  "2": {
    "selected_classes": ["helicopter", "chainsaw", "siren"],
    "generator_class_group": "mechanical" // 1 generator per gruppo
  }
}
```

### 4. Training selettivo su classi specifiche

Nodo che ha accesso a molte classi ma vuole allenare il generator solo su alcune:

```json
"nodes": {
  "0": {
    "selected_classes": ["dog", "rooster", "pig", "cow", "frog", "rain", "sea_waves"],
    "generator_train_classes": ["dog", "cow", "frog"]  // Solo animali domestici
  }
}
```

**Uso**: Utile quando alcune classi hanno pochi esempi o qualità bassa

## Esempi Completi

### File di configurazione:
- [a2v_generator_training_per_group.json](a2v_generator_training_per_group.json) - Esempio base con gruppi
- [a2v_node_custom_generators.json](a2v_node_custom_generators.json) - Esempio avanzato con tutte le features

## Note Implementative

### File modificati:
- [system/flcore/clients/clientA2V.py](../system/flcore/clients/clientA2V.py)
  - Linea ~174-192: Lettura configurazioni per-nodo
    - `generator_granularity`: Override granularity
    - `generator_train_classes`: Subset classi con validazione
    - `generator_class_group`: Gruppo specifico
  - Linea ~1580-1585: Uso di `generator_train_classes` invece di tutte le classi
  - Linea ~1587-1622: Logica mapping aggiornata per supportare subset

### Parametri per-nodo disponibili:

| Parametro | Tipo | Descrizione | Default |
|-----------|------|-------------|---------|
| `generator_granularity` | string | Override granularity: `unified`, `per_class`, `per_group` | Da config globale |
| `generator_train_classes` | list | Subset di `selected_classes` per training | Tutte le `selected_classes` |
| `generator_class_group` | string | Nome gruppo per modalità `per_group` | Auto-detect da `generator_class_groups` |

### Validazione:

1. **generator_train_classes**: Il sistema verifica automaticamente che le classi siano un sottoinsieme valido di `selected_classes`
   - Classi invalide vengono rimosse con warning
   - Se tutte le classi sono invalide, il nodo non allena generators

2. **generator_granularity**: Valori accettati: `unified`, `per_class`, `per_group`
   - Altri valori vengono ignorati e si usa il default globale

### Checkpoint naming:

I checkpoint seguono questi pattern:
- **Unified**: `{base_name}_unified_node{id}.pth`
- **Per-class**: `{base_name}_class_{class_name}_node{id}.pth`
- **Per-group**: `{base_name}_group_{group_name}_node{id}.pth`

Esempio: `vae_custom_class_dog_node0.pth`
