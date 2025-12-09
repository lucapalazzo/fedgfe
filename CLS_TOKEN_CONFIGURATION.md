# CLS Token Only Configuration

## Overview

Questa funzionalità permette di utilizzare solo il **CLS token** (classification token) dall'output del modello AST (Audio Spectrogram Transformer) invece dell'intera sequenza di token.

## Motivazione

Il modello AST produce una sequenza di token come output, dove:
- Il **primo token** (indice 0) è il CLS token, che rappresenta una rappresentazione globale dell'intero input audio
- I **token successivi** rappresentano informazioni più locali/specifiche della sequenza

Utilizzare solo il CLS token può offrire diversi vantaggi:
1. **Riduzione della dimensionalità**: Invece di processare l'intera sequenza di token (tipicamente 1214 token per AST), si lavora con un singolo token
2. **Efficienza computazionale**: Minor carico di memoria e computazione negli adapters e nelle proiezioni
3. **Rappresentazione semantica globale**: Il CLS token è progettato per catturare informazioni semantiche di alto livello dell'intero audio
4. **Semplificazione dell'architettura**: Gli adapters possono lavorare con un input più compatto

## Implementazione

### Modifiche al Codice

#### 1. DownstreamSinestesiaAdapters

Il parametro `use_cls_token_only` è stato aggiunto al costruttore:

```python
def __init__(self,
             args,
             num_classes=10,
             wandb_log=False,
             device=None,
             use_classifier_loss=True,
             diffusion_type=None,
             loss_weights=None,
             torch_dtype=torch.float32,
             enable_diffusion=False,
             use_cls_token_only=False):  # Nuovo parametro
```

Nel metodo `forward()`, quando `use_cls_token_only=True`:
```python
if self.use_cls_token_only:
    x_cls = x[:, 0:1, :]  # Estrae solo il primo token (CLS)
    # Shape: (batch_size, 1, hidden_dim) invece di (batch_size, 1214, hidden_dim)
    x = x_cls
```

#### 2. clientA2V

Il client legge la configurazione e la passa al modello:

```python
# Get use_cls_token_only configuration from feda2v config
self.use_cls_token_only = self.feda2v_config.get('use_cls_token_only', False)

self.model = DownstreamSinestesiaAdapters(
    args=args,
    wandb_log=not args.no_wandb,
    device=self.device,
    use_classifier_loss=False,
    diffusion_type=self.global_model.diffusion_type,
    enable_diffusion=False,
    use_cls_token_only=self.use_cls_token_only  # Passa la configurazione
)
```

#### 3. serverA2V

Il server legge la configurazione e la passa al modello globale:

```python
# Get use_cls_token_only configuration from feda2v config
use_cls_token_only = getattr(self.config.feda2v, 'use_cls_token_only', False)

self.global_model = DownstreamSinestesiaAdapters(
    args,
    diffusion_type=self.diffusion_type,
    use_cls_token_only=use_cls_token_only
)
```

### Configurazione JSON

La funzionalità supporta **due livelli di configurazione** con priorità:

1. **Configurazione globale** (nella sezione `feda2v`)
2. **Configurazione per nodo** (nella sezione specifica di ogni nodo)

**Priorità**: Configurazione nodo > Configurazione globale > Default (false)

#### Opzione 1: Configurazione Globale

Per abilitare la funzionalità per tutti i nodi:

```json
{
  "feda2v": {
    "_comment_cls_token": "=== CLS TOKEN CONFIGURATION ===",
    "_comment_cls_token_note": "When enabled, only the CLS token (first token) from AST output is used instead of all sequence tokens",
    "use_cls_token_only": true,

    // ... altre configurazioni
  }
}
```

#### Opzione 2: Configurazione Per Nodo

Per configurare singolarmente ogni nodo (override della configurazione globale):

```json
{
  "feda2v": {
    "use_cls_token_only": false  // Configurazione globale (default)
  },
  "nodes": {
    "0": {
      "dataset": "ESC50",
      "use_cls_token_only": true,  // Node 0 usa solo CLS token
      // ... altre configurazioni
    },
    "1": {
      "dataset": "ESC50",
      "use_cls_token_only": false,  // Node 1 usa tutti i token
      // ... altre configurazioni
    },
    "2": {
      "dataset": "ESC50",
      // Node 2 usa il valore globale (false)
      // ... altre configurazioni
    }
  }
}
```

#### Opzione 3: Configurazione Mista

Combinazione di configurazione globale con override specifici:

```json
{
  "feda2v": {
    "use_cls_token_only": true  // Tutti i nodi usano CLS token di default
  },
  "nodes": {
    "0": {
      "dataset": "ESC50",
      // Node 0 usa il valore globale (true)
    },
    "1": {
      "dataset": "ESC50",
      "use_cls_token_only": false,  // Node 1 fa override: usa tutti i token
    }
  }
}
```

## File di Esempio

File di configurazione disponibili:
- `configs/a2v_cls_token_only_example.json` - Configurazione globale per tutti i nodi
- `configs/a2v_cls_token_mixed_nodes.json` - Configurazione mista con nodi diversi

## Casi d'Uso per Configurazione Per Nodo

La possibilità di configurare `use_cls_token_only` per singolo nodo è utile in diversi scenari:

### 1. Esperimenti Comparativi
Confrontare le performance tra nodi che usano CLS token e nodi che usano l'intera sequenza:
```json
{
  "nodes": {
    "0": { "use_cls_token_only": true },   // Gruppo sperimentale
    "1": { "use_cls_token_only": true },
    "2": { "use_cls_token_only": false },  // Gruppo di controllo
    "3": { "use_cls_token_only": false }
  }
}
```

### 2. Ottimizzazione della Memoria Eterogenea
Nodi con risorse limitate usano CLS token, altri usano l'intera sequenza:
```json
{
  "nodes": {
    "0": { "use_cls_token_only": true },   // Nodo con GPU limitata
    "1": { "use_cls_token_only": false },  // Nodo con GPU potente
    "2": { "use_cls_token_only": true },   // Nodo con GPU limitata
  }
}
```

### 3. Specializzazione per Tipo di Dato
Diversi tipi di dati potrebbero beneficiare di diverse rappresentazioni:
```json
{
  "nodes": {
    "0": {
      "selected_classes": ["dog", "cat", "sheep"],  // Suoni animali
      "use_cls_token_only": true  // Rappresentazione globale sufficiente
    },
    "1": {
      "selected_classes": ["keyboard_typing", "mouse_click"],  // Suoni complessi
      "use_cls_token_only": false  // Serve informazione dettagliata
    }
  }
}
```

## Comportamento

### Con `use_cls_token_only=false` (default)

```
AST Output: (batch_size, 1214, 768)
    ↓
Adapters e Projections processano tutti i 1214 token
    ↓
Output finale: dipende dalla configurazione degli adapters
```

### Con `use_cls_token_only=true`

```
AST Output: (batch_size, 1214, 768)
    ↓
Estrazione CLS Token: (batch_size, 1, 768)
    ↓
Adapters e Projections processano solo 1 token
    ↓
Output finale: dipende dalla configurazione degli adapters
```

## Compatibilità

- ✅ Compatibile con `diffusion_type='sd'`
- ✅ Compatibile con `diffusion_type='flux'`
- ✅ Compatibile con tutti i tipi di aggregazione
- ✅ Compatibile con modalità generator e non-generator
- ✅ Backwards compatible: il comportamento default (senza specificare il parametro) rimane invariato

## Logging

Il sistema fornisce diversi livelli di logging per tracciare la configurazione:

### 1. Inizializzazione del Nodo

All'avvio di ogni client, viene registrato quale configurazione viene utilizzata:

```
INFO: Node 0: Using node-specific use_cls_token_only=True
INFO: Node 1: Using node-specific use_cls_token_only=False
INFO: Node 2: Using global use_cls_token_only=True
```

Questo permette di verificare immediatamente:
- Se il nodo usa una configurazione specifica o quella globale
- Quale valore è stato applicato

### 2. Forward Pass

Durante ogni forward pass, quando `use_cls_token_only=true`, viene registrato:

```
INFO: Using CLS token only. Original shape: torch.Size([8, 1214, 768]), CLS shape: torch.Size([8, 1, 768])
```

Questo mostra:
- La shape originale dell'output AST
- La shape dopo l'estrazione del CLS token
- Conferma che la funzionalità è attiva

### 3. Verifica della Configurazione

Per verificare la configurazione durante l'esecuzione, monitora i log per:
- Messaggi di inizializzazione dei nodi
- Shape dei tensor nei forward pass
- Eventuali warning o errori relativi alla configurazione

### Esempio di Output Completo

```
INFO: Node 0: Using node-specific use_cls_token_only=True
INFO: Node 1: Using node-specific use_cls_token_only=False
INFO: Node 2: Using global use_cls_token_only=False
...
INFO: Using CLS token only. Original shape: torch.Size([8, 1214, 768]), CLS shape: torch.Size([8, 1, 768])
```

In questo esempio:
- Node 0 usa CLS token (configurazione specifica)
- Node 1 usa tutti i token (configurazione specifica)
- Node 2 usa tutti i token (configurazione globale)
- Il messaggio del forward pass conferma l'estrazione del CLS token per Node 0

## Note Tecniche

1. **Dimensione mantenuta**: Anche estraendo solo il CLS token, manteniamo la dimensione della sequenza come `(batch_size, 1, hidden_dim)` invece di `(batch_size, hidden_dim)` per mantenere la compatibilità con gli adapters esistenti.

2. **Detach**: L'output AST viene automaticamente detached dal grafo computazionale prima di essere passato agli adapters (comportamento esistente).

3. **Performance**: L'uso del CLS token può ridurre significativamente il tempo di forward pass e l'utilizzo di memoria, specialmente con batch size grandi.

## Test

Per testare la funzionalità:

```bash
# Usa la configurazione di esempio
python main.py --config configs/a2v_cls_token_only_example.json

# Oppure modifica una configurazione esistente aggiungendo:
# "use_cls_token_only": true nella sezione feda2v
```

## Riferimenti

- AST Model: [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- CLS Token concept: Derivato da architetture transformer come BERT e ViT
- File modificati:
  - `system/flcore/trainmodel/downstreamsinestesiaadapters.py`
  - `system/flcore/clients/clientA2V.py`
  - `system/flcore/servers/serverA2V.py`
