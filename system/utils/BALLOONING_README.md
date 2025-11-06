# GPU Memory Ballooning System

Sistema per allocare e riservare memoria GPU per prevenire errori OOM (Out Of Memory) causati da altri processi.

## Problema

Quando si eseguono processi multipli su GPU condivise, può succedere che:
1. Un processo alloca memoria gradualmente
2. Altri processi tentano di allocare memoria contemporaneamente
3. Si verifica un errore OOM anche se c'è memoria sufficiente inizialmente

Il memory ballooning risolve questo problema **pre-allocando** memoria all'inizio per riservare lo spazio necessario.

## Come Funziona

Il sistema alloca tensori "dummy" (palloncini/balloons) sulla GPU che:
- **Inflazione (inflate)**: Allocano memoria riservandola per il tuo processo
- **Deflazione (deflate)**: Rilasciano la memoria quando non più necessaria

## Formati di Specifica Memoria Supportati

Il sistema supporta formati flessibili per specificare quanta memoria allocare:

- **MB (int)**: `2000` = 2000 MB
- **GB (str)**: `"2GB"`, `"2.5GB"` = 2 o 2.5 GB
- **MB (str)**: `"2048MB"` = 2048 MB
- **Percentuale (str)**: `"50%"`, `"75%"` = percentuale della memoria libera
- **Frazione (float)**: `0.5`, `0.7` = frazione della memoria libera (0.0-1.0)

Esempio:
```python
balloon.allocate_memory("2GB")      # 2 GB
balloon.allocate_memory(2048)       # 2048 MB
balloon.allocate_memory("50%")      # 50% della memoria libera
balloon.allocate_memory(0.7)        # 70% della memoria libera
```

## Utilizzo Base

### Singola GPU

```python
from system.utils.ballooning import GPUMemoryBalloon

# Metodo 1: Specifica la quantità in MB
balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=4000)  # Riserva 4GB
balloon.inflate()

# ... esegui il tuo codice ...

balloon.deflate()  # Rilascia la memoria

# Metodo 2: Specifica una frazione della memoria libera
balloon = GPUMemoryBalloon(gpu_id=0, reserve_fraction=0.8)  # Riserva 80% della memoria libera
balloon.inflate()
# ... codice ...
balloon.deflate()
```

### Context Manager (Consigliato)

```python
from system.utils.ballooning import GPUMemoryBalloon

# Pulizia automatica
with GPUMemoryBalloon(gpu_id=0, reserve_mb=4000):
    # Memoria riservata qui
    model = MyModel().cuda()
    train(model)
# Memoria automaticamente rilasciata all'uscita
```

### Multiple GPU

```python
from system.utils.ballooning import MultiGPUMemoryBalloon

# Stessa quantità su tutte le GPU
multi_balloon = MultiGPUMemoryBalloon(
    gpu_ids=[0, 1, 2],
    reserve_mb_per_gpu=2000  # 2GB per GPU
)
multi_balloon.inflate_all()

# ... codice multi-GPU ...

multi_balloon.deflate_all()

# Oppure quantità diverse per GPU
multi_balloon = MultiGPUMemoryBalloon(
    gpu_ids=[0, 1, 2],
    reserve_mb_per_gpu=[4000, 2000, 2000]  # 4GB su GPU 0, 2GB su GPU 1 e 2
)
```

### Context Manager Multi-GPU

```python
from system.utils.ballooning import reserve_multi_gpu_memory

with reserve_multi_gpu_memory(gpu_ids=[0, 1], reserve_mb_per_gpu=2000):
    # 2GB riservati su GPU 0 e 1
    # ... training distribuito ...
# Memoria rilasciata automaticamente
```

## Parametri

### `GPUMemoryBalloon`

- **`gpu_id`**: ID della GPU (0, 1, 2, ...)
- **`reserve_mb`**: Quantità di memoria in MB da riservare (es: 4000 = 4GB)
- **`reserve_fraction`**: Frazione della memoria libera da riservare (0.0-1.0)
  - Es: 0.5 = 50%, 0.8 = 80%
- **`chunk_size_mb`**: Dimensione di ogni chunk di allocazione (default: 100MB)
  - Chunks più piccoli = allocazione più sicura ma più lenta
  - Chunks più grandi = allocazione più veloce ma rischio maggiore di OOM
  - Raccomandato: 100-500 MB

**Note**: Specificare **esattamente uno** tra `reserve_mb` e `reserve_fraction`.

### `MultiGPUMemoryBalloon`

- **`gpu_ids`**: Lista di ID GPU
- **`reserve_mb_per_gpu`**: MB da riservare per GPU
  - `int`: stessa quantità su tutte
  - `List[int]`: quantità specifica per ogni GPU
- **`reserve_fraction_per_gpu`**: Frazione da riservare per GPU
  - `float`: stessa frazione su tutte
  - `List[float]`: frazione specifica per ogni GPU

## Metodi Principali

### Metodi Base

#### `inflate(verbose=True)`
Alloca memoria (gonfia il palloncino).
- **Returns**: Quantità di memoria allocata in MB
- **Raises**: `RuntimeError` se già inflated o allocazione fallisce

#### `deflate(verbose=True)`
Dealloca memoria (sgonfia il palloncino).
- **Returns**: Quantità di memoria rilasciata in MB

#### `get_status()`
Ottieni informazioni sullo stato corrente.
- **Returns**: Dictionary con:
  - `gpu_id`: ID della GPU
  - `is_inflated`: Se il balloon è attivo
  - `allocated_mb`: Memoria allocata in MB
  - `num_chunks`: Numero di chunks allocati
  - `free_memory_mb`: Memoria libera sulla GPU
  - `total_memory_mb`: Memoria totale sulla GPU

```python
balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=2000)
balloon.inflate()
status = balloon.get_status()
print(status)
# {'gpu_id': 0, 'is_inflated': True, 'allocated_mb': 2000, ...}
```

### Metodi Condizionali (Nuovi!)

#### `inflate_if_not_inflated(verbose=True)`
Alloca memoria solo se non già allocata. Utile per chiamate ripetute sicure.
- **Returns**: MB allocati (0 se già inflated)

```python
balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)
balloon.inflate_if_not_inflated()  # Alloca 1000 MB
balloon.inflate_if_not_inflated()  # Non fa nulla (già allocato)
```

#### `deflate_if_inflated(verbose=True)`
Dealloca memoria solo se attualmente allocata.
- **Returns**: MB rilasciati (0 se non inflated)

```python
balloon.deflate_if_inflated()  # Dealloca se necessario
balloon.deflate_if_inflated()  # Non fa nulla (già deallocato)
```

### Metodi di Controllo Avanzati

#### `resize_balloon(new_reserve_mb=None, new_reserve_fraction=None, verbose=True)`
Ridimensiona il balloon. Dealloca e rialloca con nuova dimensione.
- **Returns**: MB allocati dopo resize

```python
balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)
balloon.inflate()
# Cambia dimensione a 2GB
balloon.resize_balloon(new_reserve_mb=2000)
# Oppure usa frazione
balloon.resize_balloon(new_reserve_fraction=0.5)
```

#### `set_chunk_size(chunk_size_mb)`
Imposta la dimensione dei chunk per future allocazioni.

```python
balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=2000)
balloon.set_chunk_size(200)  # Usa chunk da 200 MB
balloon.inflate()  # Userà chunk da 200 MB
```

### Metodi di Allocazione Flessibile (Nuovi!)

#### `allocate_memory(amount, verbose=True)`
Alloca memoria con specifica flessibile. Alternative user-friendly a `inflate()`.

**Supporta tutti i formati**: MB, GB, percentuale, frazione.

```python
balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)

# Vari formati supportati
balloon.allocate_memory("2GB")      # Alloca 2 GB
balloon.allocate_memory(2048)       # Alloca 2048 MB
balloon.allocate_memory("50%")      # Alloca 50% della memoria libera
balloon.allocate_memory(0.7)        # Alloca 70% della memoria libera
```

#### `allocate_additional(amount, verbose=True)`
Alloca memoria **aggiuntiva** senza deallocare quella esistente.

**Allocazione incrementale** - aggiungi memoria sopra quella già allocata.

```python
balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)
balloon.allocate_memory("2GB")         # Alloca 2GB
print(balloon.get_status()['allocated_mb'])  # 2048 MB

balloon.allocate_additional("1GB")     # Aggiunge 1GB
print(balloon.get_status()['allocated_mb'])  # 3072 MB (totale)

balloon.allocate_additional("50%")     # Aggiunge 50% della memoria libera residua
print(balloon.get_status()['allocated_mb'])  # Ancora più memoria!
```

#### `allocate_to_target(target, verbose=True)`
Alloca o dealloca per raggiungere un **target specifico**.

Aumenta o diminuisce automaticamente la memoria allocata.

```python
balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)
balloon.allocate_memory("2GB")
print(balloon.get_status()['allocated_mb'])  # 2048 MB

# Aumenta a 5GB
balloon.allocate_to_target("5GB")
print(balloon.get_status()['allocated_mb'])  # 5120 MB

# Diminuisce a 3GB
balloon.allocate_to_target("3GB")
print(balloon.get_status()['allocated_mb'])  # 3072 MB

# Usa percentuale
balloon.allocate_to_target("80%")  # Alloca fino all'80% della memoria libera
```

## Funzioni Helper per Tutte le GPU (Nuove!)

### `get_all_available_gpus()`
Restituisce lista di tutte le GPU disponibili nel sistema.

```python
from system.utils.ballooning import get_all_available_gpus

gpus = get_all_available_gpus()
print(f"GPUs disponibili: {gpus}")  # [0, 1, 2, 3]
```

### `get_cuda_visible_devices()`
Restituisce lista di GPU visibili al processo corrente (rispetta `CUDA_VISIBLE_DEVICES`).

```python
import os
from system.utils.ballooning import get_cuda_visible_devices

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
gpus = get_cuda_visible_devices()
print(f"GPUs visibili: {gpus}")  # [0, 2]
```

### `create_balloon_for_all_gpus(...)`
Crea automaticamente un MultiGPUMemoryBalloon per tutte le GPU disponibili.

```python
from system.utils.ballooning import create_balloon_for_all_gpus

# Riserva 2GB su tutte le GPU
balloon = create_balloon_for_all_gpus(reserve_mb_per_gpu=2000)
balloon.inflate_all()

# Oppure frazione
balloon = create_balloon_for_all_gpus(reserve_fraction_per_gpu=0.6)
balloon.inflate_all()
```

### `reserve_all_gpus_memory(...)` (Context Manager)
Context manager che riserva memoria su tutte le GPU disponibili.

```python
from system.utils.ballooning import reserve_all_gpus_memory

# Riserva 2GB su tutte le GPU automaticamente
with reserve_all_gpus_memory(reserve_mb_per_gpu=2000):
    # Memoria riservata su TUTTE le GPU
    # ... training ...
# Memoria rilasciata automaticamente su tutte le GPU

# Oppure con frazione
with reserve_all_gpus_memory(reserve_fraction_per_gpu=0.5):
    # Riserva 50% su ogni GPU
    pass
```

## Esempi Pratici

### Server Federato con Memory Protection

```python
from system.utils.ballooning import GPUMemoryBalloon

class ServerGFE:
    def __init__(self, args):
        self.args = args
        self.gpu_id = args.device_id

        # Riserva memoria per prevenire OOM
        self.memory_balloon = GPUMemoryBalloon(
            gpu_id=self.gpu_id,
            reserve_mb=3000  # Riserva 3GB
        )
        self.memory_balloon.inflate()

        self.model = self.load_model()

    def train(self):
        # Training con memoria protetta
        pass

    def cleanup(self):
        # Rilascia memoria
        self.memory_balloon.deflate()
```

### Client con Protezione Automatica

```python
from system.utils.ballooning import reserve_gpu_memory

class ClientGFE:
    def local_training(self):
        # Proteggi memoria durante training
        with reserve_gpu_memory(gpu_id=self.gpu_id, reserve_fraction=0.7):
            # 70% della memoria libera è riservata
            for epoch in range(self.epochs):
                self.train_epoch()
        # Memoria rilasciata automaticamente
```

### Multi-GPU Training con DDP

```python
from system.utils.ballooning import MultiGPUMemoryBalloon
import torch.distributed as dist

def train_ddp(rank, world_size):
    # Setup DDP
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Riserva memoria su tutte le GPU
    balloon = MultiGPUMemoryBalloon(
        gpu_ids=list(range(world_size)),
        reserve_fraction_per_gpu=0.6  # 60% su ogni GPU
    )
    balloon.inflate_all()

    # Training
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    train(model)

    # Cleanup
    balloon.deflate_all()
    dist.destroy_process_group()
```

### Allocazione Dinamica Basata su Carico

```python
from system.utils.ballooning import GPUMemoryBalloon
from system.utils.gpu_utils import get_gpu_info

def smart_reserve(gpu_id):
    """Riserva memoria in base al carico attuale della GPU."""
    gpu_info = get_gpu_info()
    gpu = gpu_info[gpu_id]

    if gpu['utilization'] < 20:
        # GPU quasi libera, riserva molto
        fraction = 0.9
    elif gpu['utilization'] < 50:
        # GPU moderatamente usata
        fraction = 0.6
    else:
        # GPU molto usata, riserva poco
        fraction = 0.3

    balloon = GPUMemoryBalloon(gpu_id=gpu_id, reserve_fraction=fraction)
    balloon.inflate()
    return balloon
```

### Controllo Chunk Size per Grandi Allocazioni

```python
from system.utils.ballooning import GPUMemoryBalloon

# Allocazione grande con chunk grandi (più veloce)
balloon = GPUMemoryBalloon(
    gpu_id=0,
    reserve_mb=10000,  # 10 GB
    chunk_size_mb=500   # Chunk da 500 MB
)
balloon.inflate()  # Più veloce

# Allocazione con chunk piccoli (più sicuro)
balloon_safe = GPUMemoryBalloon(
    gpu_id=0,
    reserve_mb=10000,
    chunk_size_mb=50   # Chunk da 50 MB
)
balloon_safe.inflate()  # Più lento ma più sicuro
```

### Gestione Automatica di Tutte le GPU

```python
from system.utils.ballooning import reserve_all_gpus_memory

# Caso d'uso: Training distribuito su tutte le GPU disponibili
with reserve_all_gpus_memory(reserve_fraction_per_gpu=0.7):
    # Memoria riservata automaticamente su TUTTE le GPU
    # Non serve sapere quante GPU ci sono!

    import torch.distributed as dist
    dist.init_process_group("nccl")

    # ... training distribuito ...

# Memoria rilasciata automaticamente
```

### Allocazione Dinamica con Formati Flessibili

```python
from system.utils.ballooning import GPUMemoryBalloon

# Inizia con allocazione base
balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)
balloon.allocate_memory("2GB")  # User-friendly!

# Durante il training, serve più memoria
balloon.allocate_additional("1GB")  # Aggiunge 1GB senza deallocare

# Verifica memoria
status = balloon.get_status()
print(f"Memoria allocata: {status['allocated_mb']} MB")
print(f"Memoria libera: {status['free_memory_mb']} MB")

# Se serve ancora più memoria, alloca fino a target
balloon.allocate_to_target("5GB")  # Alloca fino a 5GB totali

# Alla fine, rilascia tutto
balloon.deflate()
```

### Allocazione Adattiva Basata su Fase di Training

```python
from system.utils.ballooning import GPUMemoryBalloon

balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)

# Fase 1: Pre-training - serve molta memoria
balloon.allocate_memory("80%")  # Usa 80% della memoria disponibile
print(f"Pre-training: {balloon.get_status()['allocated_mb']} MB")

# ... pre-training ...

# Fase 2: Fine-tuning - serve meno memoria
balloon.allocate_to_target("50%")  # Riduce al 50%
print(f"Fine-tuning: {balloon.get_status()['allocated_mb']} MB")

# ... fine-tuning ...

balloon.deflate()
```

## Gestione Errori

```python
from system.utils.ballooning import GPUMemoryBalloon

try:
    balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=50000)  # Troppa memoria
    balloon.inflate()
except RuntimeError as e:
    print(f"Failed to allocate: {e}")
    # Prova con meno memoria
    balloon = GPUMemoryBalloon(gpu_id=0, reserve_fraction=0.5)
    balloon.inflate()
```

## Riepilogo Metodi di Allocazione

| Metodo | Quando Usarlo | Esempio |
|--------|---------------|---------|
| `inflate()` | Allocazione base con config iniziale | `balloon.inflate()` |
| `allocate_memory()` | Specifica flessibile (GB, %, ecc.) | `balloon.allocate_memory("2GB")` |
| `allocate_additional()` | Aggiungere memoria incrementalmente | `balloon.allocate_additional("1GB")` |
| `allocate_to_target()` | Raggiungere una quantità target | `balloon.allocate_to_target("5GB")` |
| `resize_balloon()` | Cambiare completamente dimensione | `balloon.resize_balloon(new_reserve_mb=3000)` |
| `deflate()` | Rilasciare tutta la memoria | `balloon.deflate()` |

## Best Practices

1. **Usa formati leggibili** per specificare la memoria
   ```python
   # ✅ Facile da leggere
   balloon.allocate_memory("2GB")
   balloon.allocate_memory("50%")

   # ❌ Meno chiaro
   balloon.allocate_memory(2048)
   ```

2. **Usa Context Managers** quando possibile per cleanup automatico
   ```python
   with GPUMemoryBalloon(gpu_id=0, reserve_mb=2000):
       # codice
   # cleanup automatico
   ```

3. **Controlla lo stato** prima di operazioni critiche
   ```python
   status = balloon.get_status()
   if status['free_memory_mb'] < 1000:
       print("Warning: Low memory!")
   ```

4. **Usa allocazione incrementale** invece di riallocare tutto
   ```python
   # ✅ Efficiente
   balloon.allocate_memory("2GB")
   balloon.allocate_additional("1GB")  # Aggiunge senza deallocare

   # ❌ Inefficiente
   balloon.allocate_memory("2GB")
   balloon.deflate()
   balloon.allocate_memory("3GB")  # Rialloca tutto
   ```

5. **Usa `reserve_fraction` o percentuali** per codice portabile
   ```python
   # ✅ Funziona su GPU di dimensioni diverse
   balloon.allocate_memory("70%")
   balloon = GPUMemoryBalloon(gpu_id=0, reserve_fraction=0.7)
   ```

6. **Rilascia sempre la memoria** quando finito
   ```python
   try:
       balloon.inflate()
       # ... lavoro ...
   finally:
       balloon.deflate()  # Sempre eseguito
   ```

7. **Per multi-GPU, usa helper automatici**
   ```python
   # ✅ Semplice - gestisce tutte le GPU automaticamente
   with reserve_all_gpus_memory(reserve_mb_per_gpu="2GB"):
       pass

   # ❌ Manuale - devi specificare ogni GPU
   multi_balloon = MultiGPUMemoryBalloon(gpu_ids=[0, 1, 2, 3], ...)
   ```

## Logging

Il modulo usa Python logging. Configura il livello per vedere i dettagli:

```python
import logging

logging.basicConfig(level=logging.INFO)  # Vedi allocazioni
logging.basicConfig(level=logging.DEBUG) # Vedi tutto
logging.basicConfig(level=logging.WARNING) # Solo warnings/errors
```

## Testing

Esegui gli esempi integrati:

```bash
cd /home/lpala/fedgfe
python -m system.utils.ballooning
```

Questo eseguirà vari esempi e mostrerà l'output.

## Limitazioni

- Richiede PyTorch con CUDA
- La memoria è effettivamente allocata (non solo "riservata virtualmente")
- L'allocazione può fallire se non c'è abbastanza memoria libera
- Il chunk size influenza la velocità e la sicurezza dell'allocazione

## Troubleshooting

**Problema**: `RuntimeError: CUDA out of memory` durante `inflate()`
- **Soluzione**: Riduci `reserve_mb` o `reserve_fraction`

**Problema**: Altri processi ancora causano OOM
- **Soluzione**: Aumenta la memoria riservata o esegui `inflate()` prima

**Problema**: Allocazione molto lenta
- **Soluzione**: Aumenta `chunk_size_mb` (default: 100)

**Problema**: Balloon già inflated
- **Soluzione**: Chiama `deflate()` prima di ri-inflare
