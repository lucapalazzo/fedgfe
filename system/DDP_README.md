# FedGFE with DistributedDataParallel (DDP) Support

Questa implementazione aggiunge supporto PyTorch DistributedDataParallel (DDP) al framework FedGFE per training distribuito multi-GPU in scenari di federated learning.

## Architettura

### Componenti Principali

1. **DDPMixin** (`flcore/clients/ddp_mixin.py`): Classe mixin che aggiunge funzionalità DDP
2. **clientGFEDDP** (`flcore/clients/clientgfe_ddp.py`): Client federated con supporto DDP
3. **FedGFEDDP** (`flcore/servers/servergfe_ddp.py`): Server federated con supporto DDP
4. **Utilities** (`utils/ddp_utils.py`): Funzioni di supporto per DDP

### Design Pattern

```python
# Uso del mixin pattern per mantenere compatibilità
class clientGFEDDP(DDPMixin, clientGFE):
    pass

# Server dedicato per gestire client DDP
class FedGFEDDP(FedGFE):
    pass
```

## Funzionalità Implementate

### ✅ Process Group Management
- Inizializzazione automatica tramite environment variables
- Setup di `world_size`, `rank`, `local_rank`
- Backend selection (`nccl` per GPU, `gloo` per CPU)

### ✅ Model Wrapping
- Wrapping automatico di `backbone`, `downstream_task`, `pretext_tasks` con DDP
- Gestione modelli VITFC multi-componente
- Parametro `find_unused_parameters=True` per federated learning

### ✅ Distributed DataLoaders
- `DistributedSampler` automatico per training e test data
- Gestione epoch shuffling tramite `set_epoch()`
- Fallback su DataLoader standard se DDP non disponibile

### ✅ Parameter Synchronization
- Sincronizzazione parametri tra processi DDP
- All-reduce con averaging per federated aggregation
- Barrier synchronization per consistency

### ✅ Enhanced Device Management
- Device assignment automatico basato su `local_rank`
- GPU memory cleanup ottimizzato
- Compatibilità con existing `_move_to_gpu()/_move_to_cpu()`

### ✅ Error Handling & Cleanup
- Context manager `DDPErrorHandler` per cleanup automatico
- Destructor cleanup per process groups
- Graceful degradation a single-process mode

## Setup e Utilizzo

### 1. GPU Selection

**Selezione Automatica:**
```bash
# Tool interattivo per selezione GPU
./scripts/gpu_select.py

# Selezione automatica
./scripts/gpu_select.py --processes 4 --auto
```

**Selezione Manuale:**
```bash
# Specifica GPU IDs
export CUDA_VISIBLE_DEVICES="0,2,3"

# O tramite parametri script
--gpu_ids "0,2,3"
```

**Formati GPU Selection:**
- `"0,2,3"` - GPU specifici 0, 2, 3
- `"0-3"` - Range GPU da 0 a 3
- `"all"` - Tutti i GPU disponibili

### 2. Environment Variables

**⚠️ Nota**: Se le environment variables DDP non sono impostate correttamente, il sistema automaticamente passa a **single process mode**.

**Per DDP multi-processo (tutte necessarie):**
```bash
export RANK=0                    # Global rank
export LOCAL_RANK=0              # Local rank su nodo  
export WORLD_SIZE=2              # Numero totale processi
export MASTER_ADDR=localhost     # Indirizzo master node
export MASTER_PORT=29500         # Porta comunicazione
export CUDA_VISIBLE_DEVICES="0,2,3"  # GPU da usare (opzionale)
```

**Per single process (default se environment DDP mancanti):**
```bash
export CUDA_VISIBLE_DEVICES="0,2,3"  # Solo per selezione GPU
# Nessuna altra variable necessaria
```

### 3. Script di Esempio

**Single Process:**
```bash
./scripts/run_ddp.sh single              # Tutti i GPU
./scripts/run_ddp.sh single "0,2"        # Solo GPU 0,2
```

**Multi-Process DDP:**
```bash
./scripts/run_ddp.sh multi 2             # 2 processi, tutti GPU
./scripts/run_ddp.sh multi 2 "0,1"       # 2 processi, GPU 0,1
./scripts/run_ddp.sh multi 4 "0-3"       # 4 processi, GPU 0,1,2,3
```

**Lista GPU Disponibili:**
```bash
./scripts/run_ddp.sh list-gpus
```

### 4. Uso Programmatico
```python
from flcore.servers.servergfe_ddp import FedGFEDDP
from utils.ddp_utils import parse_gpu_selection

# Con selezione GPU
args.gpu_ids = "0,2,3"
selected_gpus = parse_gpu_selection(args.gpu_ids)
server = FedGFEDDP(args, times=[])
```

## Integrazione nel Codice Esistente

### Modifica Minima
```python
# Prima (client normale)
from flcore.servers.servergfe import FedGFE
server = FedGFE(args, times=[])

# Dopo (client con supporto DDP)
from flcore.servers.servergfe_ddp import FedGFEDDP  
server = FedGFEDDP(args, times=[])
```

### Controllo DDP Status
```python
for client in server.clients:
    if hasattr(client, 'get_distributed_info'):
        info = client.get_distributed_info()
        print(f"Client {client.id}: DDP={info['is_distributed']}")
```

## Compatibilità

### ✅ Backward Compatibility
- Codice esistente funziona senza modifiche
- Fallback automatico a single-process se DDP non disponibile
- Stessi args e API del FedGFE originale

### ✅ Model Compatibility
- Supporto completo per modelli VITFC
- Gestione pretext tasks (patch_ordering, simclr, etc.)
- Downstream tasks (classification, segmentation)

### ✅ Training Compatibility
- SSL + Downstream training sequences
- Memory management esistente preservato
- Optimizer e scheduler management unchanged

## Performance Considerations

### Memory Usage
- DDP mantiene copie del model per processo
- GPU memory usage: `base_usage * num_processes_per_gpu`
- Cleanup automatico implementato

### Communication Overhead
- All-reduce operations per parameter sync
- Bandwidth usage proporzionale a model size
- Overlapping computation/communication in DDP

### Scalability
- Linear scaling teorico con numero GPUs
- Federated learning aggiunge communication rounds
- Ottimale per modelli large (ViT, etc.)

## Testing

### Unit Testing
```python
# Test DDP initialization
from utils.ddp_utils import verify_ddp_setup
assert verify_ddp_setup() == True

# Test client creation
from flcore.clients.clientgfe_ddp import clientGFEDDP
client = clientGFEDDP(args, ...)
assert client.is_distributed == True
```

### Integration Testing
```bash
# Test con esempio completo
python examples/ddp_example.py --num_clients 2 --global_rounds 2
```

## Troubleshooting

### Common Issues

1. **DDP Initialization Error (Missing MASTER_ADDR)**
   ```
   Error initializing torch.distributed: environment variable MASTER_ADDR expected
   ```
   **Soluzione**: 
   - Per single process: Nessuna action necessaria (fallback automatico)
   - Per multi-process: Usare `torchrun` o impostare tutte le variables DDP
   
2. **CUDA OOM Error**
   - Ridurre batch_size
   - Usare meno processi per GPU
   - Verificare memory cleanup

3. **Hanging Processes**
   - Verificare MASTER_ADDR/MASTER_PORT
   - Controllare firewall settings
   - Timeout in distributed operations

4. **Model Parameter Mismatch**  
   - Verificare model wrapping corretto
   - Controllare sync_model_parameters() calls
   - Debug parameter shapes

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose DDP info
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
```

## Future Enhancements

- [ ] Multi-node DDP support
- [ ] Dynamic client scaling
- [ ] Communication compression
- [ ] Fault tolerance mechanisms
- [ ] Performance profiling integration

## Esempi Pratici GPU Selection

### Server con 8 GPU
```bash
# Usa solo GPU 0,2,4,6 (evita conflitti)
./scripts/run_ddp.sh multi 4 "0,2,4,6"

# Usa GPU a coppie per memory sharing
./scripts/run_ddp.sh multi 4 "0,1,4,5"
```

### Server con GPU eterogenee  
```bash
# Lista GPU e memoria
./scripts/run_ddp.sh list-gpus

# Selezione automatica GPU ottimali
./scripts/gpu_select.py --processes 3 --auto --min-memory 6000
```

### Testing su GPU specifici
```bash
# Test su GPU singolo
./scripts/run_ddp.sh single "2"

# Test distributing su 2 GPU non adiacenti
./scripts/run_ddp.sh multi 2 "1,3"
```

## File Structure

```
system/
├── flcore/clients/
│   ├── ddp_mixin.py           # DDP mixin class
│   ├── clientgfe_ddp.py       # DDP-enabled client
├── flcore/servers/
│   └── servergfe_ddp.py       # DDP-enabled server
├── utils/
│   ├── ddp_utils.py           # DDP utilities
│   └── gpu_utils.py           # GPU selection utilities
├── examples/
│   └── ddp_example.py         # Complete example
├── scripts/
│   ├── run_ddp.sh             # Launch script
│   └── gpu_select.py          # Interactive GPU selection
└── DDP_README.md              # This file
```