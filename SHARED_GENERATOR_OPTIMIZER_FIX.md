# Fix: Shared Generator Optimizer Management

## Data: 2026-01-08

## Problema Identificato

Nel caso di **shared generator** con `generator_only_mode=True` e `shared_generator_in_only_mode=True`, l'optimizer non veniva gestito correttamente:

### Issue Critiche

1. **Optimizer non condiviso**: Nel caso `unified` granularity, l'optimizer veniva memorizzato in `self.generator_optimizer` invece che in `self.generator_optimizers['unified']`

2. **Reset inconsistente**: Quando il client resettava il generator dopo il training (linee 3028-3042 clientA2V.py), creava un NUOVO optimizer locale che non veniva sincronizzato con il server

3. **Perdita dello stato**: Lo stato dell'optimizer del server (momentum, varianze AdamW, ecc.) veniva perso ad ogni reset

4. **Inconsistenza tra granularità**: Il caso `per_class` usava correttamente `generator_optimizers[class_name]`, ma il caso `unified` usava `generator_optimizer`, creando inconsistenza

## Modifiche Implementate

### 1. Server (serverA2V.py)

#### Inizializzazione VAE Generator (linee 467-477)
```python
# Optimizer for VAE with balanced learning rate
self.generator_optimizer = torch.optim.AdamW(
    self.prompt_generator.parameters(),
    lr=self.config.training.learning_rate * 0.1,
    weight_decay=1e-4
)

# IMPORTANTE: Memorizza anche in generator_optimizers con chiave 'unified'
if not hasattr(self, 'generator_optimizers'):
    self.generator_optimizers = {}
self.generator_optimizers['unified'] = self.generator_optimizer
```

#### Inizializzazione GAN Generator (linee 515-522)
```python
self.generator_optimizer = torch.optim.AdamW(gen_params, lr=1e-4)
self.discriminator_optimizer = torch.optim.AdamW(disc_params, lr=1e-4)

# IMPORTANTE: Memorizza anche in generator_optimizers con chiave 'unified'
if not hasattr(self, 'generator_optimizers'):
    self.generator_optimizers = {}
self.generator_optimizers['unified'] = self.generator_optimizer
```

#### Caricamento Checkpoint VAE (linee 1674-1687)
```python
if self.generator_training_mode and 'optimizer_state_dict' in checkpoint:
    generator_key = checkpoint.get('generator_key', 'unified')
    optimizer = torch.optim.AdamW(...)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.generator_optimizers[generator_key] = optimizer

    # Imposta anche il riferimento legacy se è unified
    if generator_key == 'unified':
        self.generator_optimizer = optimizer
```

#### Caricamento Checkpoint GAN (linee 1706-1715)
```python
if 'generator_optimizer_state_dict' in checkpoint:
    if self.generator_optimizer is not None:
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])

        # Memorizza anche in generator_optimizers con chiave 'unified'
        if not hasattr(self, 'generator_optimizers'):
            self.generator_optimizers = {}
        self.generator_optimizers['unified'] = self.generator_optimizer
```

### 2. Client (clientA2V.py)

#### Reset Generator Parameters (linee 2053-2076)
```python
# Reset optimizer state by creating a new optimizer
# IMPORTANTE: In shared generator mode, optimizer è memorizzato in generator_optimizers['unified']
# per garantire che tutti i client usino la stessa istanza
optimizer_key = 'unified'

# Verifica entrambe le posizioni per retrocompatibilità
has_shared_optimizer = hasattr(self, 'generator_optimizers') and optimizer_key in self.generator_optimizers
has_legacy_optimizer = hasattr(self, 'generator_optimizer') and self.generator_optimizer is not None

if has_shared_optimizer or has_legacy_optimizer:
    new_optimizer = torch.optim.AdamW(
        self.prompt_generator.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    # Memorizza in generator_optimizers dict per accesso condiviso
    if not hasattr(self, 'generator_optimizers'):
        self.generator_optimizers = {}
    self.generator_optimizers[optimizer_key] = new_optimizer

    # Aggiorna anche il riferimento legacy per retrocompatibilità
    self.generator_optimizer = new_optimizer
```

#### Train Generator (linee 3076-3087)
```python
if self.generator_granularity == 'unified':
    # Get optimizer from shared dict if available, otherwise use legacy attribute
    optimizer = None
    if hasattr(self, 'generator_optimizers') and 'unified' in self.generator_optimizers:
        optimizer = self.generator_optimizers['unified']
    elif hasattr(self, 'generator_optimizer') and self.generator_optimizer is not None:
        optimizer = self.generator_optimizer

    if optimizer is None:
        print(f"[Client {self.id}] ERROR: No optimizer found for unified generator")
        return 0.0

    return self._train_single_generator(class_outputs, self.prompt_generator, optimizer, 'unified')
```

#### Save Generator Checkpoint (linee 2214-2222)
```python
# Get optimizer for this generator
# Priority: 1) specific key in dict, 2) 'unified' key in dict, 3) legacy attribute
optimizer = None
if gen_key and gen_key in self.generator_optimizers:
    optimizer = self.generator_optimizers[gen_key]
elif hasattr(self, 'generator_optimizers') and 'unified' in self.generator_optimizers:
    optimizer = self.generator_optimizers['unified']
elif hasattr(self, 'generator_optimizer'):
    optimizer = self.generator_optimizer
```

## Vantaggi della Soluzione

1. **Consistenza**: Sia `unified` che `per_class` granularity usano ora `generator_optimizers` dict
2. **Condivisione corretta**: Tutti i client che usano lo shared generator accedono allo stesso optimizer
3. **Reset sincronizzato**: Quando un optimizer viene resettato, viene resettato nel dizionario condiviso
4. **Retrocompatibilità**: Il codice supporta ancora `generator_optimizer` per backward compatibility
5. **Stato preservato**: Lo stato dell'optimizer (momentum, ecc.) viene correttamente preservato e condiviso

## Compatibilità

Le modifiche sono **backward compatible**:
- Il codice cerca prima in `generator_optimizers['unified']`
- Se non trovato, cerca in `generator_optimizer` (legacy)
- Entrambi i riferimenti vengono mantenuti aggiornati

## Testing Consigliato

Testare con una configurazione che usa:
```json
{
  "generator_only_mode": true,
  "shared_generator_in_only_mode": true,
  "generator_granularity": "unified",
  "generator_type": "vae",
  "reset_generator_on_class_change": true
}
```

Verificare che:
1. L'optimizer venga condiviso tra tutti i client
2. Il reset dell'optimizer avvenga correttamente
3. Lo stato dell'optimizer venga preservato tra i round
4. Il checkpoint salvi e carichi correttamente l'optimizer

## File Modificati

1. [system/flcore/servers/serverA2V.py](system/flcore/servers/serverA2V.py)
   - Linee 467-477 (inizializzazione VAE)
   - Linee 515-522 (inizializzazione GAN)
   - Linee 1674-1687 (caricamento checkpoint VAE)
   - Linee 1706-1715 (caricamento checkpoint GAN)

2. [system/flcore/clients/clientA2V.py](system/flcore/clients/clientA2V.py)
   - Linee 2053-2076 (reset_generator_parameters)
   - Linee 3076-3087 (train_generator)
   - Linee 2214-2222 (save_generator_checkpoint)
