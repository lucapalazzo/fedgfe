# ClientA2V Local Adapter Mechanism

## Overview

The `clientA2V` class implements a **local adapter training mechanism** where each federated node:

1. Creates **local copies** of adapters and projections
2. Trains **only these local copies** (not the global model's backbone)
3. Synchronizes trained parameters back to the global model
4. Shares trained adapters with the server for federated aggregation

This ensures that:
- The **backbone** (AST model, diffusion pipeline) remains **frozen**
- Only **lightweight adapters/projections** are trained per node
- Memory and computation are optimized
- Federated learning focuses on adapter parameters only

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Global Model                     │
│  ┌───────────────────────────────────────────────┐ │
│  │         Audio2Image (Frozen Backbone)         │ │
│  │  - AST Model (frozen)                         │ │
│  │  - Diffusion Pipeline (frozen)                │ │
│  │                                                │ │
│  │  Trainable Components (to be copied):         │ │
│  │  - clip_adapter                               │ │
│  │  - clip_projection                            │ │
│  │  - t5_adapter                                 │ │
│  │  - t5_projection                              │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                         │
                         │ __init__: deepcopy adapters
                         ↓
┌─────────────────────────────────────────────────────┐
│                   Client Node                        │
│  ┌───────────────────────────────────────────────┐ │
│  │          Uses Global Backbone (shared)        │ │
│  │  - AST Model ──────────────────────────┐      │ │
│  │  - Diffusion Pipeline                  │      │ │
│  │                                         │      │ │
│  │          Local Copies (node-specific)  │      │ │
│  │  - local_clip_adapter     ◄────────────┘      │ │
│  │  - local_clip_projection                      │ │
│  │  - local_t5_adapter                           │ │
│  │  - local_t5_projection                        │ │
│  │                                                │ │
│  │  Optimizer → tracks ONLY local copies         │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                         │
                         │ sync_local_to_global()
                         ↓
              Updates global model adapters
```

---

## Key Methods

### 1. `_create_and_inject_local_adapters()`

Called in `__init__`, this method:

```python
# Create deep copies of adapters
self.local_clip_adapter = copy.deepcopy(self.audio2image_model.clip_adapter)
self.local_clip_projection = copy.deepcopy(self.audio2image_model.clip_projection)
self.local_t5_adapter = copy.deepcopy(self.audio2image_model.t5_adapter)
self.local_t5_projection = copy.deepcopy(self.audio2image_model.t5_projection)

# Inject local copies into the model
self.audio2image_model.clip_adapter = self.local_clip_adapter
self.audio2image_model.clip_projection = self.local_clip_projection
# ... etc
```

**Result**: The model now uses local copies for forward passes and gradient computation.

---

### 2. `setup_optimizer()`

Sets up the optimizer to track **only local adapter parameters**:

```python
trainable_params = []

if hasattr(self, 'local_clip_adapter') and self.local_clip_adapter is not None:
    trainable_params.extend(self.local_clip_adapter.parameters())

if hasattr(self, 'local_clip_projection') and self.local_clip_projection is not None:
    trainable_params.extend(self.local_clip_projection.parameters())

# ... same for t5_adapter and t5_projection

self.train_optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
```

**Result**: Only local adapters receive gradient updates.

---

### 3. `train()` → `train_a2v()` → `sync_local_to_global()`

Training flow:

1. **`train()`**: Calls `train_a2v()` for local training
2. **`train_a2v()`**: Runs gradient descent on local adapters
3. **`sync_local_to_global()`**: Copies trained local adapter weights to global model

```python
# In train() method:
self.train_a2v(local_epochs, trainloader, client_device=device)

# Synchronize trained local adapters back to global model
self.sync_local_to_global()
```

**Result**: After training, the global model reflects the node's trained adapter state.

---

### 4. `sync_local_to_global()`

Synchronizes trained local adapters to the global model:

```python
def sync_local_to_global(self):
    global_audio2image = self.global_model.get_audio2image_model()

    # Copy parameters from local to global
    for local_param, global_param in zip(
        self.local_clip_adapter.parameters(),
        global_audio2image.clip_adapter.parameters()
    ):
        global_param.data.copy_(local_param.data)

    # ... same for other adapters

    # Verify optimizer still tracks local adapters
    self._verify_optimizer_tracking()
```

**Result**: Global model's adapters updated with trained weights.

---

### 5. `_verify_optimizer_tracking()`

Sanity check to ensure optimizer tracks local adapters:

```python
def _verify_optimizer_tracking(self):
    optimizer_param_ids = {id(p) for pg in self.train_optimizer.param_groups
                                   for p in pg['params']}

    local_param_ids = set()
    for param in self.local_clip_adapter.parameters():
        local_param_ids.add(id(param))
    # ... collect all local param IDs

    tracked_count = len(local_param_ids & optimizer_param_ids)
    total_local = len(local_param_ids)

    return tracked_count == total_local
```

**Output**: ✓ or ⚠ verification message.

---

### 6. `set_parameters(global_model)`

Updates local adapters from the global model (after federated aggregation):

```python
def set_parameters(self, global_model):
    global_audio2image = global_model.get_audio2image_model()

    # Update local adapters from global
    for global_param, local_param in zip(
        global_audio2image.clip_adapter.parameters(),
        self.local_clip_adapter.parameters()
    ):
        if not torch.equal(local_param.data, global_param.data):
            local_param.data.copy_(global_param.data)

    # ... same for other adapters
```

**Result**: Local adapters updated with aggregated global weights.

---

### 7. `get_local_adapter_state_dict()` / `load_local_adapter_state_dict()`

For federated aggregation:

```python
# Extract local adapter state for sharing with server
state_dict = client.get_local_adapter_state_dict()
# Returns: {'clip_adapter': ..., 'clip_projection': ..., ...}

# Load aggregated state from server
client.load_local_adapter_state_dict(aggregated_state_dict)
```

---

## Training Flow

### Federated Round Lifecycle

```
1. INITIALIZATION (Round 0)
   └─> Node creates local adapter copies from global model

2. FOR EACH ROUND:

   a) RECEIVE GLOBAL MODEL
      └─> set_parameters(global_model)
          └─> Updates local adapters with aggregated weights

   b) LOCAL TRAINING
      └─> train()
          └─> train_a2v()
              └─> Forward pass through model (uses local adapters)
              └─> Backward pass updates local adapters only
              └─> Optimizer steps on local adapter parameters

   c) SYNC TO GLOBAL
      └─> sync_local_to_global()
          └─> Copies trained local adapters to global model
          └─> Verifies optimizer still tracks local adapters

   d) SHARE WITH SERVER
      └─> Server collects: get_local_adapter_state_dict()
      └─> Server aggregates adapter weights
      └─> Server updates global model

3. REPEAT for N rounds
```

---

## Parameter Tracking Verification

The implementation includes automatic verification that the optimizer correctly tracks local parameters:

```
Node 0 - Creating local copies of adapters and projections
  - Created and injected local clip_adapter
  - Created and injected local clip_projection
Node 0 - Total trainable parameters in local adapters: 1,182,720

Node 0 - Optimizer verification: ✓ All 1182720 local parameters tracked
```

---

## Benefits

1. **Memory Efficient**: Only lightweight adapters are replicated per node
2. **Computation Efficient**: Frozen backbone reduces computational overhead
3. **Clear Separation**: Local vs. global parameters are explicitly managed
4. **Verifiable**: Automatic tracking verification ensures correctness
5. **Flexible**: Supports different diffusion types (SD, Flux, CogX)

---

## Component Overview

### Trainable Components (Local Copies)

For **Stable Diffusion (SD)**:
- `clip_adapter`: Maps AST output to CLIP space
- `clip_projection`: Projects to SD prompt embeddings

For **Flux**:
- `clip_adapter`: CLIP space adapter
- `clip_projection`: Pooled prompt embeddings projection
- `t5_adapter`: Maps AST output to T5 space
- `t5_projection`: Projects to T5 prompt embeddings

For **CogX**:
- `t5_adapter`: Maps AST output to T5 space
- `t5_projection`: Projects to CogX prompt embeddings

### Frozen Components (Shared)

- **AST Model**: Audio spectrogram transformer (feature extractor)
- **Diffusion Pipeline**: Image generation pipeline (SD/Flux/CogX)
- **Feature Extractor**: Audio preprocessing

---

## Example Usage

```python
# Server side
global_model = create_global_a2v_model()

# Node initialization
client = clientA2V(args, node_id=0, node_config=config, global_model=global_model)
# → Creates local adapter copies

# Training
client.train()
# → Trains local adapters
# → Syncs to global model

# Server aggregation
adapter_states = [client.get_local_adapter_state_dict() for client in clients]
aggregated = aggregate_adapters(adapter_states)

# Update global and distribute
global_model.load_adapter_state_dict(aggregated)
for client in clients:
    client.set_parameters(global_model)
```

---

## Testing

To verify the mechanism works correctly:

1. Check local adapter creation: `client.local_clip_adapter is not None`
2. Check parameter separation: `id(client.local_clip_adapter) != id(global_model.audio2image.clip_adapter)`
3. Check optimizer tracking: `client._verify_optimizer_tracking()` returns `True`
4. Train and verify only locals change before sync
5. Call `sync_local_to_global()` and verify global model updates

---

## Notes

- The optimizer **always** tracks local adapter parameters (never global)
- `sync_local_to_global()` does **not** change optimizer tracking (it only copies parameter values)
- `set_parameters()` updates local adapter **data**, not references
- The global model's backbone remains frozen and shared across all nodes
