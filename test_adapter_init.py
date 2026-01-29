#!/usr/bin/env python3
"""
Diagnostic script to check adapter initialization
"""
import sys
sys.path.append('/home/lpala/fedgfe/system')

import torch
from flcore.trainmodel.downstreamsinestesiaadapters import DownstreamSinestesiaAdapters
from types import SimpleNamespace

# Create minimal args
args = SimpleNamespace(
    device='cpu',
    model='hf_vit',
    pretrained=False
)

# Test adapter creation
print("Testing DownstreamSinestesiaAdapters initialization...")
model = DownstreamSinestesiaAdapters(
    args=args,
    wandb_log=False,
    device='cpu',
    use_classifier_loss=False,
    diffusion_type='flux',  # Same as config
    enable_diffusion=False,
    use_cls_token_only=False,
    adapter_dropout=0.1,
    generators_dict=None,
    init_ast_model=False
)

print(f"\nAdapters dict keys: {list(model.adapters.keys())}")
print(f"Number of adapters: {len(model.adapters)}")

for adapter_name, adapter in model.adapters.items():
    print(f"\n{adapter_name} adapter:")
    print(f"  Type: {type(adapter)}")
    print(f"  Modules: {adapter}")

    params = list(adapter.parameters())
    print(f"  Number of parameters: {len(params)}")

    if len(params) > 0:
        print(f"  First param shape: {params[0].shape}")
        print(f"  First param requires_grad: {params[0].requires_grad}")
        total_params = sum(p.numel() for p in params)
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    else:
        print(f"  WARNING: No parameters found!")

print("\n" + "="*60)
print("Testing optimizer creation...")
print("="*60)

trainable_params_dict = {}
for adapter_name, adapter in model.adapters.items():
    trainable_params_dict[adapter_name] = adapter.parameters()

print(f"trainable_params_dict keys: {list(trainable_params_dict.keys())}")

# Try to create optimizers
train_optimizers = {}
for module_name, params in trainable_params_dict.items():
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
    train_optimizers[module_name] = optimizer
    print(f"\nCreated optimizer for '{module_name}':")
    print(f"  {optimizer}")

    for pg_idx, param_group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in param_group["params"] if p.requires_grad)
        print(f"  Param group {pg_idx}: {num_params:,} trainable parameters")

print(f"\nTotal optimizers created: {len(train_optimizers)}")

if len(train_optimizers) == 0:
    print("\n❌ ERROR: No optimizers were created!")
    print("This explains why training fails!")
else:
    print(f"\n✓ Successfully created {len(train_optimizers)} optimizers")
