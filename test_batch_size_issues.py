"""
Test per identificare problemi con batch size > 1 negli adapter
"""
import sys
sys.path.insert(0, '/home/lpala/fedgfe/system')

print("=" * 80)
print("TEST BATCH SIZE ISSUES")
print("=" * 80)

try:
    import torch
    import torch.nn as nn
    from flcore.trainmodel.Audio2Visual_NoData.src.models.projection_improved import Adapter
    from flcore.trainmodel.Audio2Visual_NoData.src.models.multi_head_attention import MAPBlock

    print("\n✓ Imports successful\n")

    # Test configurations
    batch_sizes = [1, 2, 8, 16, 32]
    seq_len = 1214  # AST output sequence length
    embed_dim = 768

    # Create models
    print("Creating models...")
    adapter_clip = Adapter(input_dim=768, hidden_dims=[1024, 2048, 2048], output_dim=768)
    projection_clip = MAPBlock(n_latents=1, embed_dim=768, n_heads=8)

    adapter_t5 = Adapter(input_dim=768, hidden_dims=[1024, 2048, 2048], output_dim=4096)
    projection_t5 = MAPBlock(n_latents=17, embed_dim=4096, n_heads=8)

    print("✓ Models created\n")

    # Test CLIP adapter with different batch sizes
    print("-" * 80)
    print("TESTING CLIP ADAPTER")
    print("-" * 80)

    clip_issues = []
    for bs in batch_sizes:
        try:
            # Simulate audio embeddings from AST
            audio_emb = torch.randn(bs, seq_len, embed_dim)

            # Forward through adapter
            out_adapter = adapter_clip(audio_emb)

            # Forward through projection
            out_projection = projection_clip(out_adapter)

            # Simulate target (pooled embeddings from CLIP text encoder)
            target = torch.randn(bs, 768)

            # Compute loss
            loss_fn = nn.MSELoss()
            loss = loss_fn(out_projection, target)

            print(f"  Batch size {bs:2d}: "
                  f"audio={audio_emb.shape} → "
                  f"adapter={out_adapter.shape} → "
                  f"projection={out_projection.shape} | "
                  f"target={target.shape} | "
                  f"loss={loss.item():.4f} ✓")

            # Check shapes
            if out_projection.shape != target.shape:
                clip_issues.append(f"Batch {bs}: shape mismatch! output={out_projection.shape} vs target={target.shape}")

            # Check for NaN/Inf
            if torch.isnan(out_projection).any() or torch.isinf(out_projection).any():
                clip_issues.append(f"Batch {bs}: NaN or Inf detected in output!")

            if torch.isnan(loss) or torch.isinf(loss):
                clip_issues.append(f"Batch {bs}: NaN or Inf detected in loss!")

        except Exception as e:
            clip_issues.append(f"Batch {bs}: ERROR - {str(e)}")
            print(f"  Batch size {bs:2d}: ERROR - {str(e)}")

    # Test T5 adapter with different batch sizes
    print("\n" + "-" * 80)
    print("TESTING T5 ADAPTER")
    print("-" * 80)

    t5_issues = []
    for bs in batch_sizes:
        try:
            # Simulate audio embeddings from AST
            audio_emb = torch.randn(bs, seq_len, embed_dim)

            # Forward through adapter
            out_adapter = adapter_t5(audio_emb)

            # Forward through projection
            out_projection = projection_t5(out_adapter)

            # Simulate target (T5 text embeddings)
            target = torch.randn(bs, 17, 4096)

            # Compute loss (need to handle sequence dimension)
            loss_fn = nn.MSELoss()
            loss = loss_fn(out_projection, target)

            print(f"  Batch size {bs:2d}: "
                  f"audio={audio_emb.shape} → "
                  f"adapter={out_adapter.shape} → "
                  f"projection={out_projection.shape} | "
                  f"target={target.shape} | "
                  f"loss={loss.item():.4f} ✓")

            # Check shapes
            if out_projection.shape != target.shape:
                t5_issues.append(f"Batch {bs}: shape mismatch! output={out_projection.shape} vs target={target.shape}")

            # Check for NaN/Inf
            if torch.isnan(out_projection).any() or torch.isinf(out_projection).any():
                t5_issues.append(f"Batch {bs}: NaN or Inf detected in output!")

            if torch.isnan(loss) or torch.isinf(loss):
                t5_issues.append(f"Batch {bs}: NaN or Inf detected in loss!")

        except Exception as e:
            t5_issues.append(f"Batch {bs}: ERROR - {str(e)}")
            print(f"  Batch size {bs:2d}: ERROR - {str(e)}")

    # Test gradient flow
    print("\n" + "-" * 80)
    print("TESTING GRADIENT FLOW (batch_size=8)")
    print("-" * 80)

    bs = 8
    audio_emb = torch.randn(bs, seq_len, embed_dim, requires_grad=False)
    target_clip = torch.randn(bs, 768)
    target_t5 = torch.randn(bs, 17, 4096)

    # CLIP
    out_clip = projection_clip(adapter_clip(audio_emb))
    loss_clip = nn.MSELoss()(out_clip, target_clip)
    loss_clip.backward()

    grad_found = False
    for name, param in list(adapter_clip.named_parameters())[:3]:
        if param.grad is not None:
            print(f"  CLIP {name}: grad_norm={param.grad.norm().item():.6f}")
            grad_found = True

    if not grad_found:
        clip_issues.append("No gradients found in CLIP adapter!")
    else:
        print("  ✓ CLIP gradients OK")

    # T5
    adapter_t5.zero_grad()
    projection_t5.zero_grad()

    out_t5 = projection_t5(adapter_t5(audio_emb))
    loss_t5 = nn.MSELoss()(out_t5, target_t5)
    loss_t5.backward()

    grad_found = False
    for name, param in list(adapter_t5.named_parameters())[:3]:
        if param.grad is not None:
            print(f"  T5 {name}: grad_norm={param.grad.norm().item():.6f}")
            grad_found = True

    if not grad_found:
        t5_issues.append("No gradients found in T5 adapter!")
    else:
        print("  ✓ T5 gradients OK")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if clip_issues:
        print(f"\n⚠️  CLIP ADAPTER ISSUES ({len(clip_issues)}):")
        for issue in clip_issues:
            print(f"  - {issue}")
    else:
        print("\n✓ CLIP ADAPTER: NO ISSUES FOUND")

    if t5_issues:
        print(f"\n⚠️  T5 ADAPTER ISSUES ({len(t5_issues)}):")
        for issue in t5_issues:
            print(f"  - {issue}")
    else:
        print("\n✓ T5 ADAPTER: NO ISSUES FOUND")

    print("\n" + "=" * 80)

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
