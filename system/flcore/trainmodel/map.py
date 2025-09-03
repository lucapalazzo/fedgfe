import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAPAttention(nn.Module):
    """
    Multihead Attention pooling using PyTorch 2.0's scaled_dot_product_attention.
    Computes attention between learnable seed queries and input tokens without external dependencies.
    """
    def __init__(self, embed_dim: int, n_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        # projection layers for Q, K, V and output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        query: (B, K, D)
        key_value: (B, N, D)
        returns: (B, K, D)
        """
        B, K, D = query.shape
        # project
        Q = self.q_proj(query)
        K_ = self.k_proj(key_value)
        V = self.v_proj(key_value)
        # reshape for multi-head: (B, K, H, head_dim)
        Q = Q.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, K, head_dim)
        K_ = K_.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        V = V.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        # scaled dot-product attention (PyTorch 2.0)
        attn_out = F.scaled_dot_product_attention(Q, K_, V, attn_mask=None, dropout_p=0.0, is_causal=False)
        # attn_out: (B, H, K, head_dim)
        # combine heads
        attn_out = attn_out.transpose(1, 2).reshape(B, K, D)  # (B, K, D)
        return self.out_proj(attn_out)

class MAPBlock(nn.Module):
    """
    Set Transformer MAP block: attention + feedforward on latents.
    Aggregates N tokens to K latents.
    """
    def __init__(self, embed_dim: int, n_heads: int = 8, n_latents: int = 1, mlp_ratio: float = 4.0):
        super().__init__()
        self.n_latents = n_latents
        self.embed_dim = embed_dim
        self.latents = nn.Parameter(torch.zeros(n_latents, embed_dim))

        self.attn = MAPAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.latents)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)

        attn_out = self.attn(latents, x)
        latents = self.norm1(latents + attn_out)

        mlp_out = self.mlp(latents)
        latents = self.norm2(latents + mlp_out)
        return latents  # (B, K, D)