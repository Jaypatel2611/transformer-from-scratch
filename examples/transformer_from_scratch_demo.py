"""Educational demonstration of transformer components from scratch.

This script walks through the key components of a transformer architecture,
showing how embeddings, attention, MLPs, and residual connections work
at a fundamental level.

Run with: python -m examples.transformer_from_scratch_demo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def demonstrate_embeddings():
    """Show how token and positional embeddings work."""
    print("\n" + "="*60)
    print("1. EMBEDDINGS: Converting Tokens to Vectors")
    print("="*60)
    
    # Token Embedding
    vocab_size = 10
    n_embd = 3
    
    token_embedding_table = nn.Embedding(vocab_size, n_embd)
    print(f"\nToken Embedding Shape: {token_embedding_table.weight.shape}")
    print(f"Embedding Table (first 3 rows):\n{token_embedding_table.weight[:3]}")
    
    # Positional Embedding
    B, T, C = 2, 5, 3
    block_size = 8
    position_embedding_table = nn.Embedding(block_size, C)
    
    idx = torch.randint(0, vocab_size, (B, T))
    tok_emb = token_embedding_table(idx)
    pos = torch.arange(0, T, dtype=torch.long)
    pos_emb = position_embedding_table(pos)
    x = tok_emb + pos_emb
    
    print(f"\nBatch size: {B}, Sequence length: {T}, Embedding dim: {C}")
    print(f"Token embedding shape: {tok_emb.shape}")
    print(f"Position embedding shape: {pos_emb.shape}")
    print(f"Combined embedding shape: {x.shape}")


def demonstrate_self_attention():
    """Show how basic self-attention works."""
    print("\n" + "="*60)
    print("2. SELF-ATTENTION: How Tokens Interact")
    print("="*60)
    
    B, T, C = 1, 4, 2
    
    # Sample input: "A crane ate fish"
    X = torch.tensor([[[0.1, 0.1],
                       [1.0, 0.2],
                       [0.1, 0.9],
                       [0.8, 0.0]]]).float()
    
    print(f"\nInput sequence: 'A crane ate fish'")
    print(f"Input shape: {X.shape}")
    
    # Project to Q, K, V
    q_proj = nn.Linear(C, C, bias=False)
    k_proj = nn.Linear(C, C, bias=False)
    v_proj = nn.Linear(C, C, bias=False)
    
    torch.manual_seed(42)
    q_proj.weight.data = torch.randn(C, C)
    k_proj.weight.data = torch.randn(C, C)
    v_proj.weight.data = torch.randn(C, C)
    
    q = q_proj(X)
    k = k_proj(X)
    v = v_proj(X)
    
    # Compute attention scores
    scores = q @ k.transpose(-2, -1)
    d_k = k.size(-1)
    scaled_scores = scores / math.sqrt(d_k)
    attention_weights = F.softmax(scaled_scores, dim=-1)
    
    print(f"\nAttention weights (before causal mask):\n{attention_weights}")
    
    # Apply causal mask
    mask = torch.tril(torch.ones(scaled_scores.shape[1], scaled_scores.shape[1]))
    masked_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))
    causal_attention_weights = F.softmax(masked_scores, dim=-1)
    
    print(f"\nAttention weights (after causal mask):\n{causal_attention_weights}")
    
    # Aggregate values
    output = causal_attention_weights @ v
    print(f"\nAttention output shape: {output.shape}")


def demonstrate_multihead_attention():
    """Show how multi-head attention works."""
    print("\n" + "="*60)
    print("3. MULTI-HEAD ATTENTION: Parallel Attention Heads")
    print("="*60)
    
    B, T, C = 1, 4, 768
    n_head = 12
    
    q = torch.randn(B, T, C)
    k = torch.randn(B, T, C)
    v = torch.randn(B, T, C)
    
    head_dim = C // n_head
    
    # Reshape for multi-head attention
    q_reshaped = q.view(B, T, n_head, head_dim)
    q_final = q_reshaped.transpose(1, 2)
    
    k_final = k.view(B, T, n_head, head_dim).transpose(1, 2)
    v_final = v.view(B, T, n_head, head_dim).transpose(1, 2)
    
    print(f"\nOriginal query shape: {q.shape}")
    print(f"Reshaped for {n_head} heads: {q_final.shape}")
    print(f"Head dimension: {head_dim}")
    
    # Attention across all heads in parallel
    scaled_scores = (q_final @ k_final.transpose(-2, -1)) / math.sqrt(head_dim)
    attention_weights = F.softmax(scaled_scores, dim=-1)
    output_per_head = attention_weights @ v_final
    
    print(f"Attention output per head: {output_per_head.shape}")
    
    # Merge heads back
    merged_output = output_per_head.transpose(1, 2).contiguous().view(B, T, C)
    c_proj = nn.Linear(C, C)
    final_output = c_proj(merged_output)
    
    print(f"Final merged output shape: {final_output.shape}")


def demonstrate_mlp():
    """Show how the MLP (feed-forward) layer works."""
    print("\n" + "="*60)
    print("4. MLP: Position-Wise Feed-Forward Network")
    print("="*60)
    
    # The expansion, activation, and contraction patterns
    x = torch.tensor([[[0.5, -0.5]]])  # Single token input (1, 1, 2)
    
    fc = nn.Linear(2, 8)
    torch.manual_seed(1337)
    fc.weight.data = torch.randn(8, 2)
    fc.bias.data = torch.randn(8)
    x_expanded = fc(x)
    
    print(f"\nExpansion layer - input: {x.shape} -> output: {x_expanded.shape}")
    
    x_activated = F.gelu(x_expanded)
    print(f"After GELU activation: {x_activated.shape}")
    
    proj = nn.Linear(8, 2)
    torch.manual_seed(42)
    proj.weight.data = torch.randn(2, 8)
    proj.bias.data = torch.randn(2)
    x_projected = proj(x_activated)
    
    print(f"Contraction layer - output: {x_projected.shape}")
    
    drop = nn.Dropout(0.1)
    final_output = drop(x_projected)
    print(f"After dropout: {final_output.shape}")


def demonstrate_residual_and_layernorm():
    """Show residual connections and layer normalization."""
    print("\n" + "="*60)
    print("5. RESIDUAL CONNECTIONS & LAYER NORMALIZATION")
    print("="*60)
    
    # Residual connection
    x_initial = torch.tensor([[[0.2, 0.1, 0.3, 0.4]]])
    attention_output = torch.tensor([[[0.1, -0.1, 0.2, -0.2]]])
    
    print(f"\nResidual Connection:")
    print(f"Original X: {x_initial}")
    print(f"Attention output (adjustment): {attention_output}")
    x_after_attn = x_initial + attention_output
    print(f"After residual: {x_after_attn}")
    
    # Layer normalization
    print(f"\nLayer Normalization:")
    x_token = torch.tensor([[[0.3, -0.2, 0.8, 0.5]]])
    mean = x_token.mean(dim=-1, keepdim=True)
    std = x_token.std(dim=-1, keepdim=True)
    
    print(f"Input: {x_token}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    eps = 1e-5
    x_hat = (x_token - mean) / (torch.sqrt(std**2 + eps))
    print(f"Normalized: {x_hat}")
    
    # With learnable parameters
    gamma = torch.tensor([1.5, 1.0, 1.0, 1.0])
    beta = torch.tensor([0.5, 0.0, 0.0, 0.0])
    y = gamma * x_hat + beta
    print(f"After learnable transform: {y}")


if __name__ == "__main__":
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  TRANSFORMER COMPONENTS - EDUCATIONAL WALKTHROUGH  ".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    demonstrate_embeddings()
    demonstrate_self_attention()
    demonstrate_multihead_attention()
    demonstrate_mlp()
    demonstrate_residual_and_layernorm()
    
    print("\n" + "="*60)
    print("Demonstration complete!")
    print("="*60 + "\n")
