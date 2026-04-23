"""Multi-head causal self-attention module."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer.
    
    Implements scaled dot-product attention with causal masking to ensure
    that tokens can only attend to themselves and previous tokens (for
    autoregressive generation).
    """
    
    def __init__(self, config: GPTConfig):
        """Initialize the attention layer.
        
        Args:
            config: GPTConfig object containing model hyperparameters.
        """
        super().__init__()
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Fused linear layer: projects input to Q, K, V matrices
        # Shape: (C) -> (3*C) for split into Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        
        # Output projection after concatenating heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Dropout for regularization
        self.resid_drop = nn.Dropout(config.dropout)
        
        # Create causal mask as a buffer (not a parameter)
        # Shape: (1, 1, block_size, block_size) - lower triangular matrix
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        ))
    
    def forward(self, x):
        """Apply multi-head causal self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Attention output of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()
        
        # Project input to Q, K, V and split
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for multi-head attention
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: Q @ K.T / sqrt(head_dim)
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask: set future positions to -inf before softmax
        # Access the registered buffer and slice it to match current sequence length T
        mask = self.bias[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to convert scores to attention weights
        weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = weights @ v

        # Merge heads back together
        # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply output projection and dropout
        y = self.resid_drop(self.c_proj(y))

        return y