"""Transformer block module."""

import torch.nn as nn
import torch
from .config import GPTConfig
from .attention import CausalSelfAttention
from .mlp import MLP


class Block(nn.Module):
    """Transformer block (encoder block).
    
    Combines self-attention and feed-forward layers with layer normalization
    and residual connections. Uses pre-normalization (normalize before applying
    sub-layer), which is more stable than post-normalization.
    
    Architecture:
        x -> LayerNorm -> CausalSelfAttention -> + x (residual)
        x -> LayerNorm -> MLP -> + x (residual)
    """
    
    def __init__(self, config: GPTConfig):
        """Initialize the transformer block.
        
        Args:
            config: GPTConfig object containing model hyperparameters.
        """
        super().__init__()
        
        # Layer normalization before attention
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # Multi-head self-attention layer
        self.attn = CausalSelfAttention(config)

        # Layer normalization before MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Feed-forward network
        self.mlp = MLP(config)
    
    def forward(self, x):
        """Apply transformer block layers with residual connections.
        
        Uses pre-normalization pattern:
            - x = x + Attention(LayerNorm(x))
            - x = x + MLP(LayerNorm(x))
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Self-attention block with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # Feed-forward (MLP) block with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        
        return x