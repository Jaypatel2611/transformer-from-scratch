"""Feed-forward network (MLP) module."""

import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class MLP(nn.Module):
    """Position-wise feed-forward network.
    
    Implements a two-layer fully connected network with GELU activation,
    as used in the original Transformer architecture:
    - First layer: expands from n_embd to 4*n_embd
    - GELU activation
    - Second layer: projects back from 4*n_embd to n_embd
    - Dropout for regularization
    """
    
    def __init__(self, config: GPTConfig):
        """Initialize the MLP layer.
        
        Args:
            config: GPTConfig object containing model hyperparameters.
        """
        super().__init__()
        
        # Expansion layer: projects from n_embd to 4*n_embd
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        
        # Projection layer: projects back from 4*n_embd to n_embd
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
        # Dropout for regularization
        self.drop = nn.Dropout(config.dropout)
    
    def forward(self, x):
        """Apply feed-forward transformation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Expand
        x = self.fc(x)
        
        # Activate with GELU (used in GPT-2)
        x = F.gelu(x)
        
        # Project back and apply dropout
        x = self.drop(self.proj(x))
        
        return x
