"""Configuration module for GPT-style transformer model."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Hyperparameter configuration for GPT-style transformer model."""
    
    vocab_size: int = 50257    # Vocabulary size (default for GPT-2)
    block_size: int = 1024     # Maximum context window / sequence length
    n_layer: int = 12          # Number of transformer blocks
    n_head: int = 12           # Number of attention heads
    n_embd: int = 768          # Embedding dimensionality
    dropout: float = 0.1       # Dropout probability for regularization
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )