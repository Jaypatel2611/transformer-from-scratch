"""
GPT-style Transformer implementation from scratch.

This package provides a clean, modular implementation of a GPT-2 style
transformer language model built from first principles using PyTorch.

Main components:
    - GPTConfig: Model hyperparameter configuration
    - CausalSelfAttention: Multi-head causal self-attention layer
    - MLP: Position-wise feed-forward network
    - Block: Transformer block combining attention and MLP
    - GPT2: Complete language model
    - generate: Convenience function for text generation

Example usage:
    >>> from src import GPTConfig, GPT2, generate
    >>> config = GPTConfig(vocab_size=256, block_size=128)
    >>> model = GPT2(config)
    >>> initial_tokens = torch.zeros((1, 1), dtype=torch.long)
    >>> generated = generate(model, initial_tokens, max_new_tokens=100)
"""

from .config import GPTConfig
from .attention import CausalSelfAttention
from .mlp import MLP
from .block import Block
from .model import GPT2
from .generation import generate

__all__ = [
    "GPTConfig",
    "CausalSelfAttention",
    "MLP",
    "Block",
    "GPT2",
    "generate",
]
