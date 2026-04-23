"""
Transformer From Scratch - Complete GPT-2 style implementation.

This package provides a clean, modular implementation of a GPT-2 style
transformer language model built from first principles using PyTorch.

The main public API is exported here for clean imports:
    >>> from src import GPTConfig, GPT2, generate, CausalSelfAttention, Block, MLP

For more details, see the src.using_library module which contains the full
implementation.
"""

# Import and re-export the public API from using_library
from .using_library import (
    GPTConfig,
    CausalSelfAttention,
    MLP,
    Block,
    GPT2,
    generate,
)

__all__ = [
    "GPTConfig",
    "CausalSelfAttention",
    "MLP",
    "Block",
    "GPT2",
    "generate",
]

__version__ = "1.0.0"
