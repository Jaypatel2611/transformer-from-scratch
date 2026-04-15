"""Text generation utilities."""

import torch


def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Generate text using a language model.
    
    This is a convenience wrapper around the model's generate method.
    For advanced use cases, you can call model.generate() directly.
    
    Args:
        model: Language model instance with a generate method
        idx: Initial token IDs of shape (batch_size, initial_seq_len)
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature (> 1.0 = more random, < 1.0 = more deterministic)
        top_k: If specified, only sample from top-k most likely tokens
        
    Returns:
        Generated token IDs of shape (batch_size, initial_seq_len + max_new_tokens)
        
    Example:
        >>> from src import GPT2, GPTConfig, generate
        >>> config = GPTConfig(vocab_size=256)
        >>> model = GPT2(config)
        >>> initial_tokens = torch.zeros((1, 1), dtype=torch.long)
        >>> generated = generate(model, initial_tokens, max_new_tokens=50)
    """
    return model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
