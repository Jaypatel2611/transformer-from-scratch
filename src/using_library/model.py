"""GPT-2 style language model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .block import Block


class GPT2(nn.Module):
    """GPT-2 style causal language model.
    
    Implements a transformer-based language model with:
    - Token and positional embeddings
    - Stack of transformer blocks
    - Language model head for next-token prediction
    - Optional cross-entropy loss computation during training
    
    The model uses weight tying: the embedding weights are shared with
    the output layer weights for efficiency.
    """
    
    def __init__(self, config: GPTConfig):
        """Initialize the GPT-2 model.
        
        Args:
            config: GPTConfig object containing model hyperparameters.
        """
        super().__init__()
        self.config = config
        
        # Input embedding layers
        # Token embedding: converts token IDs to embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional embedding: adds positional information
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        # Dropout after embeddings
        self.drop = nn.Dropout(config.dropout)
        
        # Stack of transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output layer: projects from embedding dimension to vocabulary size
        # Note: weight is tied with wte (word token embedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between embedding and output layer
        self.lm_head.weight = self.wte.weight
    
    def forward(self, idx, targets=None):
        """Forward pass of the language model.
        
        Args:
            idx: Input token IDs of shape (batch_size, seq_len)
            targets: Optional target token IDs of shape (batch_size, seq_len)
                    for computing cross-entropy loss during training.
                    If provided, should be the tokens to predict.
            
        Returns:
            logits: Predicted logits of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (only if targets are provided)
        """
        B, T = idx.size()
        
        # Validate sequence length
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block size {self.config.block_size}"
        )
        
        # Generate position indices
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # Get embeddings and combine with positional embeddings
        token_emb = self.wte(idx)      # (B, T, n_embd)
        pos_emb = self.wpe(pos)        # (1, T, n_embd)
        x = token_emb + pos_emb        # (B, T, n_embd) - broadcasting
        
        # Apply dropout to embeddings
        x = self.drop(x)
        
        # Apply transformer blocks
        for block in self.h:
            x = block(x)
        
        # Apply final layer normalization
        x = self.ln_f(x)
        
        # Project to vocabulary size
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy loss computation
            # Cross-entropy expects shape (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens auto-regressively.
        
        Generates tokens one at a time by:
        1. Running the model on the current context
        2. Getting logits for the last position
        3. Sampling the next token
        4. Appending it to the sequence
        5. Repeating until max_new_tokens are generated
        
        Args:
            idx: Initial token IDs of shape (batch_size, initial_seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (> 1.0 = more random, < 1.0 = more deterministic)
            top_k: If specified, only sample from top-k most likely tokens
            
        Returns:
            Generated token IDs of shape (batch_size, initial_seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds the maximum block size
            idx_cond = idx[:, -self.config.block_size:]
            
            # Get predictions from the model
            logits, _ = self(idx_cond)
            
            # Use only the logits for the last position
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature scaling
            logits = logits / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                # Find the k largest values
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all values below the k-th largest to -inf
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the sampled token to the sequence
            idx = torch.cat((idx, next_token), dim=1)
        
        return idx
