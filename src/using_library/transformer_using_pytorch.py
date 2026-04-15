import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__() # It will call the constructor of nn.Module class
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # The single fused linear layer for Q, K & V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False) # Infact of 3 seperate LL we use 1 fused and much faster LL
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # The Final Projection
        self.resid_drop = nn.Dropout(config.dropout)
        self.bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", self.bias)

    def forward(self, x):
        B, T, C = x.size()

        # 1. Get Q, K & V from a single LL & SPLIT into heads
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hd)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hd)

        # 2. Run Casual Mask on each head in parallel
        att = (q @ k.transpose(-2, -1)) / (1.0 / math.sqrt(head_dim))
        # [THE CASUAL MASK] We slice our sorted mask to match the current sequence length T
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        # 3. Merging heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        
        return y
    
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__() # It will call the constructor of nn.Module class
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)  # GPT-2 uses GELU
        x = self.drop(self.proj(x))
        return x
    
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__() # It will call the constructor of nn.Module class
        # We define the layerNORM layers here
        self.ln_1 = nn.LayerNorm(config.n_embd) # The STABILISER we just built
        self.attn = CausalSelfAttention(config) # The COMMUNICATION Layer we built
        self.ln_2 = nn.LayerNorm(config.n_embd) # Another STABILISER
        self.mlp = MLP(config) # The THINKING Layer we built
    
    def forward(self, x):
        """
        The Forward pass of a single Transformer Block
        Mantra: Normalise, Process, Add
        """
        # --- LayerNorm is applied BEFORE the sub-layer ---
        x = x + self.attn(self.ln_1(x))
        # --- And here again ---
        x = x + self.mlp(self.ln_2(x))

        return x
    
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__() # It will call the constructor of nn.Module class
        self.config = config

        # Part-1: The Input Layers
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # nn.Embedding is to convert words into meaningful vectors : A giant Lookup Table
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Part-2: The Core Processing Layers
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Part-3: The Output Layers
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block size."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    @torch.no_grad() # It tells pytorch, we arn't not training, we are just predicting
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Crop the context if it's too long
            idx_cond = idx[:, -self.config.block_size:]

            # Predict: get the logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # We throw away all the predictions except the last one.This single line connects "wasteful" trainig process to efficient generation

            # (Optional) Top K- Filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample: convert the possibility into sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # It performs the "roll dice", picking one token based on those probalities.

            # Append: add the new token to the sequence
            idx = torch.cat((idx, next_token), dim=1)

        return idx
