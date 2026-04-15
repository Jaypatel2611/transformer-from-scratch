# Transformer From Scratch

A complete, modular, production-quality implementation of a GPT-2 style transformer language model from scratch. This repository provides an educational and practical reference for understanding modern transformer architecture.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

## 📋 Overview

This project implements a GPT-2 style transformer decoder-only language model. It's designed to be:

- **Educational**: Clear, well-documented code explaining each component
- **Modular**: Separated into reusable components (attention, MLP, blocks, etc.)
- **Production-Ready**: Clean imports, proper error handling, and efficient implementation
- **Interview-Ready**: Professional code structure and documentation

The model can be used for:
- Understanding transformer internals
- Fine-tuning on custom datasets
- Generating text with various sampling strategies
- Educational projects and research

## 🏗️ Architecture

### Core Components

| Module | Purpose |
|--------|----------|
| `config.py` | Model hyperparameter configuration (`GPTConfig`) |
| `attention.py` | Multi-head causal self-attention layer |
| `mlp.py` | Position-wise feed-forward network |
| `block.py` | Transformer block (attention + MLP + residuals) |
| `model.py` | Complete GPT-2 model with embeddings and output layer |
| `generation.py` | Text generation utilities |

### Model Architecture

```
Input Tokens
    ↓
Token Embedding + Positional Embedding
    ↓
Dropout
    ↓
[Transformer Block × n_layer]
  ├─ LayerNorm
  ├─ CausalSelfAttention (Multi-Head)
  ├─ LayerNorm
  └─ MLP (Position-wise Feed-Forward)
    ↓
Final LayerNorm
    ↓
Output Projection (Logits)
    ↓
Loss (if targets provided) or Generation
```

## 📦 Installation

```bash
# Clone the repository
cd transformer-from-scratch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Create and Train a Model

```python
import torch
from src import GPTConfig, GPT2

# Create model with configuration
config = GPTConfig(
    vocab_size=256,
    block_size=128,
    n_layer=12,
    n_head=8,
    n_embd=512,
    dropout=0.1
)

model = GPT2(config)

# Forward pass with random batch
batch_size, seq_len = 4, 64
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

logits, loss = model(input_ids, targets)
print(f"Loss: {loss.item():.4f}")
```

### 2. Generate Text

```python
from src import GPT2, generate

model = GPT2(config)
model.eval()

# Start with a seed token
seed = torch.tensor([[100]])

# Generate 50 new tokens
generated = generate(
    model,
    seed,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40
)

print(generated)
```

### 3. Run Example Scripts

```bash
# Educational walkthrough of transformer components
python -m examples.transformer_from_scratch_demo

# GPT-2 model demonstration
python -m examples.transformer_using_library_demo
```

## 📚 Example Usage

### Training

```python
import torch
import torch.optim as optim
from src import GPTConfig, GPT2

config = GPTConfig(vocab_size=256, block_size=128)
model = GPT2(config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):
    # Get batch
    input_ids = torch.randint(0, config.vocab_size, (32, 64))
    targets = torch.randint(0, config.vocab_size, (32, 64))
    
    # Forward pass
    logits, loss = model(input_ids, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Generation with Sampling Strategies

```python
# Temperature sampling (lower = more deterministic)
generated = model.generate(seed, max_new_tokens=100, temperature=0.7)

# Top-k sampling (only sample from top-k most likely)
generated = model.generate(seed, max_new_tokens=100, top_k=50)

# Greedy decoding (argmax)
generated = model.generate(seed, max_new_tokens=100, temperature=0.0)
```

## 🔍 Configuration Options

The `GPTConfig` class controls model behavior:

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257      # Vocabulary size
    block_size: int = 1024       # Maximum sequence length
    n_layer: int = 12            # Number of transformer blocks
    n_head: int = 12             # Number of attention heads
    n_embd: int = 768            # Embedding dimension
    dropout: float = 0.1         # Dropout probability
```

**Tips for configuration:**
- `vocab_size`: Depends on your tokenizer
- `block_size`: Longer = more context but more computation
- `n_layer`: More layers = more expressiveness but slower
- `n_head`: Should divide `n_embd` evenly
- `n_embd`: Larger = more capacity but more parameters

## 📁 Project Structure

```
transformer-from-scratch/
├── src/
│   ├── __init__.py
│   ├── config.py              # Model configuration
│   ├── attention.py           # Self-attention layer
│   ├── mlp.py                 # Feed-forward network
│   ├── block.py               # Transformer block
│   ├── model.py               # GPT-2 model
│   └── generation.py          # Text generation utilities
├── examples/
│   ├── transformer_from_scratch_demo.py
│   └── transformer_using_library_demo.py
├── docs/
│   └── transformer-architecture.md
├── tests/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## 🎓 Educational Components

The implementation includes detailed docstrings and comments explaining:

- **Multi-head Attention**: Query, Key, Value projection and causal masking
- **Layer Normalization**: Stabilization before attention and MLP layers
- **Residual Connections**: Skip connections for gradient flow
- **Feed-Forward Network**: Position-wise MLP with GELU activation
- **Weight Tying**: Sharing parameters between embedding and output layers
- **Autoregressive Generation**: Sequential token prediction with sampling

## 🔧 Key Features

✅ **Clean Modular Design**: Each component is independent and reusable  
✅ **Type Hints**: Full type annotations for clarity  
✅ **Production Imports**: No circular imports, clean dependency graph  
✅ **Comprehensive Docstrings**: Every function and class is documented  
✅ **Flexible Configuration**: Easy to customize model size and behavior  
✅ **Generation Utilities**: Multiple sampling strategies  
✅ **Error Handling**: Proper validation and error messages  

## 💡 Implementation Details

### Attention Mechanism

The attention mechanism computes:
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

With causal masking to prevent attending to future tokens:
```python
scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
```

### Residual Connections

Pre-normalization pattern:
```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

This provides better gradient flow during training.

### Weight Tying

The embedding and output layer share weights:
```python
self.lm_head.weight = self.wte.weight
```

This reduces parameters and can improve generalization.

## 📖 References

Key papers:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (Radford et al., 2019) - GPT-2 paper

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional sampling strategies (beam search, nucleus sampling)
- Performance optimizations
- Additional documentation and tutorials
- Test suite expansion

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This implementation was created as an educational resource for understanding transformer architecture. The design prioritizes clarity and learning over performance optimization.

---

**Built with clarity, precision, and educational value in mind.**
#   t r a n s f o r m e r - f r o m - s c r a t c h  
 