# Transformer From Scratch

A clean, modular, educational implementation of a GPT-2 style transformer language model built from first principles using PyTorch. Designed for learning, understanding, and validating core transformer concepts.

## Why This Project Exists

Modern transformer models power large language models, but their internal mechanics often remain opaque. This repository provides:

- **A clear reference implementation** showing how transformers actually work at each layer
- **Modular, reusable components** (attention, MLP, blocks) that map directly to research papers
- **Professional code structure** suitable for interviews and portfolio demonstration
- **Honest documentation** about what this is (an educational tool) and what it is not (a production LLM)

## Features

✅ **Clean Architecture**: Separated components (config → attention → MLP → block → model)  
✅ **Full Type Hints**: Every function is annotated for clarity  
✅ **Comprehensive Tests**: Validates config, components, and end-to-end forward passes  
✅ **Flexible Configuration**: Easy to customize model size, depth, and behavior  
✅ **Text Generation**: Multiple sampling strategies (temperature, top-k)  
✅ **Production Imports**: Clean dependency graph, no circular imports  
✅ **Detailed Docstrings**: Every class and function explains its role  

## Project Structure

```
transformer-from-scratch/
├── src/
│   ├── __init__.py                  # Public API exports
│   ├── using_library/
│   │   ├── __init__.py
│   │   ├── config.py                # GPTConfig dataclass with validation
│   │   ├── attention.py             # CausalSelfAttention (multi-head)
│   │   ├── mlp.py                   # MLP (position-wise feed-forward)
│   │   ├── block.py                 # Transformer block (attn + MLP + residuals)
│   │   ├── model.py                 # GPT2 model (embeddings + stack + logits)
│   │   └── generation.py            # generate() utility wrapper
│   └── from_scratch/                # (reference only, not part of main API)
├── examples/
│   ├── minimal_demo.py              # Lightweight quickstart script
│   ├── transformer_from_scratch_demo.py
│   └── transformer_using_library_demo.py
├── tests/
│   └── test.py                      # Comprehensive test suite (pytest)
├── docs/
│   └── transformer-architecture.md  # Detailed architecture notes
├── requirements.txt
├── pytest.ini
├── LICENSE
└── README.md
```

## Module Flow: Config → Architecture → Model

```
GPTConfig (config.py)
    ↓
    Defines: vocab_size, block_size, n_layer, n_head, n_embd, dropout
    ↓
CausalSelfAttention (attention.py)
    ↓
    Multi-head scaled dot-product attention with causal mask
    ↓
MLP (mlp.py)
    ↓
    Position-wise feed-forward: n_embd → 4*n_embd → n_embd
    ↓
Block (block.py)
    ↓
    Pre-norm transformer block: (LayerNorm → Attn + residual) → (LayerNorm → MLP + residual)
    ↓
GPT2 (model.py)
    ├── Token Embedding (wte)
    ├── Positional Embedding (wpe)
    ├── Stack of n_layer Blocks
    ├── Final LayerNorm (ln_f)
    └── Output Projection (lm_head) [weights shared with wte]
    ↓
generate() (generation.py)
    ↓
    Auto-regressive text generation with temperature & top-k sampling
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Jaypatel2611/transformer-from-scratch.git
cd transformer-from-scratch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Minimal Example (5 minutes)

```bash
python examples/minimal_demo.py
```

This runs a forward pass on dummy tokens and prints output shapes and a generation sample.

### 2. Forward Pass with Loss

```python
import torch
from src import GPT2, GPTConfig

# Create a small config for testing
config = GPTConfig(
    vocab_size=256,
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=64
)

# Instantiate model
model = GPT2(config)

# Create dummy batch
batch_size, seq_len = 2, 32
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Forward pass with loss
logits, loss = model(input_ids, targets)
print(f"Logits shape: {logits.shape}")  # (batch_size, seq_len, vocab_size)
print(f"Loss: {loss.item():.4f}")
```

### 3. Text Generation

```python
import torch
from src import GPT2, GPTConfig, generate

config = GPTConfig(vocab_size=256, block_size=128)
model = GPT2(config)
model.eval()

# Start with a seed token
seed = torch.tensor([[100]])

# Generate 50 new tokens with sampling
with torch.no_grad():
    generated = generate(
        model,
        seed,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40
    )

print(f"Generated shape: {generated.shape}")  # (1, 51)
```

## How to Validate This Repository

### Run the Test Suite

```bash
# Install pytest if not already present
pip install pytest

# Run all tests
pytest tests/test.py -v

# Run specific test class
pytest tests/test.py::TestConfig -v

# Run with coverage (optional)
pip install pytest-cov
pytest tests/test.py --cov=src --cov-report=html
```

### What the Tests Validate

**Config Validation** (`TestConfig`)
- Default and custom configuration values
- Validation rule: `n_embd` must be divisible by `n_head`
- Prevents invalid configurations at instantiation

**Attention Layer** (`TestCausalSelfAttention`)
- Forward pass produces correct output shape
- Attention respects causality (no future token access)

**Block & Model Integration** (`TestBlock`, `TestGPT2`)
- Blocks can be instantiated and run forward passes
- Full model forward pass with and without targets
- Loss computation produces a scalar
- Output shapes are correct

**Generation** (`TestGeneration`)
- Model generates tokens without error
- Output sequence grows correctly
- Respects max_new_tokens parameter

### Sanity Check: Shape Validation

Run this quick script to validate that tensors flow through the pipeline correctly:

```python
import torch
from src import GPT2, GPTConfig

config = GPTConfig(vocab_size=256, block_size=64, n_layer=2, n_head=2, n_embd=32)
model = GPT2(config)

# Input: (batch=1, seq=16)
x = torch.randint(0, 256, (1, 16))
logits, loss = model(x)

# Output should be: (batch=1, seq=16, vocab=256)
assert logits.shape == (1, 16, 256), f"Expected (1, 16, 256), got {logits.shape}"
assert loss is None, "Loss should be None without targets"

print("✓ Shape validation passed")
```

## Known Limitations

This implementation is **educational and experimental**. Be aware of these constraints:

### Scope & Design

- **Educational tool**: Designed to teach transformer concepts, not to compete with production LLMs
- **Untrained weights**: No pretrained checkpoints included (weights initialize randomly)
- **Minimal generation**: `generate()` supports temperature and top-k sampling only (no beam search, nucleus sampling, or advanced decoding)
- **No tokenizer**: You must handle tokenization externally (the model works with token IDs only)
- **Small by default**: Provided configs are demonstration scale, not large-scale training configs
- **CPU/single-GPU only**: No distributed training support or multi-GPU optimizations

### Performance & Scale

- **No inference optimization**: Model runs standard PyTorch without quantization, pruning, or kernel optimizations
- **Causal masking overhead**: Full O(T²) attention, not sparse or efficient attention variants
- **No caching**: Generation naively re-computes the full sequence at each step (no KV-cache)
- **Educational MLP**: Feed-forward is standard dense layers (no gating, expert routing, etc.)

### Not Included

- Pre-trained weights or fine-tuning examples on real datasets
- Byte-pair encoding or other tokenization schemes
- Benchmarks or performance comparisons to other implementations
- Advanced features: LoRA, quantization, mixed precision, etc.
- Distributed training code or multi-GPU support

## Future Improvements

Potential enhancements for future versions:

- [ ] KV-cache for efficient generation
- [ ] Nucleus (top-p) sampling
- [ ] Flash Attention or other efficient attention implementations
- [ ] Gradient checkpointing for reduced memory usage
- [ ] Example training loop on a small dataset (e.g., text8, WikiText)
- [ ] Serialization/checkpoint saving utilities
- [ ] Inference optimization (torchscript export, etc.)
- [ ] Visualization utilities for attention patterns

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Give me 100 min, I will make Transformer click forever](https://www.youtube.com/watch?v=CfJ3Cxtlcps&t=598s) (Zachary Huang)
- [Give me 100 min, I will make Transformer click forever](https://github.com/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/transformer.md) (Zachary Huang, Github)

---

**Built with clarity, precision, and honest documentation in mind.**