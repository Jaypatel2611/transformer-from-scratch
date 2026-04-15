# Transformer Architecture — From Scratch

## 🗺️ Roadmap

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | Tokenization | Convert raw text into token IDs |
| 2 | Embedding | Convert tokens into vectors |
| 3 | Positional Encoding | Add position awareness |
| 4 | Transformer Block | Core processing unit |
| 4.1 | Multi-Head Attention | Context understanding |
| 4.2 | Layer Normalization | Stabilization |
| 4.3 | Feed Forward (MLP) | Token-wise processing |
| 5 | Output Layer | Generate logits |
| 6 | Loss Calculation | Measure prediction error |
| 7 | Training Loop | Update weights |
| 8 | Generation | Auto-regressive prediction |

These are the knobs you can turn to change the size and power of the model.

![Model Hyperparameters](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-04%20200217.png)

---

## I. Embedding: Turning Words into Vectors

### i. Word Token Embedding

**Bad Way (Token IDs):** Assigning integers like `{'red': 1, 'orange': 2, 'blue': 8}` carries no semantic meaning — the computer can't infer that orange sits between red and blue on a color chart.

**Good Way (Vectors):** Represent each word as a coordinate in a semantic space:
- `red`    → `[0.9, 0.1]`
- `orange` → `[0.8, 0.2]`
- `blue`   → `[0.1, 0.9]`

`Line 6` — `nn.Embedding` converts words into meaningful vectors. Think of it as a giant lookup table.

**Flaw:** This embedding is *static* and *context-free*.

> Example:
> 1. "I sat at the river *bank*"
> 2. "I visited the *bank* today"
>
> The vector for `bank` is identical in both sentences.

**Solution:** Positional Embedding — learn a unique vector for each position.

---

### ii. Positional Token Embedding

Implemented using `nn.Embedding` — still a giant lookup table, but now for *position*.

- **Same Space:** Word and position vectors exist in the same 768-dimensional space.
- **Smart Addition:** Adding them creates a unique point in that space.

> `'the'` at position `n` ≠ `'the'` at position `m`

`Line 28` — `nn.Embedding` for position: a giant lookup table for positional information.

**Edge case — if sequence length < `block_size`:**

```
Case: T=5, block_size=8
Solution: torch.arange(0, T)  →  generates positions only for the actual sequence length
```

---

## II. Transformer Block

### "COMMUNICATION (ATTENTION) LAYER" — Tokens Interact with Each Other

Every word has a vector. Every position has a vector. But we still have the *"river bank"* vs *"money bank"* problem — the model has no way to adjust a word's meaning based on its context.

This is the **heart of the Transformer**.

> Example:
> 1. "The crane ate a fish"
> 2. "The crane lifted the machine"
>
> The starting vectors for both sentences are identical, so we need **Self-Attention**.

**Attention Formula:**

```math
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V
```

| Symbol | Role | Description |
|--------|------|-------------|
| `Q` (Query) | "search query" | What are we looking for? |
| `K` (Key) | "label / keyword" | What does this token represent? |
| `V` (Value) | "payload" | What information does it offer? |

**Self-Attention** is a communication mechanism that allows every word to dynamically pull in information from its context (other words in the sentence) to refine its own meaning.

![Self-Attention Overview](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-04%20214812.png)


#### Self-Attention Steps: Score → Normalize → Aggregate

We use a simple 2D space:
- Dimension 1: "Is it an Animal?"
- Dimension 2: "Is it a Machine?"

![2D Semantic Space](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-04%20215121.png)

**Sentence 1: "A crane ate fish"**

| Token | Query | Key | Value |
|-------|-------|-----|-------|
| crane | `[0.7, 0.7]` | `[0.7, 0.7]` | `[0.5, 0.5]` |
| ate   | — | `[0.9, 0.1]` | `[0.9, 0.1]` |
| fish  | — | `[0.8, 0.2]` | `[0.8, 0.2]` |

Step 1 — Scoring (`Q · K^T`): "crane" probes every word
```
Score(crane → crane): [0.7, 0.7] · [0.7, 0.7] = 0.98  (high self-affinity)
Score(crane → ate):   [0.7, 0.7] · [0.9, 0.1] = 0.70  (high match)
Score(crane → fish):  [0.7, 0.7] · [0.8, 0.2] = 0.70  (high match)
```

Step 2 — Normalizing (`softmax`): convert scores to percentages
```
Raw Scores: [0.98, 0.70, 0.70]  →  Attention Weights: [0.4, 0.3, 0.3]
"CRANE" will listen:  40% to itself,  30% to "ate",  30% to "fish"
```

Step 3 — Aggregating:
```
new_vector(crane) = 0.4 × [0.5, 0.5] + 0.3 × [0.9, 0.1] + 0.3 × [0.8, 0.2]
                 = [0.71, 0.29]  →  heavily skewed toward Dimension 1 (Animal)
```

---

**Sentence 2: "A crane lifted steel"**

| Token | Query | Key | Value |
|-------|-------|-----|-------|
| crane  | `[0.7, 0.7]` | `[0.7, 0.7]` | `[0.5, 0.5]` |
| lifted | — | `[0.1, 0.9]` | `[0.1, 0.9]` |
| steel  | — | `[0.2, 0.8]` | `[0.2, 0.8]` |

Step 1 — Scoring:
```
Score(crane → crane):  [0.7, 0.7] · [0.7, 0.7] = 0.98
Score(crane → lifted): [0.7, 0.7] · [0.1, 0.9] = 0.70
Score(crane → steel):  [0.7, 0.7] · [0.2, 0.8] = 0.70
```

Step 2 — Normalizing:
```
Raw Scores: [0.98, 0.70, 0.70]  →  Attention Weights: [0.4, 0.3, 0.3]
"CRANE" will listen:  40% to itself,  30% to "lifted",  30% to "steel"
```

Step 3 — Aggregating:
```
new_vector(crane) = 0.4 × [0.5, 0.5] + 0.3 × [0.1, 0.9] + 0.3 × [0.2, 0.8]
                 = [0.29, 0.71]  →  heavily skewed toward Dimension 2 (Machine)
```

The exact same initial "crane" vector has been transformed into two completely different, **context-aware** vectors.

---

`Lines 74 & 75` — Raw attention scores output:

```python
torch.Size([4, 4])
tensor([[ 0.0012,  0.0427, -0.0295,  0.0403],  # 'A'     scores for ['A', 'crane', 'ate', 'fish']
        [-0.0034,  0.1632, -0.2006,  0.1700],  # 'crane' scores for ['A', 'crane', 'ate', 'fish']
        [ 0.0166,  0.3065, -0.1234,  0.2732],  # 'ate'   scores for ['A', 'crane', 'ate', 'fish']
        [-0.0058,  0.0778, -0.1417,  0.0894]], grad_fn=<MmBackward0>)
```

![Raw Attention Scores](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-05%20004501.png)


---

### The Causal Mask

There is still a massive flaw: **during token generation, tokens can see into the future.**

> Example: `'crane'` should only have scores for itself and previous tokens. During training we know the next word, but during inference we must predict it — so we can't allow future tokens to influence the current one.

We want to build an **auto-regressive model** — a model that generates one token at a time.

**The Causal Mask** modifies the attention scores by setting future positions to negative infinity *before* the softmax:

```math
\text{softmax}(\tilde{x}_i) = \frac{e^{\tilde{x}_i}}{\sum e^{\tilde{x}_i}}
```

Since `e^{-∞} = 0`, those positions contribute nothing after softmax.

**Implementation:**

```python
# Step 1: Create a lower-triangular mask matching the score shape
mask = torch.tril(torch.ones(scaled_scores.shape[1], scaled_scores.shape[1]))

# Step 2: Mask out future positions with -inf
masked_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

# Step 3: Re-apply softmax
attention_weights = F.softmax(masked_scores, dim=-1)
```

**Output after masking:**

```python
tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
         [0.4706, 0.5294, 0.0000, 0.0000],
         [0.3192, 0.3918, 0.2891, 0.0000],
         [0.2476, 0.2627, 0.2249, 0.2648]]], grad_fn=<SoftmaxBackward0>)
```

You now have a rich attention tensor with no future-leakage.

---

#### Registering the Mask as a Buffer

When implementing this inside `nn.Module`, the mask is registered as a **buffer**:

```python
self.register_buffer(
    "bias",
    torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1, 1, config.block_size, config.block_size)
)
```

A buffer is:
- Part of the model's state (like weights), saved with `state_dict`
- Moved to GPU automatically with `.to(device)`
- Not a parameter — the optimizer does not update it

---

### Multi-Head Attention

A single attention head is like one person trying to do everything at once — tracking grammar, meaning, and long-range context simultaneously.

**Multi-Head Attention** runs several attention heads in parallel, each specializing in a different relationship:

| Head | Specialization |
|------|---------------|
| Head 1 | Syntax (verb-object relationships) |
| Head 2 | Semantics (relative meaning) |
| Head 3 | Pronoun reference tracking |
| ... | ... |

**Dimensions:**
- Embedding dimension `C = 768`
- Number of heads `n_head = 12`
- Dimensions per head `head_dim = C / n_head = 64`

#### Steps

**1. Split**

```python
# i. Reshape to carve up the last dimension
q_reshaped = q.view(B, T, n_head, head_dim)   # Shape: (1, 4, 12, 64)

# ii. Transpose to bring n_head forward for parallel processing
q_final = q_reshaped.transpose(1, 2)           # Shape: (1, 12, 4, 64)
```

PyTorch's broadcasting treats `n_head` as a new "batch" dimension — all 12 attention calculations happen independently and in parallel.

**2. Apply**

```python
# Scaled dot-product attention across all heads at once
# (B, nh, T, hd) @ (B, nh, hd, T) -> (B, nh, T, T)
scaled_scores = (q_final @ k_final.transpose(-2, -1)) / math.sqrt(head_dim)
attention_weights = F.softmax(scaled_scores, dim=-1)

# (B, nh, T, T) @ (B, nh, T, hd) -> (B, nh, T, hd)
output_per_head = attention_weights @ v_final
```

**3. Merge**

```python
# i. Transpose and reshape to merge heads back together
merged_output = output_per_head.transpose(1, 2).contiguous().view(B, T, C)

# ii. Pass through the final projection layer
c_proj = nn.Linear(C, C)
final_output = c_proj(merged_output)
```

![Multi-Head Attention](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-09%20222618.png)

Each token vector now contains the combined, context-aware information from all 12 attention heads.

---

### "THINKING (MLP) LAYER" — Per-Token Processing

The MLP processes each token's information independently.

**Standard MLP Architecture:**

| Step | Layer | Operation |
|------|-------|-----------|
| 1 | Expansion (`fc`) | `n_embd` → `4 × n_embd` |
| 2 | Non-Linearity (`gelu`) | Activation function |
| 3 | Contraction (`proj`) | `4 × n_embd` → `n_embd` |
| 4 | Dropout (`drop`) | Regularization |

`nn.Linear` computes: `output = input @ W^T + b`

#### Demystifying `nn.Linear`

Projecting a vector of size 2 up to size 4 (`C_in=2`, `C_out=4`):

```python
linear_layer = nn.Linear(C_in, C_out)
# Learnable parameters:
#   Weights (.weight): shape (C_out, C_in) = (4, 2)
#   Biases  (.bias):   shape (C_out,)      = (4,)
```

#### A Token's Journey Through the MLP

```python
x = torch.tensor([[[0.5, -0.5]]])  # Input: shape (1, 1, 2)

# Step 1: Expansion (C=2 → 4*C=8)
x_expanded = fc(x)

# Step 2: Activation
x_activated = F.gelu(x_expanded)

# Step 3: Contraction (8 → 2)
x_projected = proj(x_activated)

# Step 4: Dropout (regularization during training; no-op during inference)
drop = nn.Dropout(0.1)
final_output = drop(x_projected)
```

```
Input:  [[[0.5, -0.5]]]
         ↓ MLP THINKING LAYER
Output: [[[0.4118, 0.1168]]]
```

The MLP transforms the input vectors while **preserving their shape** `(B, T, C)`.

---

### Residual Connection — The Gradient Highway

```python
x = x + self.attn(self.ln_1(x))
```

This `x = x + ...` pattern is a **residual** (or skip) connection.

**The Problem — Vanishing Gradients:**
In a very deep network, the learning signal must travel backward from the final output all the way to the first layer. Like a long game of telephone, the signal gets a little distorted at each step until it vanishes entirely.

**The Solution:**
The residual connection creates a shortcut — an "express lane" — that lets gradients flow directly through the addition operator, bypassing the transformations inside the attention layer.

![Residual Connection — Express Lane](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-10%20223631.png)

The output of the attention sub-layer is just an *update* to the original vector. The tensor shape remains unchanged — a critical property.

![Residual Connection — Shape Preservation](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-10%20225000.png)

This express lane allows us to build deeper, more powerful models. But a highway with no rules leads to chaos.

---

### Layer Normalization — The Stabilizer

**The Problem — Internal Covariate Shift:**
As data flows through a deep network, the distribution of activations at each layer shifts constantly during training. The mean and variance of inputs to a given layer can vary wildly from one batch to the next. This makes training unstable — each layer is constantly chasing a moving target.

**The Solution — Layer Normalization:**
For each individual token vector in our `(B, T, C)` tensor, Layer Norm performs the following steps independently:

1. Calculate the mean `μ` and variance `σ²` across the `C` (embedding) dimension.
2. Normalize the vector:

```math
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
```

3. Apply learnable parameters:

```math
y = \gamma \cdot \hat{x} + \beta
```

Where `γ` (gain) and `β` (bias) allow the model to learn the optimal scale and shift for the next layer — after forcing the distribution to a standard normal (mean 0, std 1).

![Layer Normalization](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-11%20213603.png)

We use **Pre-Norm** (normalize before the sub-layer), because it is more robust during training.

---

### The Data's Journey Through a Block

![Block Forward Pass](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-11%20214032.png)

The logic for each sub-layer is identical and elegantly simple: **Normalize → Process → Add**.

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))  # Attention sub-layer
    x = x + self.mlp(self.ln_2(x))   # MLP sub-layer
    return x
```

The most critical property of this block is that the output shape is identical to the input shape `(B, T, C)`. This is what makes the block **stackable**.

---

### Stacking Transformer Blocks

A single block performs one round of "Communication" and "Thinking." Complex language understanding requires many such rounds — a *symposium* of meetings:

| Meeting | Block | What Happens |
|---------|-------|-------------|
| 1 | Block 0 | Raw token embeddings are discussed; each token gains a first-level understanding |
| 2 | Block 1 | The team reconvenes with refined understanding; higher-level concepts emerge |
| ... | ... | ... |
| 12 | Block 11 | Deep, nuanced understanding — from basic syntax to rich semantic meaning |

The output of one block becomes the input for the next, building a hierarchical understanding of the text.

#### Depth vs. Width

![Depth vs Width](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-11%20214932.png)

#### Implementing Depth with `nn.ModuleList`

```python
# From the GPT2 class __init__ method
self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
```

- `[Block(config) for _ in range(config.n_layer)]` creates `n_layer` (e.g., 12) *separate* Block instances.
- `nn.ModuleList` registers all 12 blocks so PyTorch can track their parameters.

**Are weights shared between blocks?**
No. Each call to `Block(config)` creates a new object with its own unique weights and biases for `attn` and `mlp`. This is essential — the skills needed for the first meeting (processing raw data) differ from those needed for the last (refining abstract representations). Each block specializes in its stage of the pipeline.

---

### The Full Model Architecture

![Full GPT-2 Architecture](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-11%20223807.png)

```python
def __init__(self, config):
    super().__init__()
    self.config = config

    # Part 1: Input Layers
    self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Word Token Embedding
    self.wpe = nn.Embedding(config.block_size, config.n_embd)  # Positional Embedding
    self.drop = nn.Dropout(config.dropout)

    # Part 2: Core Processing Layers
    self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

    # Part 3: Output Layers
    self.ln_f = nn.LayerNorm(config.n_embd)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.lm_head.weight = self.wte.weight  # Weight tying
```

**Part 1 — Input Layer:**
- `self.wte`: Word Token Embedding
- `self.wpe`: Positional Token Embedding
- `self.drop`: Dropout applied to the sum of embeddings (prevents overfitting)

**Part 2 — Core Processing:**
- `self.h`: The heart of the model — `n_layer` independent Block instances flowing data sequentially, each adding more contextual refinement.

**Part 3 — Output Layer:**
- `self.ln_f`: Final Layer Norm — stabilizes the output before projection.
- `self.lm_head`: Projects from internal vector space (`C` dims) into vocabulary space (`vocab_size` dims), applied in parallel to every token position along `T`.
- `self.lm_head.weight`: Tied to `self.wte.weight` (see Weight Tying below).

![Output Layer Projection](https://raw.githubusercontent.com/jaypatel2611/transformer-from-scratch/main/Resources/Screenshot%202026-04-12%20122823.png)
