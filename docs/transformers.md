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

![Model Hyperparameters](../Resources/Screenshot%202026-04-04%20200217.png)

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

![Self-Attention Overview](../Resources/Screenshot%202026-04-04%20214812.png)


#### Self-Attention Steps: Score → Normalize → Aggregate

We use a simple 2D space:
- Dimension 1: "Is it an Animal?"
- Dimension 2: "Is it a Machine?"

![2D Semantic Space](../Resources/Screenshot%202026-04-04%20215121.png)

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

![Raw Attention Scores](../Resources/Screenshot%202026-04-05%20004501.png)


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

![Multi-Head Attention](../Resources/Screenshot%202026-04-09%20222618.png)

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

![Residual Connection — Express Lane](../Resources/Screenshot%202026-04-10%20223631.png)

The output of the attention sub-layer is just an *update* to the original vector. The tensor shape remains unchanged — a critical property.

![Residual Connection — Shape Preservation](../Resources/Screenshot%202026-04-10%20225000.png)

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

![Layer Normalization](../Resources/Screenshot%202026-04-11%20213603.png)

We use **Pre-Norm** (normalize before the sub-layer), because it is more robust during training.

---

### The Data's Journey Through a Block

![Block Forward Pass](../Resources/Screenshot%202026-04-11%20214032.png)

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

![Depth vs Width](../Resources/Screenshot%202026-04-11%20214932.png)

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

![Full GPT-2 Architecture](../Resources/Screenshot%202026-04-11%20223807.png)

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

![Output Layer Projection](../Resources/Screenshot%202026-04-12%20122823.png)

---

#### Why Make T Predictions — Isn't It Wasteful?

No, because we use all T predictions in two ways:

**1. Training:**
The model learns to predict the next token at every position simultaneously in a single forward pass.

```
Input:  "A crane ate fish"  (T=4)
Learns:
  Given "A"           → predict "crane"
  Given "A crane"     → predict "ate"
  Given "A crane ate" → predict "fish"

output[:, 0, :] → prediction from "A"         → compare to target "crane"
output[:, 1, :] → prediction from "A crane"   → compare to target "ate"
output[:, 2, :] → prediction from "A crane ate" → compare to target "fish"
```

This works because the causal mask ensures prediction at position `t` only uses tokens `0` through `t`.

**2. Inference (Generation):**
```
Input:  "A crane ate"  (T=3)
Output: shape (1, 3, vocab_size)
We discard all but the last: output[:, -1, :]
```

This is a deliberate trade-off — the architecture is optimized for massive parallel computation during training, and we reuse the same architecture for inference even if only the final prediction is needed.

---

### Weight Tying

```python
self.lm_head.weight = self.wte.weight
```

This single line does not copy the matrix — it makes `lm_head.weight` point to the *exact same tensor object in memory* as `wte.weight`. In Python, this is a reference assignment, not a value copy. You can verify this:

```python
# Both point to the same object — same id, same data_ptr
assert model.lm_head.weight is model.wte.weight          # True: same Python object
assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()  # True: same memory
```

Because they share the same underlying storage, any gradient update that backpropagation applies to `wte.weight` is automatically reflected in `lm_head.weight` — there is no separate update step needed.

![Weight Tying](../Resources/Screenshot%202026-04-12%20125349.png)

Their functions are perfectly symmetric — these two operations are *inverses of each other*.

**Benefits:**

1. Massive Parameter Reduction — this matrix has `50,257 × 768 ≈ 38.5 million` parameters. Weight tying eliminates the need for a second separate matrix of the same size, saving ~40 million parameters with one line of code.
2. Improved Performance — acts as a form of regularization, reducing overfitting and leading to more stable models.

---

## III. Loss Calculation

The loss quantifies how wrong the model's predictions are — it is the signal used to update every single weight in the network.

**Training Goal:** Next-token prediction.

For each chunk of training data, `idx` is the input sequence and `targets` is the same sequence shifted one position to the left:

```
idx:     "A",     "crane", "ate"
targets: "crane", "ate",   "fish"
```

**Forward Method of the GPT2 class:**

```python
def forward(self, idx, targets=None):
    B, T = idx.size()
    assert T <= self.config.block_size, "Sequence length exceeds block size."

    pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

    x = self.wte(idx) + self.wpe(pos)   # 1. Get embeddings
    x = self.drop(x)
    for block in self.h:                 # 2. Process through blocks
        x = block(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)            # 3. Get logits

    loss = None
    if targets is not None:
        loss = F.cross_entropy(          # 4. Calculate loss
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

    return logits, loss
```

**Step-by-step:**

1. Get Embeddings — create positional + token embeddings for the initial `(B, T, C)` tensor:
   ```python
   x = self.wte(idx) + self.wpe(pos)
   ```

2. Process through Blocks — the tensor flows through the stack, becoming more context-aware:
   ```python
   for block in self.h:
       x = block(x)
   ```

3. Get Logits — project final vectors into a `(B, T, vocab_size)` tensor:
   ```python
   logits = self.lm_head(self.ln_f(x))
   ```

4. Calculate Loss — compare predictions (`logits`) to ground truth (`targets`):
   ```python
   loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
   ```

`F.cross_entropy()` expects 2D input. `.view()` squashes the `B` and `T` dimensions together to get the right shape.

![Shape Transformation for Loss](../Resources/Screenshot%202026-04-12%20183021.png)

`cross_entropy` calculates the loss for each of the `B×T` predictions individually, then averages them into a single scalar `LOSS` value — making training stable.

---

## IV. The Complete Training Loop (6 Stages)

Let's say `block_size` (`T`) = 4.

**Stage 1 — Raw Data Stream (Token IDs):**
```
[5, 12, 8, 21, 6, 33, 9, 4, 15, 7, 2, ...]
```

**Stage 2 — Creating (`idx`, `targets`) Pairs:**
```
Sample 1:  idx = [5, 12, 8, 21]   targets = [12, 8, 21, 6]
Sample 2:  idx = [6, 33, 9, 4]    targets = [33, 9, 4, 15]
```

With batch size `B=2`, both samples are processed in parallel on the GPU:
```python
idx     = [[5, 12, 8, 21], [6, 33, 9, 4]]   # shape (2, 4)
targets = [[12, 8, 21, 6], [33, 9, 4, 15]]  # shape (2, 4)
```

![Training Loop Batch](../Resources/Screenshot%202026-04-12%20185004.png)

**Stage 3 — Forward Pass:**
Feed `idx` through the full model to get `logits` of shape `(B, T, vocab_size)` and compute the `loss` scalar by comparing against `targets`.
```python
logits, loss = model(idx, targets)
```

**Stage 4 — Zero the Gradients:**
Before computing new gradients, clear any gradients left over from the previous step. Failing to do this accumulates gradients across steps, which corrupts the update.
```python
optimizer.zero_grad()
```

**Stage 5 — Backward Pass:**
PyTorch walks backward through the computation graph and computes `∂loss/∂w` for every learnable parameter `w` in the model.
```python
loss.backward()
```

**Stage 6 — Weight Update:**
The optimizer uses the computed gradients to nudge every weight in the direction that reduces the loss.
```python
optimizer.step()
```

This entire 6-stage process is one training step. For each batch there is exactly one backward pass and one weight update, driven by the average performance across all token predictions in the batch.

---

## V. Auto-Regressive Generation

### The Core Generation Loop

1. **Predict** — feed the current sequence in, get a probability distribution over the next token.
2. **Sample** — pick a token from that distribution.
3. **Append** — add the new token to the end of the sequence.
4. **Repeat** — go back to step 1 with the longer sequence.

**Example:**

```
Iteration 1:
  idx = [5, 12]
  Predict → 40% "ate", 30% "lifted"
  Sample  → pick "ate" (ID 8)
  Append  → idx = [5, 12, 8]

Iteration 2:
  idx = [5, 12, 8]
  Predict → 90% "fish" (model now sees "ate")
  Sample  → pick "fish" (ID 21)
  Append  → idx = [5, 12, 8, 21]
```

---

### The `generate` Method — In Detail

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 1. Crop context to block_size
        idx_cond = idx[:, -self.config.block_size:]

        # 2. Get logits from the last position only
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        # 3. Optional Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 4. Sample the next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 5. Append and continue
        idx = torch.cat((idx, next_token), dim=1)

    return idx
```

**Key details:**

1. `@torch.no_grad()` — disables gradient computation during inference, saving memory and computation.

2. Context Cropping — if the sequence grows longer than `block_size`, we crop to the last `block_size` tokens. This is the model's "memory window."

3. Getting the Final Logits — `logits[:, -1, :]` discards all predictions except the last one. Dividing by `temperature` controls output "creativity."

4. Sampling — `F.softmax` converts logits to a probability distribution; `torch.multinomial` performs the weighted random draw.

5. Appending — `torch.cat` concatenates the new token to the sequence for the next iteration.

![Generation Sampling Controls](../Resources/Screenshot%202026-04-12%20192433.png)

Randomly sampling from the full distribution can sometimes produce strange or nonsensical words. Two knobs help control this:

#### Temperature

`temperature` is a scalar that divides the logits *before* softmax:

```python
logits = logits[:, -1, :] / max(temperature, 1e-8)
```

| Temperature | Effect |
|-------------|--------|
| `< 1.0` (e.g. `0.5`) | Sharpens the distribution — high-probability tokens become even more dominant. Output is more focused and predictable ("colder"). |
| `= 1.0` | No change — the raw model distribution is used. |
| `> 1.0` (e.g. `1.5`) | Flattens the distribution — low-probability tokens get a bigger chance. Output is more random and creative ("hotter"). |

#### Top-K Filtering

`top_k` restricts sampling to only the `k` most likely tokens, setting all others to `-inf` before softmax:

```python
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
```

| top_k | Effect |
|-------|--------|
| `None` | Sample from the full vocabulary (can pick very unlikely words) |
| `1` | Always pick the single most likely token (greedy / deterministic) |
| `50` | Sample only from the top 50 candidates — a good balance of quality and variety |

Temperature and top-k are typically used together: top-k narrows the candidate pool, and temperature controls how evenly you sample within that pool.

---

## 🚧 Common Problems & Solutions

| Problem | Solution |
|---------|----------|
| Using token IDs like `{'red': 1, 'orange': 2}` carries no semantic meaning | Use vector embeddings via `nn.Embedding` |
| Same word (`"bank"`) has the same embedding regardless of context | Use self-attention to make embeddings context-aware |
| Model can "see" future tokens during training | Apply a causal mask to block future positions |
| Training instability due to shifting activation distributions | Use Layer Normalization |
| Vanishing gradients in deep networks | Use residual (skip) connections |
| Single attention head cannot capture multiple relationship types | Use Multi-Head Attention |
| Overfitting during training | Apply Dropout |
| Output logits shape mismatch for loss calculation | Use `.view()` to reshape before passing to `F.cross_entropy()` |
| Sequence length exceeds model's context window during generation | Crop input with `idx[:, -block_size:]` |
| `LayerNorm` initialized with `γ=1, β=0` does nothing initially | The model learns optimal `γ` and `β` values during training |
| Confused whether `lm_head.weight = wte.weight` copies or shares the tensor | It is a reference assignment — both point to the same memory; no copy is made |
| `torch.cat(idx, next_token, dim=-1)` raises a `TypeError` | Wrap tensors in a tuple: `torch.cat((idx, next_token), dim=1)` |
| `def forword(...)` typo causes `model.forward()` to fail | Rename to `def forward(...)` |
| Output is repetitive or nonsensical during generation | Lower `temperature` (< 1.0) or reduce `top_k` to constrain sampling |
