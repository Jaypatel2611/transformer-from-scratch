import torch.nn as nn, torch, torch.nn.functional as F, math
from dataclasses import dataclass

# Word Embedding

vocab_size = 10
n_embd = 3 # The no of dimension in our "semantic space"

token_embedding_table = nn.Embedding(vocab_size, n_embd)

# The book itself is the .weight attri. Each row is a word's coords
print("Shape of our coordinate book: ", token_embedding_table.weight.shape)
print("Content of the book:") # (initially random coords)
print(token_embedding_table.weight)

# Parameter containing:
# tensor([[ 1.1618,  0.5857, -2.5029],
#         ..,
#         [ 0.5756, -1.1977,  0.7718]], requires_grad=True) 
# requires_grad=True : it means during training, the model will learn the best possible coords for each word.. for better pred

# Position Embedding

B, T, C = 2, 5, 3 # Batch, Time (sequence length) & Channel {n_embd}
vocab_size = 10
block_size = 8 # Our model's max sequence length

# ------- The Layers -------
position_embedding_table = nn.Embedding(block_size, C)

idx = torch.randint(0, vocab_size, (B, T)) # Shape: (2, 5)

# Step 1: Getting Word Embedding
tok_emb = token_embedding_table(idx) # Shape: (2, 5, 3)

# Step 2: Getting Positional Embedding
pos = torch.arange(0, T, dtype= torch.long) # Shape: (0, 5)=(5) => tensor([0, 1, 2, 3, 4])
pos_emb = position_embedding_table(pos)

# Step 3: Combining via addition
x = tok_emb + pos_emb

print("Shape of token Embedding: ", tok_emb.shape)
print("Shape of position Embedding: ", pos_emb.shape)
print("Shape of combined Embedding: ", x.shape) # It happens coz pytorch uses "Broadcasting"


# SELF ATTENTION

B, T, C = 1, 4, 2
n_head = 12

# A pretend input for "A crane ate fish"
X = torch.tensor([[[0.1, 0.1],
                 [1.0, 0.2],
                 [0.1, 0.9],
                 [0.8, 0.0]]]).float()

# Step 1: PROJECTING 'X' INTO Q, K & V
q_proj = nn.Linear(C, C, bias=False)
k_proj = nn.Linear(C, C, bias=False)
v_proj = nn.Linear(C, C, bias=False)

torch.manual_seed(42)
q_proj.weight.data = torch.randn(C, C)
k_proj.weight.data = torch.randn(C, C)
v_proj.weight.data = torch.randn(C, C)

q = q_proj(X)
k = k_proj(X)
v = v_proj(X)

# Step 2: CALCULATE THE TENSION SCORES ('Q @ K.T')
scores = q @ k.transpose(-2, -1) # Shape: (1, 4, 4)
print(scores.shape)
print(scores)

# Step 3 & 4: SCALE & SOFTMAX
d_k = k.size(-1)
scaled_scores = scores / math.sqrt(d_k)
attention_weights = F.softmax(scaled_scores, dim=-1) # softmax turn each row into weights that sum to 1
print('\n------ Attention Weights Output ------')
print(attention_weights)

# Step 5: AGGREGATE THE VALUES ('ATTENTION WEIGHTS @ V')

output = attention_weights @ v
print('\n------ Output (Context Aware Vectors) ------')
print("Shape of result: ", output.shape)
print(output)
# Transformed out input 'X' into an output of the exact same shape

# The Causal Mask

mask = torch.tril(torch.ones(scaled_scores.shape[1], scaled_scores.shape[1]))
masked_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

# Re-run Softmax

attention_weights = F.softmax(masked_scores, dim=-1)
print('\n------ Attention Weight Output (After Casual Mask) ------')
print(attention_weights)

# MULTI-HEAD ATTENTION

# ---- Dummy Q tensor with realistic weights
B, T, C = 1, 4, 768
q = torch.randn(B, T, C)
k = torch.randn(B, T, C)
v = torch.randn(B, T, C)

head_dim = int(C / n_head)

# Reshaping
# 1. Start with q: (B, T, C)
print("\nOriginal Q shape:", q.shape)
# 2. Reshape to add the n_head dimension. We "carve up" the last dimension
q_reshaped = q.view(B, T, n_head, head_dim)
print("Reshaped q:", q_reshaped.shape)
# 3. The Transpose Operation
q_final = q_reshaped.transpose(1, 2)
print("Final reshaped Q shape:", q_final.shape)

k_final = k.view(B, T, n_head, head_dim).transpose(1, 2)
v_final = v.view(B, T, n_head, head_dim).transpose(1, 2)

# Attention Calculation (happening 12 times at once)
# (B, nh, T, hd) @ (B, nh, hd, T) -> (B, nh, T, T)
scaled_scores = (q_final @ k_final.transpose(-2, -1)) / math.sqrt(head_dim)
attention_weights = F.softmax(scaled_scores, dim=-1)

# (B, nh, T, T) @ (B, nh, T, hd) -> (B, nh, T, hd)
output_per_head = attention_weights @ v_final
print("Shape of the output from each head:", output_per_head.shape)

# 3. Merging
# 1. Tranpose and reshape to merge the heads back together
merged_output = output_per_head.transpose(1, 2).contiguous().view(B, T, C)
print("Shape of Merged output:", merged_output.shape)

# 2. Pass through the final projection layer
c_proj = nn.Linear(C, C)
final_output = c_proj(merged_output)
print("Shape of Final Output:", final_output.shape)


# MLP[THINKING] LAYER
print("\n\n\n---------- MLP LAYER ----------")
# PROBLEM: DEMYSTIFY `NN.LINEAR` [Project a vector of size 2 upto a vector of size 4]
C_in = 2
C_out = 4
linear_layer = nn.Linear(C_in, C_out)
# Manually Setting weights and biases
linear_layer.weight.data = torch.tensor([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]])
linear_layer.bias.data = torch.tensor([1., 1., -1., -1.])

# Our input vector
input_vector = torch.tensor([0.5, -0.5])
output_vector = linear_layer(input_vector)
print("Output Layer:", output_vector)

# 1. The EXPANSION Layer
# A Token's Journey Through The MLP
# Input: A single token vector with `C=2`. MLP will expand to `4 * C = 8`. Shape: (B, T, C) -> (1, 1, 2)

x = torch.tensor([[[0.5, -0.5]]])
fc = nn.Linear(2, 8)
torch.manual_seed(1337)
fc.weight.data = torch.randn(8, 2)
fc.bias.data = torch.randn(8)
x_expanded = fc(x)

print("\n------- After EXPANSION Layer -------")
print("Shape:", x_expanded.shape)
print("Values:", x_expanded)

# 2. The ACTIVATION Layer(`GELU`)
x_activated = F.gelu(x_expanded)

print("\n------- After GELU Layer -------")
print("Shape:", x_activated.shape)
print("Values:", x_activated)

# 3. The CONTRACTION Layer(`PROJ`)

proj = nn.Linear(8, 2)
torch.manual_seed(42)
fc.weight.data = torch.randn(2, 8)
fc.bias.data = torch.randn(2)
x_projected = proj(x_activated)

print("\n------- After CONTRACTION Layer -------")
print("Shape:", x_projected.shape)
print("Values:", x_projected)

# 4. The DROPOUT Layer

drop = nn.Dropout(0.1)
final_output = drop(x_projected)

print("\n------- After DROPOUT Layer -------")
print("Shape:", final_output.shape)
print("Values:", final_output)

# RESIDUAL CONNECTION

print("\n\n\n------- RESIDUAL CONNECTION -------")
# Input vector for a single token 'X'
x_initial = torch.tensor([[[0.2, 0.1, 0.3, 0.4]]])
print("\nOriginal X:", x_initial)

# The output of `self.attn(self.ln_1(x))`
# This is the "adjustment" to be made
attention_output = torch.tensor([[[0.1, -0.1, 0.2, -0.2]]])
print("\nOutput from attention(the `adjustment`):", attention_output)

# The residual connection: x = x + ...
x_after_attn = x_initial + attention_output
print("\nValues after the first residual connection:", x_after_attn)

# STABILIZER
print("\n\n\n------- STABILIZER -------")

# 1. Calculating Mean and Std
x_token = torch.tensor([[[0.3, -0.2, 0.8, 0.5]]])
mean = x_token.mean(dim=-1, keepdim=True)
std = x_token.std(dim=-1, keepdim=True)

print("\nInput to LayerNorm (X):", x_token)
print("Mean of the input:", mean)
print("Standard Deviation of the input:", std)

# 2. Normalises the vector

epilson = 1e-5
x_hat = (x_token - mean)/(torch.sqrt(std**2 + epilson))

print("Normalized vector (x_hat):\n", x_hat.data.round(decimals=2))
print(f"\nMean of x_hat: {x_hat.mean().item():.2f}")
print(f"Std Dev of x_hat: {x_hat.std().item():.2f}")

C_ = 4
ln = nn.LayerNorm(C_)

print("\n------ Initial Parameters ------")
print("LayerNorm.weight (gamma) initial:\n", ln.weight.data)
print("LayerNorm.bias (beta) initial:\n", ln.bias.data)

"""
    The above output is the result of the layerNorm method of pytorch which initialises weight as all 1's and bias as all 0's
    But since multiplying with `1` and adding `0` doesn't do anything, so we initialises `gamma` & `beta` with some random values
"""

# 3. Applying Learnable Parameter

gamma = torch.tensor([1.5, 1.0, 1.0, 1.0])
beta = torch.tensor([0.5, 0.0, 0.0, 0.0])
y = gamma * x_hat + beta

print("\n---- After Applying Learnable Parameter ----")
print("Final Output:\n", y.round(decimals=2))
