"""
Demonstration of GPT-style language model using the transformer library.

This script shows how to:
1. Create a GPT-2 style model
2. Perform a forward pass with the model
3. Generate text using the model

Run with: python -m examples.transformer_using_library_demo
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.using_library import GPTConfig, GPT2, generate


def main():
    """Run the language model demo."""
    
    print("\n" + "="*60)
    print("GPT-2 Style Language Model Demo")
    print("="*60)
    
    # Step 1: Create a configuration
    print("\n[Step 1] Creating model configuration...")
    config = GPTConfig(
        vocab_size=256,      # Small vocabulary for demo
        block_size=128,      # Max context window
        n_layer=4,           # Number of transformer blocks
        n_head=4,            # Number of attention heads
        n_embd=64,           # Embedding dimension
        dropout=0.1
    )
    print(f"Config: vocab_size={config.vocab_size}, block_size={config.block_size}, "
          f"n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")
    
    # Step 2: Instantiate the model
    print("\n[Step 2] Instantiating GPT-2 model...")
    model = GPT2(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Step 3: Forward pass with random data
    print("\n[Step 3] Forward pass with random batch...")
    batch_size, seq_len = 2, 32 # B, T
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(idx, targets)
    print(f"  Input shape: {idx.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Step 4: Generation
    print("\n[Step 4] Generating text...")
    model.eval()
    
    # Generate with fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Start with a single token [100]
    initial_token = torch.tensor([[100]], dtype=torch.long)
    
    # Generate 50 tokens
    generated = generate(
        model,
        initial_token,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None
    )
    
    print(f"  Initial token: {initial_token.tolist()}")
    print(f"  Generated sequence length: {generated.shape[1]}")
    print(f"  Generated tokens: {generated[0].tolist()}")
    
    # Step 5: Generation with temperature
    print("\n[Step 5] Generating with different temperatures...")
    torch.manual_seed(42)
    
    for temp in [0.5, 1.0, 2.0]:
        generated = generate(
            model,
            initial_token,
            max_new_tokens=20,
            temperature=temp,
            top_k=None
        )
        print(f"  Temperature {temp}: {generated[0, :25].tolist()}")
    
    # Step 6: Top-k sampling
    print("\n[Step 6] Generating with top-k sampling...")
    torch.manual_seed(42)
    
    generated = generate(
        model,
        initial_token,
        max_new_tokens=30,
        temperature=1.0,
        top_k=20
    )
    print(f"  With top-k=20: {generated[0].tolist()[:35]}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
