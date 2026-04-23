"""
Minimal demonstration of the Transformer model.

This script shows how to:
1. Create a small transformer model with GPTConfig
2. Run a forward pass on dummy token IDs
3. Compute loss with targets
4. Generate text auto-regressively

No external datasets or complex setup required—just PyTorch.
Run with: python examples/minimal_demo.py
"""

import torch
from src import GPT2, GPTConfig, generate


def main():
    """Run minimal transformer demonstration."""
    
    print("=" * 60)
    print("Transformer From Scratch - Minimal Demo")
    print("=" * 60)
    
    # =========================================================================
    # 1. Create a configuration
    # =========================================================================
    print("\n[1] Creating GPTConfig...")
    config = GPTConfig(
        vocab_size=256,        # Small vocabulary
        block_size=128,        # Max sequence length
        n_layer=4,             # Number of transformer blocks
        n_head=4,              # Number of attention heads
        n_embd=64,             # Embedding dimension
        dropout=0.1
    )
    print(f"    vocab_size={config.vocab_size}")
    print(f"    block_size={config.block_size}")
    print(f"    n_layer={config.n_layer}")
    print(f"    n_head={config.n_head}")
    print(f"    n_embd={config.n_embd}")
    
    # =========================================================================
    # 2. Instantiate the model
    # =========================================================================
    print("\n[2] Instantiating GPT2 model...")
    model = GPT2(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    # =========================================================================
    # 3. Forward pass without targets (inference mode)
    # =========================================================================
    print("\n[3] Forward pass (without targets)...")
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(input_ids)
    
    print(f"    Input shape: {input_ids.shape}")
    print(f"    Output logits shape: {logits.shape}")
    print(f"    Loss: {loss}")  # Should be None since no targets
    
    # =========================================================================
    # 4. Forward pass with targets (training mode)
    # =========================================================================
    print("\n[4] Forward pass (with targets for loss)...")
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids, targets)
    print(f"    Input shape: {input_ids.shape}")
    print(f"    Targets shape: {targets.shape}")
    print(f"    Output logits shape: {logits.shape}")
    print(f"    Cross-entropy loss: {loss.item():.4f}")
    
    # =========================================================================
    # 5. Text generation (auto-regressive)
    # =========================================================================
    print("\n[5] Generating text (auto-regressive)...")
    model.eval()
    
    # Start with a single token (seed)
    seed_token = torch.tensor([[42]])  # One batch, one token
    
    print(f"    Seed token: {seed_token.item()}")
    print(f"    Generating 30 new tokens with temperature=0.8, top_k=50...")
    
    with torch.no_grad():
        generated = generate(
            model,
            seed_token,
            max_new_tokens=30,
            temperature=0.8,
            top_k=50
        )
    
    print(f"    Generated sequence shape: {generated.shape}")
    print(f"    Generated tokens: {generated.tolist()[0][:10]}... [truncated]")
    
    # =========================================================================
    # 6. Deterministic generation (low temperature)
    # =========================================================================
    print("\n[6] Generating text (deterministic, temperature=0.1)...")
    
    with torch.no_grad():
        generated_greedy = generate(
            model,
            seed_token,
            max_new_tokens=20,
            temperature=0.1,  # Low temperature = mostly argmax
            top_k=None
        )
    
    print(f"    Generated sequence shape: {generated_greedy.shape}")
    print(f"    Generated tokens: {generated_greedy.tolist()[0]}")
    
    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Demo complete! ✓")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  • Model accepts token IDs and outputs logits")
    print("  • Can compute loss for training (if targets provided)")
    print("  • Can generate sequences auto-regressively")
    print("  • Supports multiple sampling strategies (temperature, top-k)")
    print("\nNext steps:")
    print("  • Try modifying the config (e.g., n_layer, n_embd)")
    print("  • Look at examples/transformer_using_library_demo.py for more details")
    print("  • Run: pytest tests/test.py -v")


if __name__ == "__main__":
    main()
