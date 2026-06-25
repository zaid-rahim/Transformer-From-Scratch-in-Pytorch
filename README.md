# Transformer from Scratch — PyTorch

A full implementation of the Transformer architecture from the ground up in PyTorch, based on the original **"Attention Is All You Need"** paper (Vaswani et al., 2017). Every component is built as a standalone `nn.Module` with detailed markdown explanations covering the math and intuition behind each design decision.

## What's implemented

Every building block of the original Transformer is coded step by step:

| Step | Component | Description |
|---|---|---|
| 1 | `InputEmbedding` | Token IDs → dense vectors, scaled by √d_model |
| 2 | `PositionalEncoding` | Sinusoidal position signals added to embeddings |
| 3 | `LayerNormalization` | Custom layer norm with learnable α and β parameters |
| 4 | `FeedForwardNetwork` | Position-wise FFN: Linear → ReLU → Linear |
| 5 | `MultiHeadAttention` | Scaled dot-product attention across h parallel heads |
| 6 | `ResidualConnection` | Pre-norm residual: x + sublayer(LayerNorm(x)) |
| 7 | `EncoderBlock` | Self-attention + FFN with residual connections |
| 8 | `Encoder` | Stack of N encoder blocks + final layer norm |
| 9 | `DecoderBlock` | Masked self-attention + cross-attention + FFN |
| 10 | `Decoder` | Stack of N decoder blocks + final layer norm |
| 11 | `Transformer` | Full encoder-decoder assembly with projection layer |

## Architecture overview

```
Input tokens
     │
     ▼
InputEmbedding (× √d_model)
     │
     ▼
PositionalEncoding (sin/cos waves)
     │
     ▼
┌─────────────────────────┐
│     Encoder Block × N   │
│  ┌──────────────────┐   │
│  │  MultiHead       │   │
│  │  Self-Attention  │   │
│  └──────────────────┘   │
│  ResidualConnection      │
│  ┌──────────────────┐   │
│  │  FeedForward     │   │
│  │  Network         │   │
│  └──────────────────┘   │
│  ResidualConnection      │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│     Decoder Block × N   │
│  Masked Self-Attention   │
│  Cross-Attention         │
│  FeedForward Network     │
└─────────────────────────┘
     │
     ▼
Projection Layer → Output vocabulary
```

## Key concepts explained in the notebook

**Why scale embeddings by √d_model** — Positional encoding values range between -1 and +1. Without scaling, these fixed sine/cosine waves would overpower the learned embedding values. Multiplying by √d_model balances both signals so word identity stays dominant.

**Why sinusoidal positional encoding** — Attention is permutation-invariant: without positional encoding, `"cat sat"` and `"sat cat"` produce identical representations. Sine and cosine waves at different frequencies give every position a unique, continuous signature the model can learn from.

**Why LayerNorm instead of BatchNorm** — BatchNorm normalizes across the batch dimension, which breaks for variable-length sequences where padding differs per sample. LayerNorm normalizes across the feature dimension — one sample at a time — making it sequence-length agnostic.

**Why residual connections** — In deep networks, gradients shrink as they propagate back through many layers (vanishing gradient problem). Residual connections (`x + sublayer(x)`) create a direct gradient highway from the loss all the way to early layers.

**Multi-head attention reshape trick** — Rather than running h separate attention operations, the implementation reshapes `(batch, seq, d_model)` into `(batch, h, seq, d_k)` and runs attention in parallel across all heads with a single matrix operation.

## Multi-head attention — shape walkthrough

```
d_model = 512, h = 8 → d_k = 512 / 8 = 64

Input x:         (batch, seq, 512)
After W_Q/K/V:   (batch, seq, 512)
After reshape:   (batch, seq, 8, 64)
After transpose: (batch, 8, seq, 64)   ← each head sees (seq, 64)
Attention scores:(batch, 8, seq, seq)
After @ V:       (batch, 8, seq, 64)
After transpose: (batch, seq, 8, 64)
After reshape:   (batch, seq, 512)     ← back to original d_model
After W_O:       (batch, seq, 512)     ← final output
```

## Tech stack

Python 3.12 · PyTorch · Jupyter Notebook

## How to run

```bash
pip install torch jupyter
jupyter notebook Transformer_From_Scratch_in_Pytorch.ipynb
```

## References

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- Video: [Umar Jamil — Coding a Transformer from scratch on PyTorch](https://www.youtube.com/watch?v=ISNdQcPhsts)

## Context

This is Week 2 of a 12-week AI/LLM engineering roadmap. Built after implementing a neural network from scratch in NumPy and PyTorch in Week 1. Next: training the transformer on a real translation task and implementing the full data pipeline.
