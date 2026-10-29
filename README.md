# GPT From Scratch (NumPy)

Building a GPT from scratch in pure **NumPy**, one component at a time, to understand how every layer works: forward pass, backward pass, and the math behind it. No PyTorch, no autograd.

## Roadmap

Progress follows the video chapters. Checked items are done and pushed.

| # | Component | What it does | Status |
|---|-----------|--------------|--------|
| 1 | Embedding layer | Turns token IDs into vectors | [ ] |
| 2 | Softmax (forward) | Turns logits into probabilities | [x] |
| 3 | Softmax (numerically stable) | Subtracts max logit so `exp()` can't overflow | [x] |
| 4 | Softmax (backward) | Gradient through softmax | [ ] |
| 5 | Cross-entropy loss | Measures how wrong the predicted probabilities are | [ ] |
| 6 | Stable cross-entropy | Log-sum-exp trick | [ ] |
| 7 | Self-attention (single, then all embeddings) | Each token looks at the others | [ ] |
| 8 | Attention with trainable weights | Q, K, V projections | [ ] |
| 9 | Causal attention | Mask so tokens can't see the future | [ ] |
| 10 | Multi-head attention | Several attention patterns in parallel | [ ] |
| 11 | Layer normalization | Stabilizes activations | [ ] |
| 12 | Feed-forward network | Per-token MLP | [ ] |
| 13 | Transformer block | Attention + FFN combined | [ ] |
| 14 | Residual connections | Skip connections for gradient flow | [ ] |
| 15 | Full GPT class | Everything stacked end to end | [ ] |

## Project structure

```
GPT_From_Scratch/
├── docs/        # handwritten and diagram notes for each concept
├── gpt.py       # the implementation
├── README.md
└── .gitignore
```

## Setup

Requires Python 3.12+ and NumPy.

```bash
git clone https://github.com/imraghavbansal/gpt-from-scratch.git
cd gpt-from-scratch
pip install numpy
python gpt.py
```

