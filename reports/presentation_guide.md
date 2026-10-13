# GraphForge — Presentation Guide

## Overview
This guide helps you deliver a confident, technically sound presentation of the GraphForge project. Each section maps to an assignment and includes talking points, anticipated questions, and key experimental evidence to cite.

---

## Assignment 1: GCN from Scratch

### What to say
> "I implemented a two-layer Graph Convolutional Network entirely from PyTorch tensor operations — no GNN library layers. This forced me to understand every mathematical step: self-loops, symmetric degree normalization, and the message passing aggregation."

### Key equations to have ready
```
Ã  = A + I                          (self-loops)
Â  = D̃^{-½} Ã D̃^{-½}              (symmetric normalization)
H¹ = ReLU( Â · X · W⁰ )            (layer 0)
Z  = softmax( Â · H¹ · W¹ )         (layer 1)
```

### Manual verification — talk through the node-0 calculation
> "I selected a fixed 5-node graph with known weights. I computed node 0's output by hand:
> - Node 0 connects to nodes 0 (self), 1, and 2 — all degree 3.
> - Â[0,j] = 1/√(3·3) = 1/3 for j ∈ {0,1,2}
> - H[0] = (1/3)(X[0]·W + X[1]·W + X[2]·W)
> The PyTorch implementation matched to 1e-5 tolerance."

### Permutation equivariance — why it matters
> "A GNN must not depend on the arbitrary numbering of nodes. I verified this by permuting all 120 nodes simultaneously — reordering features, adjacency rows+columns, and labels — and confirmed the model's predictions permuted identically."

### Experimental results to cite (fill in after running)
| Model | Test Accuracy |
|-------|--------------|
| GCN (original graph) | TBD |
| MLP (features only) | TBD |
| GCN (randomized edges) | TBD |

### Anticipated questions
**Q: Why does symmetric normalization work better than row normalization?**
> Row normalization D⁻¹A only accounts for the receiver's degree. Symmetric normalization D⁻½AD⁻½ downweights contributions from high-degree senders too — preventing popular hub nodes from dominating every message. It also preserves the symmetry of undirected graphs.

**Q: Does your GCN actually use graph structure?**
> Yes. Feature 1 in the synthetic data has zero class signal (mean is identical for both classes). An MLP that only sees features cannot use Feature 1. The GCN, by aggregating neighbors, effectively votes on a node's class using its neighbors' Feature 0 (which does have signal). The performance gap between GCN and both MLP and randomized-edge GCN is the experimental evidence.

**Q: What is permutation equivariance vs. permutation invariance?**
> Equivariance: the output transforms with the input. If you reorder nodes, predictions reorder the same way. Invariance: the output is completely unchanged. Graph-level predictions (e.g., "is this molecule toxic?") should be invariant. Node-level predictions should be equivariant. Our GCN is equivariant because we care about each individual node's class.

---

## Documented Implementation Problem: PyTorch CPU/CUDA Install Conflict

### Problem
When running `uv pip install --index-url https://download.pytorch.org/whl/cpu`, uv still resolved and downloaded the full CUDA PyTorch build (torch: 864 MB + nvidia-cudnn: 634 MB + nvidia-cublas: 347 MB = ~2.5 GB total), making local development setup impractical.

### Investigation
- `uv pip install` is a "bypass" tool that operates directly, ignoring project-level `[tool.uv.sources]` configuration.
- When no explicit index is locked in `pyproject.toml`, uv resolves `torch` from PyPI where the latest version includes CUDA dependencies.
- The `--index-url` CLI flag sets the *primary* index but uv's solver still pulls from PyPI for packages not found on the CPU index.

### Resolution
Added to `pyproject.toml`:
```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url  = "https://download.pytorch.org/whl/cpu"
explicit = true          # ONLY use this index for packages assigned to it

[tool.uv.sources]
torch = { index = "pytorch-cpu" }  # force torch → CPU index
```
Then used `uv sync --extra dev` instead of `uv pip install`.

**Result:** torch downloads as 166 MB CPU wheel. No CUDA libraries are installed. Install completes in ~2 minutes instead of >20 minutes.

### Lesson
Use project-level source configuration (`[tool.uv.sources]`) for packages with multiple build variants. Never rely on CLI flags alone for reproducible dependency resolution.

---

## Assignment 2: Cora Architecture Comparison
*(To be completed in Milestone 2)*

---

## Assignment 3: Link Prediction
*(To be completed in Milestone 3)*

---

## Assignment 4: MLOps Pipeline
*(To be completed in Milestone 4)*
