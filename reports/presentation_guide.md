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

### Experimental results to cite
| Model | Test Accuracy (Seed 42) | Multi-seed (Mean ± Std) |
|-------|-------------------------|-------------------------|
| GCN (original graph) | 100.00% | **100.00% ± 0.00%** |
| MLP (features only) | 83.33% | **81.94% ± 8.56%** |
| GCN (randomized edges) | 75.00% | **59.72% ± 11.95%** |

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

### What to say
> "In Assignment 2, I benchmarked four distinct inductive biases on the Cora citation graph: an MLP baseline, a Graph Convolutional Network (GCN), GraphSAGE, and a Graph Attention Network (GAT). All four models were tested under identical hyperparameter budgets across three random seeds."

### Key results to cite
| Model | Test Accuracy | Macro-F1 | Train Time | Parameters |
|-------|---------------|----------|------------|------------|
| **GCN** | **80.60% ± 0.42%** | **0.8006** | 8.8s | 92,231 |
| **GAT** | **80.40% ± 0.24%** | **0.7982** | 17.1s | 92,373 |
| **GraphSAGE** | **78.83% ± 0.17%** | **0.7829** | 12.0s | 184,391 |
| **MLP** | **57.63% ± 0.96%** | **0.5621** | 3.2s | 92,231 |

### Anticipated Questions & Answers

**Q: Why does GCN achieve +23% higher accuracy than MLP on Cora?**
> "Cora's node features are 1,433-dimensional binary word vectors with 98.7% sparsity (averaging only 18 active words per paper). Text features alone suffer from extreme polysemy and document sparsity. Because Cora's citation graph has an edge homophily of 81.0%, neighboring documents strongly belong to the same topic. GCN message passing pools neighbor word distributions, smoothing out local feature noise."

**Q: Why didn't GAT significantly beat GCN despite having attention mechanisms?**
> "GAT computes dynamic self-attention coefficients $\alpha_{ij}$ between connected papers. However, on Cora, 81% of edges are already intra-class citations. Uniform degree-normalized aggregation $\hat{\mathbf{A}}$ is already near-optimal when the vast majority of neighbors share the true label. The extra parameterization and multi-head attention do not provide enough signal to offset the added variance and doubled training latency."

**Q: What is over-smoothing and how did you measure it?**
> "Over-smoothing occurs when stacking many GNN layers causes all node representations across the entire graph to converge toward a uniform stationary state. I tracked this by measuring the mean pairwise cosine similarity of node embeddings. At 2 layers, cosine similarity was 0.12 (well-separated clusters). By 6 layers, cosine similarity rose to 0.96 (almost total representation collapse), and test accuracy dropped from 80.6% down to 55.4%."

---

## Assignment 3: Link Prediction & Data Leakage Guards

### What to say
> "In Assignment 3, I developed an unsupervised link prediction system on Cora without using document class labels. I compared a Common-Neighbors baseline, a Dot-Product decoder, and a custom Symmetric MLP decoder designed to guarantee order invariance ($s(u, v) \equiv s(v, u)$). Crucially, I engineered automated data leakage guards and ran a deliberately leaky experiment to expose how improper message passing inflates metrics."

### Key results to cite
| Approach | Test ROC-AUC | Test Average Precision | Status |
|----------|--------------|------------------------|--------|
| **Common Neighbors** | 72.12% | 0.7192 | Non-parametric Baseline |
| **GNN + Dot-Product** | 86.92% | 0.8723 | Valid ($E_{\text{train}}$ message passing) |
| **GNN + Symmetric MLP** | **92.29%** | **0.9344** | **Valid Champion** |
| **GNN + Dot-Product (LEAKY)** | 94.17% | 0.9277 | **Fraudulent / Invalid** ($E_{\text{full}}$ leakage) |

### Anticipated Questions & Answers

**Q: How did you ensure your MLP decoder treats undirected links symmetrically?**
> "In an undirected graph, edge $(u, v)$ is identical to edge $(v, u)$. Simply concatenating $[z_u \,\|\, z_v]$ is order-dependent: the MLP might predict different scores for $(u, v)$ than $(v, u)$. I guaranteed symmetry by combining three invariant operators: $|z_u - z_v|$, $z_u \odot z_v$, and $z_u + z_v$. Because all three operations are commutative, swapping $u$ and $v$ produces the exact same input tensor to the MLP."

**Q: What is the 'Leaky Evaluation' experiment and why is it invalid?**
> "In link prediction, held-out validation and test edges must be removed from the message-passing graph. In my deliberately leaky experiment, I allowed the GCN to aggregate messages over test edges during evaluation. The ROC-AUC jumped from 86.9% to 94.2%. This gain is completely fraudulent: the model didn't learn to predict future citations; it simply detected that paper $u$ was already exchanging messages with paper $v$."

**Q: Why isn't an unobserved link considered a true negative?**
> "Under the Open-World Assumption, scientific citation graphs are incomplete. An unobserved link simply means paper $u$ hasn't cited paper $v$ yet. In our top-10 candidate predictions, several unobserved pairs had 0 common citations in the training graph but near-identical text and latent embeddings—indicating high-confidence missing literature links that warrant domain validation."

---

## Assignment 4: MLOps Pipeline
*(To be completed in Milestone 4)*
