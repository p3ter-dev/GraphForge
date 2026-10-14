# GraphForge — Technical Report

## Abstract

This report documents the design, implementation, experimental results, and engineering decisions for a four-assignment Graph Neural Network (GNN) project. The project covers: (1) implementing a two-layer GCN from scratch, (2) comparing GNN architectures on the Cora citation dataset, (3) building a link prediction system with leakage guards, and (4) deploying a reproducible MLOps pipeline. All code is available at the GraphForge repository.

---

## Assignment 1: GCN from Scratch

### 1.1 Synthetic Dataset Description

The synthetic dataset is a **two-community graph** generated using a Stochastic Block Model (SBM). The design ensures that graph connections provide useful signal beyond individual node features.

**Generation parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N (nodes) | 120 | ≥100 as required; even split of 60 per class |
| F (features) | 3 | Minimum for designed signal structure |
| p_intra | 0.25 | Edge probability within same community |
| p_inter | 0.01 | Edge probability between communities |
| Feature noise σ | 1.2 | High noise makes graph structure necessary |
| Train/val/test | 60/20/20 | Standard split; stratified by class |
| Random seed | 42 | Fixed for reproducibility |

**Feature design:**
- **Feature 0:** Strong class signal (mean +1.0 for class 0, -1.0 for class 1) + Gaussian noise σ=1.2
- **Feature 1:** Pure Gaussian noise, zero class signal. An MLP cannot use this feature to discriminate. The GCN can, because neighbors' Feature 0 values provide indirect information via aggregation.
- **Feature 2:** Weak class signal (mean ±0.3) + Gaussian noise

The ratio p_intra/p_inter = 25 means within-community edges are 25× more likely. When a GCN aggregates a node's neighborhood, the majority of neighbors belong to the same class, effectively providing a "community vote."

**Target labels:** Binary (0 or 1), stored in a separate integer tensor. Labels are *never* placed in the feature matrix X.

### 1.2 GCN Mathematical Derivation

The Graph Convolutional Network (Kipf & Welling, 2017) propagation rule:

$$\mathbf{H}^{(l+1)} = \sigma\!\left(\hat{\mathbf{A}}\,\mathbf{H}^{(l)}\,\mathbf{W}^{(l)}\right)$$

Where:

$$\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-\frac{1}{2}}\,\tilde{\mathbf{A}}\,\tilde{\mathbf{D}}^{-\frac{1}{2}}, \quad \tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$$

And $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$ (degree of node *i* with self-loop).

**Entry-level interpretation of Â[i,j]:**

$$\hat{A}_{ij} = \frac{\tilde{A}_{ij}}{\sqrt{\tilde{d}_i \cdot \tilde{d}_j}}$$

This scales edge (i,j) inversely by the geometric mean of the two nodes' degrees. High-degree nodes (hubs) have their influence shrunk, preventing them from dominating all their neighbors' representations.

**Two-layer full forward pass:**

$$\mathbf{Z} = \text{softmax}\!\left(\hat{\mathbf{A}}\;\text{ReLU}\!\left(\hat{\mathbf{A}}\,\mathbf{X}\,\mathbf{W}^{(0)}\right)\mathbf{W}^{(1)}\right)$$

After two layers, each node's representation incorporates information from all nodes within 2 hops (its 2-hop neighborhood).

### 1.3 Manual Layer-Output Verification

**Graph:** 5-node graph with edges 0–1, 0–2, 1–3, 2–3, 3–4.

**Node degrees after self-loops:** [3, 3, 3, 4, 2]

**Node 0 example (F_in=3, F_out=2):**

Step 1 — Â row 0:
- Â[0,0] = 1/√(3·3) = 1/3 (self-loop)
- Â[0,1] = 1/√(3·3) = 1/3
- Â[0,2] = 1/√(3·3) = 1/3
- Â[0,3] = 0, Â[0,4] = 0

Step 2 — Node feature projections X·W:
- X[0]·W = [1.0, 0.5, -0.3] @ W = [0.37, -0.07]
- X[1]·W = [-1.0, 0.2, 0.8] @ W = [-0.46, 0.76]
- X[2]·W = [0.0, -0.4, 1.1] @ W = [0.19, 0.16]

Step 3 — Aggregation:
H[0] = (1/3)·([0.37,-0.07] + [-0.46,0.76] + [0.19,0.16]) = [0.0333, 0.2833]

**PyTorch vs. NumPy agreement:** max absolute difference < 1e-5 ✓

*(See `tests/correctness/test_gcn_manual.py::TestManualGCNVerification::test_node0_output_by_hand`)*

### 1.4 Permutation Equivariance

**Property:** f(PX, PAP^T) = P · f(X, A) for any permutation matrix P.

**Test procedure:**
1. Compute predictions on original (X, A) → pred_orig [N]
2. Generate random permutation π over N nodes
3. Compute predictions on (X[π,:], A[π,:][:,π]) → pred_perm [N]
4. Assert: pred_perm == pred_orig[π]

**Result:** 0/120 nodes mismatched across 3 different random permutations ✓

*(See `tests/correctness/test_equivariance.py`)*

### 1.5 Experimental Results

Measured performance on the synthetic two-community graph (N=120, F=3, C=2, 60/20/20 stratified split):

| Model | Val Acc | Test Acc | Params | Training Time |
|-------|---------|----------|--------|--------------|
| GCN (original graph) | 1.0000 | 1.0000 | 386 | 0.7s |
| MLP (features only, no edges) | 0.7917 | 0.8333 | 386 | 0.5s |
| GCN (randomized edges) | 0.7083 | 0.7500 | 386 | 0.5s |

**3-seed robustness evaluation (seeds: 42, 123, 999):**

| Model | Seed 42 | Seed 123 | Seed 999 | Test Acc (mean ± std) |
|-------|---------|----------|----------|----------------------|
| GCN (original graph) | 1.0000 | 1.0000 | 1.0000 | **100.00% ± 0.00%** |
| MLP (features only) | 0.8333 | 0.9167 | 0.7083 | **81.94% ± 8.56%** |
| GCN (randomized edges) | 0.7500 | 0.5833 | 0.4583 | **59.72% ± 11.95%** |

### 1.6 Analysis: Does Graph Structure Help?

**Yes, conclusively.** The measured experimental evidence confirms that the GCN actively and effectively utilizes graph topology:

1. **Massive Performance Gap over Features-Only Baseline (+18.06% on average):**
   The GCN achieves a perfect 100.00% across all evaluated random seeds, compared to 81.94% for the MLP. Because both models share identical parameter counts (386 weights), identical optimization hyperparameters, and identical input node features, the performance difference is exclusively attributable to neighborhood aggregation via $\hat{\mathbf{A}}$.

2. **Denoising Mechanism on Uninformative Features:**
   In our data design, Feature 1 contains pure random noise with identical distributions across both classes ($\mathcal{N}(0, 1)$). The MLP has no way to discount or denoise this feature. In contrast, the GCN layer aggregates $\sum_{j \in \mathcal{N}(i)} \hat{A}_{ij} X_j$. Because 96% of edges connect intra-community nodes ($p_{\text{intra}}=0.25$ vs $p_{\text{inter}}=0.01$), averaging neighbor features suppresses independent Gaussian feature noise by a factor of approximately $\sqrt{d_i}$, extracting the community consensus.

3. **Topology Destruction Ablation (Randomized Edges drops to 59.72%):**
   When edges are rewired while preserving the degree sequence and node features, GCN test accuracy plummets by over 40 percentage points to 59.72% ± 11.95% (barely better than random coin-flip guessing). This demonstrates that the GCN is genuinely dependent on the homophilic community structure, rather than any artifact of graph regularization or smoothed feature norms.

---

## Assignment 2: Architecture Comparison on Cora
*(To be completed in Milestone 2)*

---

## Assignment 3: Link Prediction
*(To be completed in Milestone 3)*

---

## Assignment 4: MLOps Pipeline
*(To be completed in Milestone 4)*

---

## References

- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*. https://arxiv.org/abs/1609.02907
- Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS 2017*. https://arxiv.org/abs/1706.02216
- Veličković, P., et al. (2018). Graph Attention Networks. *ICLR 2018*. https://arxiv.org/abs/1710.10903
- Sen, P., et al. (2008). Collective Classification in Network Data. *AI Magazine*. (Cora dataset)
