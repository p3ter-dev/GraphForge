"""
Synthetic graph generation for Assignment 1.

We build a TWO-COMMUNITY graph that is carefully designed so that
the graph structure carries class-relevant information BEYOND what
the individual node features alone can tell us.

Generation procedure
--------------------
1. Nodes 0..N//2-1  belong to class 0 (community A).
   Nodes N//2..N-1  belong to class 1 (community B).

2. Node features:
   - Feature 0: class signal + noise  (mean 1.0 for class 0, -1.0 for class 1)
   - Feature 1: pure noise            (no class signal — only graph helps here)
   - Feature 2: weak class signal     (mean 0.3 for class 0, -0.3 for class 1)

   Noise std = 1.2, so features alone are not perfectly separable.
   This is intentional: graph connections provide the extra signal
   that makes a GCN outperform an MLP.

3. Edges are drawn using the stochastic block model (SBM):
   - p_intra = 0.25  — probability of edge within the same community
   - p_inter = 0.01  — probability of edge between different communities

   Result: nodes in the same class are 25x more likely to be connected.
   A GCN that aggregates neighbor features will effectively "vote"
   on a node's class using its neighbors.

4. Labels: y[i] = 0 or 1. Labels are NEVER placed in the feature matrix X.

5. Splits: 60% train, 20% validation, 20% test — stratified by class.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor


def make_two_community_graph(
    n_nodes: int = 120,
    n_features: int = 3,
    p_intra: float = 0.25,
    p_inter: float = 0.01,
    feature_noise_std: float = 1.2,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Build a synthetic two-community graph for GCN classification.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes. Will be split evenly between two classes.
    n_features : int
        Number of features per node (must be >= 3 for the designed signal).
    p_intra : float
        Probability of an edge between two nodes in the SAME community.
    p_inter : float
        Probability of an edge between nodes in DIFFERENT communities.
    feature_noise_std : float
        Standard deviation of Gaussian noise added to features.
        Higher = harder problem = more graph structure is needed.
    train_frac, val_frac : float
        Fractions for splits (test = 1 - train_frac - val_frac).
    seed : int
        Random seed for full reproducibility.

    Returns
    -------
    dict with keys:
        x       : Tensor [N, F]  — node feature matrix
        y       : Tensor [N]     — class labels (0 or 1)
        adj     : Tensor [N, N]  — float adjacency matrix (no self-loops yet)
        edge_index : Tensor [2, E]  — COO-format edge list
        train_mask, val_mask, test_mask : BoolTensor [N]
        metadata : dict  — generation parameters for reproducibility
    """
    # ── reproducibility ──────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    assert n_nodes >= 4, "Need at least 4 nodes."
    assert n_features >= 3, "Design requires at least 3 features."
    assert 0 < train_frac + val_frac < 1.0, "Fractions must sum to < 1."

    half = n_nodes // 2
    # Handle odd n_nodes: first class gets the extra node.
    n_class0 = half + (n_nodes % 2)
    n_class1 = half

    # ── labels ───────────────────────────────────────────────────────────────
    y_np = np.array([0] * n_class0 + [1] * n_class1, dtype=np.int64)

    # ── node features ─────────────────────────────────────────────────────────
    #  Shape: [N, F]
    X = rng.normal(0.0, feature_noise_std, size=(n_nodes, n_features))

    # Feature 0: strong class signal
    class_signal_0 = np.where(y_np == 0, 1.0, -1.0)
    X[:, 0] += class_signal_0

    # Feature 1: pure noise (NO class signal added — graph must help here)
    # X[:, 1] already pure noise from above

    # Feature 2: weak class signal
    class_signal_2 = np.where(y_np == 0, 0.3, -0.3)
    X[:, 2] += class_signal_2

    # ── adjacency matrix (stochastic block model) ─────────────────────────────
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            prob = p_intra if y_np[i] == y_np[j] else p_inter
            if rng.random() < prob:
                A[i, j] = 1.0
                A[j, i] = 1.0  # undirected

    # ── splits ────────────────────────────────────────────────────────────────
    # Stratified: ensure both classes appear in every split.
    idx0 = np.where(y_np == 0)[0]
    idx1 = np.where(y_np == 1)[0]

    def split_indices(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng_local = np.random.default_rng(seed + 1)
        idx = rng_local.permutation(idx)
        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

    tr0, va0, te0 = split_indices(idx0)
    tr1, va1, te1 = split_indices(idx1)

    train_idx = np.concatenate([tr0, tr1])
    val_idx = np.concatenate([va0, va1])
    test_idx = np.concatenate([te0, te1])

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # ── edge_index (COO format for PyTorch) ───────────────────────────────────
    rows, cols = np.where(A > 0)
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)

    # ── convert to tensors ────────────────────────────────────────────────────
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    adj = torch.tensor(A, dtype=torch.float32)

    metadata = {
        "n_nodes": n_nodes,
        "n_features": n_features,
        "p_intra": p_intra,
        "p_inter": p_inter,
        "feature_noise_std": feature_noise_std,
        "n_edges": int(edge_index.shape[1]),
        "n_class0": int(n_class0),
        "n_class1": int(n_class1),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "seed": seed,
    }

    return {
        "x": x,
        "y": y,
        "adj": adj,
        "edge_index": edge_index,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "metadata": metadata,
    }


def print_graph_summary(data: dict) -> None:
    """Print a human-readable summary of the synthetic graph."""
    m = data["metadata"]
    print("=" * 50)
    print("Synthetic Two-Community Graph Summary")
    print("=" * 50)
    print(f"  Nodes         : {m['n_nodes']}  (class 0: {m['n_class0']}, class 1: {m['n_class1']})")
    print(f"  Edges         : {m['n_edges']}  (undirected, so {m['n_edges']//2} unique)")
    print(f"  Node features : {m['n_features']}")
    print(f"  Intra-class p : {m['p_intra']}  (within same community)")
    print(f"  Inter-class p : {m['p_inter']}  (between communities)")
    print(f"  Feature noise : {m['feature_noise_std']}")
    print(f"  Split         : train={m['n_train']}, val={m['n_val']}, test={m['n_test']}")
    print(f"  Seed          : {m['seed']}")
    print("=" * 50)
    print()
    print("Feature design:")
    print("  feat 0: class signal (mean ±1.0) + noise → useful but noisy")
    print("  feat 1: pure noise → GCN must use graph to classify well")
    print("  feat 2: weak signal (mean ±0.3) + noise → marginal help")
