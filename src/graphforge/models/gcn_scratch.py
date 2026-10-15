"""
GCN from Scratch — Assignment 1
================================
All operations implemented using pure PyTorch tensor math.
No torch_geometric or other GNN library layers are used here.

Mathematical Reference
----------------------
Given:
  A  : adjacency matrix        [N, N]  (without self-loops)
  X  : node feature matrix     [N, F]
  W  : weight matrix           [F_in, F_out]

Step 1 — Add self-loops:
    Ã = A + I_N

Step 2 — Compute degree of Ã:
    D̃[i,i] = sum_j Ã[i,j]   (diagonal matrix)

Step 3 — Symmetric normalization:
    Â = D̃^{-1/2} · Ã · D̃^{-1/2}
    Entry: Â[i,j] = Ã[i,j] / sqrt(D̃[i,i] * D̃[j,j])

Step 4 — One GCN layer:
    H = σ( Â · X · W )

Step 5 — Two-layer GCN:
    H1 = ReLU( Â · X  · W0 )          [N, hidden_dim]
    Z  = softmax( Â · H1 · W1 )       [N, n_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ─────────────────────────────────────────────────────────────────────────────
# Graph pre-processing utilities
# ─────────────────────────────────────────────────────────────────────────────

def add_self_loops(adj: Tensor) -> Tensor:
    """
    Add self-loops to an adjacency matrix.

    A self-loop on node i means: node i is its own neighbor.
    This ensures every node includes its OWN features when aggregating.

    Parameters
    ----------
    adj : Tensor [N, N]  — adjacency matrix (0s and 1s)

    Returns
    -------
    Tensor [N, N]  — adj + identity matrix
    """
    N = adj.shape[0]
    # torch.eye(N) creates an N×N identity matrix (1s on diagonal, 0s elsewhere).
    # We use the same device and dtype as the input adjacency matrix.
    I = torch.eye(N, dtype=adj.dtype, device=adj.device)
    return adj + I   # Ã = A + I


def compute_degree(adj_with_self_loops: Tensor) -> Tensor:
    """
    Compute the degree vector from an adjacency matrix (with self-loops).

    degree[i] = sum of row i = number of neighbors of i (including itself).

    Parameters
    ----------
    adj_with_self_loops : Tensor [N, N]

    Returns
    -------
    Tensor [N]  — degree of each node
    """
    # Sum each row: dim=1 means "sum across columns for each row".
    return adj_with_self_loops.sum(dim=1)   # shape [N]


def symmetric_normalize(adj_with_self_loops: Tensor) -> Tensor:
    """
    Compute the symmetrically normalized adjacency matrix.

    Â = D̃^{-1/2} · Ã · D̃^{-1/2}

    This is equivalent to scaling each entry Ã[i,j] by
    1 / sqrt(degree[i] * degree[j]).

    Why symmetric and not just D^{-1}·A (row normalization)?
    - Row normalization (D^{-1}·A) only normalizes by sender degree.
    - Symmetric normalization accounts for BOTH sender and receiver degree.
    - It preserves the symmetric structure of undirected graphs.
    - Kipf & Welling (2017) showed it empirically works better for GCNs.

    Parameters
    ----------
    adj_with_self_loops : Tensor [N, N]  — Ã = A + I

    Returns
    -------
    Tensor [N, N]  — symmetrically normalized Â
    """
    # Step 1: Compute degree vector [N]
    degree = compute_degree(adj_with_self_loops)

    # Step 2: Compute D^{-1/2} as a vector (only the diagonal matters).
    # torch.pow(x, -0.5) computes x^{-0.5} = 1/sqrt(x) element-wise.
    # We clamp at a minimum of 1e-9 to avoid division by zero for isolated nodes.
    d_inv_sqrt = torch.pow(degree.clamp(min=1e-9), -0.5)   # shape [N]

    # Step 3: Apply normalization using broadcasting.
    # d_inv_sqrt.unsqueeze(1) has shape [N, 1]  (column vector)
    # d_inv_sqrt.unsqueeze(0) has shape [1, N]  (row vector)
    #
    # Element-wise multiplication:
    #   [N, 1] * [N, N] * [1, N]
    # = Â[i,j] = d_inv_sqrt[i] * Ã[i,j] * d_inv_sqrt[j]
    #
    # This is mathematically equivalent to:
    #   D̃^{-1/2} @ Ã @ D̃^{-1/2}
    # but avoids constructing a full diagonal N×N matrix.
    norm_adj = d_inv_sqrt.unsqueeze(1) * adj_with_self_loops * d_inv_sqrt.unsqueeze(0)
    return norm_adj   # shape [N, N]


def preprocess_adjacency(adj: Tensor) -> Tensor:
    """
    Full adjacency preprocessing pipeline:
      adj  →  add self-loops  →  normalize  →  Â

    This is called ONCE before training and the result is reused every
    forward pass (it doesn't change during training).

    Parameters
    ----------
    adj : Tensor [N, N]  — raw adjacency matrix (no self-loops)

    Returns
    -------
    Tensor [N, N]  — normalized adjacency Â
    """
    adj_sl = add_self_loops(adj)
    return symmetric_normalize(adj_sl)


# ─────────────────────────────────────────────────────────────────────────────
# GCN Layer (no library — pure matrix multiplication)
# ─────────────────────────────────────────────────────────────────────────────

class GCNLayerScratch(nn.Module):
    """
    One GCN layer: H_out = σ( Â · H_in · W )

    Tensor shapes
    -------------
    Â      : [N, N]        — fixed normalized adjacency (pre-computed)
    H_in   : [N, F_in]    — input node representations
    W      : [F_in, F_out] — learned weight matrix (this layer's parameters)
    bias   : [F_out]       — optional learned bias
    H_out  : [N, F_out]   — output node representations

    The multiplication order matters:
    1. H_in @ W  → [N, F_out]   (linear transform: each node independently)
    2. Â @ (...)  → [N, F_out]  (aggregate: mix neighbors' transformed features)

    Note: We can also write  Â @ H_in @ W, which is equivalent because
    matrix multiplication is associative. The order  (H_in @ W) first then
    Â @ (...)  is often preferred in practice because:
      - H_in @ W  is [N×F_in] @ [F_in×F_out] = [N×F_out]  (cheaper if F_out < N)
      - Â @ [N×F_out] = [N×F_out]  (then aggregation on smaller matrix)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Parameters
        ----------
        in_features  : F_in  — number of input features per node
        out_features : F_out — number of output features per node
        bias         : whether to add a learnable bias term
        """
        super().__init__()

        # nn.Linear encapsulates a weight matrix W [F_in, F_out] and optional bias.
        # We use it purely for its parameter management (no activation applied here).
        # Internally: linear(x) computes x @ W.T + bias.
        # We will apply the graph aggregation separately.
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor, norm_adj: Tensor) -> Tensor:
        """
        Forward pass for one GCN layer.

        Parameters
        ----------
        x        : Tensor [N, F_in]  — input node features / representations
        norm_adj : Tensor [N, N]     — pre-computed normalized adjacency Â

        Returns
        -------
        Tensor [N, F_out]  — output node representations (BEFORE activation)
        Activation is applied by the caller so that the last layer can
        use a different activation (softmax) than intermediate layers (ReLU).
        """
        # Step 1: Linear transformation (same as a regular FC layer, applied per node)
        # x @ W.T  + bias
        # [N, F_in] @ [F_out, F_in].T = [N, F_in] @ [F_in, F_out] = [N, F_out]
        h = self.linear(x)   # shape: [N, F_out]

        # Step 2: Neighborhood aggregation
        # norm_adj @ h
        # [N, N] @ [N, F_out] = [N, F_out]
        # Row i of the result = weighted sum of all rows of h,
        # weighted by Â[i, :] (normalized neighbor contributions).
        out = torch.mm(norm_adj, h)   # shape: [N, F_out]

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Two-layer GCN Classifier
# ─────────────────────────────────────────────────────────────────────────────

class TwoLayerGCN(nn.Module):
    """
    Two-layer GCN for node classification.

    Architecture:
        X  [N, F_in]
        │
        ├─ GCN Layer 0 → ReLU → H1  [N, hidden_dim]
        │
        └─ GCN Layer 1 → (log_softmax) → Z  [N, n_classes]

    The normalized adjacency Â is pre-computed and passed in as an
    argument to forward() — it is NOT a learnable parameter.

    Parameters
    ----------
    in_features  : number of input features (F_in)
    hidden_dim   : number of hidden features (e.g. 64)
    n_classes    : number of output classes (e.g. 2)
    dropout      : dropout probability applied to H1 during training
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.layer0 = GCNLayerScratch(in_features, hidden_dim)
        self.layer1 = GCNLayerScratch(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x: Tensor, norm_adj: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x        : Tensor [N, F_in]  — node feature matrix
        norm_adj : Tensor [N, N]     — pre-computed normalized adjacency

        Returns
        -------
        Tensor [N, n_classes]  — log-probabilities for each node
        """
        # Layer 0: GCN + ReLU + Dropout
        h = self.layer0(x, norm_adj)         # [N, hidden_dim]
        h = F.relu(h)                        # non-linearity
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 1: GCN (no ReLU here; we apply log_softmax instead)
        out = self.layer1(h, norm_adj)       # [N, n_classes]
        return F.log_softmax(out, dim=1)     # [N, n_classes], log-probabilities

    def get_embeddings(self, x: Tensor, norm_adj: Tensor) -> Tensor:
        """
        Return the hidden-layer representations (useful for visualization).
        Sets eval mode internally for this call.

        Returns
        -------
        Tensor [N, hidden_dim]
        """
        self.eval()
        with torch.no_grad():
            h = self.layer0(x, norm_adj)
            h = F.relu(h)
        return h
