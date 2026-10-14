"""
Link Prediction Models: Encoders, Decoders, and Common Neighbors Baseline
========================================================================
Implements:
  1. GNNEncoder          — 2-layer GCN encoder producing latent embeddings Z
  2. DotProductDecoder   — Standard dot-product link probability predictor
  3. SymmetricMLPDecoder — Guaranteed symmetric MLP decoder: s(u, v) == s(v, u)
  4. CommonNeighbors     — Non-parametric topological heuristic on training graph
  5. LinkPredictor       — End-to-end differentiable link prediction system
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ─────────────────────────────────────────────────────────────────────────────
# 1. GNN Node Encoder
# ─────────────────────────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """
    Two-layer GCN encoder that maps node features X into latent embeddings Z.
    Uses strictly the training message-passing graph.
    """

    def __init__(self, in_features: int, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index)
        return z


# ─────────────────────────────────────────────────────────────────────────────
# 2. Decoders
# ─────────────────────────────────────────────────────────────────────────────

class DotProductDecoder(nn.Module):
    """
    Computes link existence probability via dot-product:
        s(u, v) = sigmoid( z_u^T z_v )

    Inherently symmetric because dot product is commutative.
    """

    def forward(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        u, v = edge_pairs[0], edge_pairs[1]
        z_u = z[u]
        z_v = z[v]
        scores = (z_u * z_v).sum(dim=-1)
        return torch.sigmoid(scores)


class SymmetricMLPDecoder(nn.Module):
    """
    Learned non-linear decoder guaranteed to be permutation invariant with respect
    to node ordering: s(u, v) == s(v, u).

    Construction:
      Combines three symmetric operator transforms:
        1. Element-wise absolute difference: |z_u - z_v| == |z_v - z_u|
        2. Element-wise Hadamard product:    z_u ⊙ z_v == z_v ⊙ z_u
        3. Element-wise addition:            z_u + z_v == z_v + z_u

      Representation:
        h_{uv} = CONCAT( |z_u - z_v|, z_u ⊙ z_v, z_u + z_v )  [size: 3 * D]

      Since h_{uv} == h_{vu}, any downstream MLP operating on h_{uv}
      is mathematically guaranteed to produce identical outputs regardless of pair order.
    """

    def __init__(self, emb_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        u, v = edge_pairs[0], edge_pairs[1]
        z_u = z[u]
        z_v = z[v]

        diff = (z_u - z_v).abs()
        prod = z_u * z_v
        summ = z_u + z_v

        h_sym = torch.cat([diff, prod, summ], dim=-1)  # [num_pairs, 3 * emb_dim]
        logits = self.mlp(h_sym).squeeze(-1)
        return torch.sigmoid(logits)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Common Neighbors Baseline (Heuristic)
# ─────────────────────────────────────────────────────────────────────────────

def compute_common_neighbors_score(
    train_message_edge_index: torch.Tensor,
    num_nodes: int,
    edge_pairs: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate common-neighbors score:
        s(u, v) = | N_train(u) ∩ N_train(v) |

    Computed STRICTLY on the training message-passing graph using
    sparse adjacency matrix multiplication: (A_train^2)[u, v].
    """
    # Build sparse binary adjacency for training graph
    indices = train_message_edge_index
    values = torch.ones(indices.shape[1], dtype=torch.float32)
    adj_train = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    # Pre-build neighbor adjacency lists for fast exact set intersection
    src, dst = train_message_edge_index[0].cpu().numpy(), train_message_edge_index[1].cpu().numpy()
    nbr_sets = [set() for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        nbr_sets[s].add(d)

    pairs_u = edge_pairs[0].cpu().numpy()
    pairs_v = edge_pairs[1].cpu().numpy()

    scores = []
    for u, v in zip(pairs_u, pairs_v):
        cn = len(nbr_sets[u].intersection(nbr_sets[v]))
        scores.append(float(cn))

    return torch.tensor(scores, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-End Link Predictor Module
# ─────────────────────────────────────────────────────────────────────────────

class LinkPredictor(nn.Module):
    """
    Wraps GNNEncoder and a choice of Decoder (DotProduct or SymmetricMLP).
    """

    def __init__(self, encoder: GNNEncoder, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        train_edge_index: torch.Tensor,
        eval_edge_pairs: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encoder(x, train_edge_index)
        scores = self.decoder(z, eval_edge_pairs)
        return scores
