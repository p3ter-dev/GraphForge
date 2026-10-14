"""
GNN Model Architectures for Cora Document Classification
=========================================================
Implements:
  1. CoraMLP       — Baseline (Node features only, ignores edges)
  2. CoraGCN       — Graph Convolutional Network (Spectral First-Order)
  3. CoraGraphSAGE — Inductive Representation Learning (Sample & Aggregate)
  4. CoraGAT       — Graph Attention Network (Self-Attention over Neighborhoods)

All models adhere to a unified interface:
  - forward(x, edge_index) -> log_softmax logits [N, num_classes]
  - get_embeddings(x, edge_index) -> hidden representations [N, hidden_dim]
  - support configurable depth (num_layers) for over-smoothing experiments
"""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-Layer Perceptron (MLP Baseline)
# ─────────────────────────────────────────────────────────────────────────────

class CoraMLP(nn.Module):
    """
    Multi-layer Perceptron baseline for Cora.
    Ignores graph structure completely (edge_index is accepted for API consistency).
    """

    def __init__(
        self,
        in_features: int = 1433,
        hidden_dim: int = 64,
        num_classes: int = 7,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_features, num_classes))
        else:
            self.layers.append(nn.Linear(in_features, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.layers[-1](h)
        return F.log_softmax(out, dim=-1)

    def get_embeddings(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            h = x
            for layer in self.layers[:-1]:
                h = layer(h)
                h = F.relu(h)
            return h


# ─────────────────────────────────────────────────────────────────────────────
# 2. Graph Convolutional Network (GCN)
# ─────────────────────────────────────────────────────────────────────────────

class CoraGCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).
    Propagates messages via symmetrically normalized adjacency:
        H^(l+1) = ReLU( D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H^(l) W^(l) )
    """

    def __init__(
        self,
        in_features: int = 1433,
        hidden_dim: int = 64,
        num_classes: int = 7,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_features, num_classes))
        else:
            self.convs.append(GCNConv(in_features, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.convs[-1](h, edge_index)
        return F.log_softmax(out, dim=-1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            h = x
            for conv in self.convs[:-1]:
                h = conv(h, edge_index)
                h = F.relu(h)
            return h


# ─────────────────────────────────────────────────────────────────────────────
# 3. GraphSAGE (Sample and Aggregate)
# ─────────────────────────────────────────────────────────────────────────────

class CoraGraphSAGE(nn.Module):
    """
    GraphSAGE (Hamilton et al., 2017) using mean aggregation.
    Concatenates node's own representation with mean neighbor representation:
        h_{N(v)}^(l) = MEAN_{u in N(v)} h_u^(l-1)
        h_v^(l) = sigma( W · CONCAT(h_v^(l-1), h_{N(v)}^(l)) )
    """

    def __init__(
        self,
        in_features: int = 1433,
        hidden_dim: int = 64,
        num_classes: int = 7,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_features, num_classes, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_features, hidden_dim, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            self.convs.append(SAGEConv(hidden_dim, num_classes, aggr=aggr))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.convs[-1](h, edge_index)
        return F.log_softmax(out, dim=-1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            h = x
            for conv in self.convs[:-1]:
                h = conv(h, edge_index)
                h = F.relu(h)
            return h


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graph Attention Network (GAT)
# ─────────────────────────────────────────────────────────────────────────────

class CoraGAT(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2018).
    Applies multi-head self-attention over neighborhoods:
        alpha_{ij} = softmax_j( LeakyReLU( a^T [Wh_i || Wh_j] ) )
    """

    def __init__(
        self,
        in_features: int = 1433,
        hidden_dim: int = 64,
        num_classes: int = 7,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATConv(in_features, num_classes, heads=1, concat=False, dropout=dropout))
        else:
            # First layer uses multi-head attention; output dim is hidden_dim // heads per head
            head_dim = max(1, hidden_dim // heads)
            self.convs.append(GATConv(in_features, head_dim, heads=heads, concat=True, dropout=dropout))
            
            # Intermediate layers (if depth > 2)
            current_dim = head_dim * heads
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(current_dim, head_dim, heads=heads, concat=True, dropout=dropout))
                current_dim = head_dim * heads

            # Final classification layer: single head, no concatenation, produces num_classes
            self.convs.append(GATConv(current_dim, num_classes, heads=1, concat=False, dropout=dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs[:-1]):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index)
            h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.convs[-1](h, edge_index)
        return F.log_softmax(out, dim=-1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            h = x
            for conv in self.convs[:-1]:
                h = conv(h, edge_index)
                h = F.elu(h)
            return h


# ─────────────────────────────────────────────────────────────────────────────
# 5. Over-smoothing & Representation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mean_pairwise_cosine_similarity(embeddings: torch.Tensor, sample_size: int = 500, seed: int = 42) -> float:
    """
    Measure over-smoothing in node embeddings.

    As GNN depth increases, representations of all nodes often collapse to
    a single vector (over-smoothing).
    When over-smoothing occurs:
        Cosine similarity -> 1.0 (mean pairwise distance -> 0.0).

    Returns
    -------
    mean_sim : float in [-1, 1], where 1.0 represents completely collapsed representations.
    """
    N = embeddings.shape[0]
    if N > sample_size:
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(N, generator=rng)[:sample_size]
        sub = embeddings[indices]
    else:
        sub = embeddings

    # Normalize vectors to unit norm
    normed = F.normalize(sub, p=2, dim=1)
    # Cosine similarity matrix [S, S]
    sim_matrix = torch.mm(normed, normed.t())

    # Average off-diagonal elements
    S = sub.shape[0]
    mask = ~torch.eye(S, dtype=torch.bool, device=embeddings.device)
    mean_sim = sim_matrix[mask].mean().item()
    return float(mean_sim)
