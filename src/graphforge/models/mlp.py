"""
MLP Baseline for Assignment 1.

This model takes node features as input but does NOT use the graph.
It serves as a comparison against the GCN to show whether graph
structure actually provides useful information.

If GCN >> MLP: the graph structure helps classification.
If GCN ≈ MLP: features alone are sufficient; graph structure adds little.
If GCN < MLP:  something is wrong with the GCN (over-smoothing, bug, etc.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for node classification.

    This MLP operates on node features INDEPENDENTLY — it has absolutely
    no knowledge of the graph topology (edges, neighbors).

    Architecture:
        X [N, F_in] → Linear → ReLU → Dropout → Linear → log_softmax → [N, C]

    Parameters
    ----------
    in_features  : number of input features per node
    hidden_dim   : width of the single hidden layer
    n_classes    : number of output classes
    dropout      : dropout probability
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x: Tensor, norm_adj: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x        : Tensor [N, F_in]   — node feature matrix
        norm_adj : ignored (accepted for API compatibility with GCN)

        Returns
        -------
        Tensor [N, n_classes]  — log-probabilities
        """
        h = self.fc1(x)                                         # [N, hidden_dim]
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.fc2(h)                                       # [N, n_classes]
        return F.log_softmax(out, dim=1)

    def get_embeddings(self, x: Tensor, norm_adj: Tensor | None = None) -> Tensor:
        """
        Return the hidden-layer representations (useful for visualization).
        Sets eval mode internally for this call.

        Returns
        -------
        Tensor [N, hidden_dim]
        """
        self.eval()
        with torch.no_grad():
            h = self.fc1(x)
            h = F.relu(h)
        return h
