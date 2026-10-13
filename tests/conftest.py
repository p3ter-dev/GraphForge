"""
Shared pytest fixtures for GraphForge tests.
"""

import pytest
import torch

from graphforge.data.synthetic import make_two_community_graph
from graphforge.models.gcn_scratch import TwoLayerGCN, preprocess_adjacency
from graphforge.models.mlp import MLP


@pytest.fixture(scope="session")
def synthetic_data():
    """Small synthetic graph for fast unit tests (session-scoped = created once)."""
    return make_two_community_graph(n_nodes=40, seed=0)


@pytest.fixture(scope="session")
def norm_adj(synthetic_data):
    return preprocess_adjacency(synthetic_data["adj"])


@pytest.fixture
def small_gcn():
    """A tiny GCN for shape/forward tests — not trained."""
    return TwoLayerGCN(in_features=3, hidden_dim=8, n_classes=2, dropout=0.0)


@pytest.fixture
def small_mlp():
    """A tiny MLP for shape/forward tests — not trained."""
    return MLP(in_features=3, hidden_dim=8, n_classes=2, dropout=0.0)
