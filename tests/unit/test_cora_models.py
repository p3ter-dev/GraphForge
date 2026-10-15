"""
Unit Tests for Cora GNN Models and Preprocessing
=================================================
"""

import pytest
import torch

from graphforge.data.cora import compute_graph_homophily
from graphforge.models.cora_models import (
    CoraGAT,
    CoraGCN,
    CoraGraphSAGE,
    CoraMLP,
    compute_mean_pairwise_cosine_similarity,
)


@pytest.fixture
def tiny_graph():
    """Synthetic tiny graph resembling Cora format (50 nodes, 100 features, 7 classes)."""
    torch.manual_seed(42)
    N = 50
    F_in = 100
    C = 7
    x = torch.randn(N, F_in)

    # Random undirected edges
    src = torch.randint(0, N, (120,))
    dst = torch.randint(0, N, (120,))
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    y = torch.randint(0, C, (N,))
    return x, edge_index, y


class TestCoraModelArchitectures:

    @pytest.mark.parametrize("model_cls", [CoraMLP, CoraGCN, CoraGraphSAGE, CoraGAT])
    def test_output_shapes_and_log_probs(self, model_cls, tiny_graph):
        x, edge_index, y = tiny_graph
        N, F_in = x.shape
        model = model_cls(in_features=F_in, hidden_dim=32, num_classes=7, num_layers=2)
        model.eval()

        with torch.no_grad():
            out = model(x, edge_index)

        assert out.shape == (N, 7)
        assert torch.isfinite(out).all()
        # Verify log_softmax property
        probs = out.exp()
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    @pytest.mark.parametrize("model_cls", [CoraMLP, CoraGCN, CoraGraphSAGE, CoraGAT])
    def test_get_embeddings_shape(self, model_cls, tiny_graph):
        x, edge_index, y = tiny_graph
        N, F_in = x.shape
        model = model_cls(in_features=F_in, hidden_dim=32, num_classes=7, num_layers=2)
        emb = model.get_embeddings(x, edge_index)
        assert emb.shape[0] == N
        assert emb.shape[1] > 0
        assert torch.isfinite(emb).all()

    @pytest.mark.parametrize("depth", [2, 3, 4])
    def test_configurable_depth(self, depth, tiny_graph):
        x, edge_index, y = tiny_graph
        model = CoraGCN(in_features=x.shape[1], hidden_dim=16, num_classes=7, num_layers=depth)
        out = model(x, edge_index)
        assert out.shape == (x.shape[0], 7)

    def test_mlp_is_strictly_edge_independent(self, tiny_graph):
        x, edge_index, y = tiny_graph
        model = CoraMLP(in_features=x.shape[1], hidden_dim=16, num_classes=7)
        model.eval()

        with torch.no_grad():
            out_with_edges = model(x, edge_index)
            out_without_edges = model(x, None)

            # Scrambled edges
            scrambled = torch.randint(0, x.shape[0], edge_index.shape)
            out_scrambled = model(x, scrambled)

        assert torch.allclose(out_with_edges, out_without_edges, atol=1e-6)
        assert torch.allclose(out_with_edges, out_scrambled, atol=1e-6)

    def test_gnn_is_edge_dependent(self, tiny_graph):
        x, edge_index, y = tiny_graph
        model = CoraGCN(in_features=x.shape[1], hidden_dim=16, num_classes=7)
        model.eval()

        with torch.no_grad():
            out_orig = model(x, edge_index)
            # Empty graph (self-loops only)
            empty_edges = torch.empty((2, 0), dtype=torch.long)
            out_empty = model(x, empty_edges)

        diff = (out_orig - out_empty).abs().max().item()
        assert diff > 1e-4, "GCN should be dependent on graph edges"


class TestRepresentationMetrics:

    def test_pairwise_cosine_similarity_bounds(self):
        # Identity embeddings (identical for all nodes) -> similarity == 1.0
        identical = torch.ones(20, 16)
        sim_high = compute_mean_pairwise_cosine_similarity(identical)
        assert abs(sim_high - 1.0) < 1e-4

        # Random embeddings -> similarity near 0
        torch.manual_seed(42)
        rand = torch.randn(100, 128)
        sim_rand = compute_mean_pairwise_cosine_similarity(rand)
        assert -0.2 < sim_rand < 0.2

    def test_homophily_computation(self):
        # Perfect homophily
        edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        y = torch.tensor([0, 0, 1, 1])
        h = compute_graph_homophily(edges, y)
        assert h == 1.0

        # Zero homophily
        edges_bipartite = torch.tensor([[0, 2], [1, 3]])
        y_bipartite = torch.tensor([0, 1, 0, 1])
        h_zero = compute_graph_homophily(edges_bipartite, y_bipartite)
        assert h_zero == 0.0
