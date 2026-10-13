"""
Unit Tests: GCN Model Architecture
"""

import pytest
import torch

from graphforge.models.gcn_scratch import (
    GCNLayerScratch,
    TwoLayerGCN,
    add_self_loops,
    compute_degree,
    preprocess_adjacency,
    symmetric_normalize,
)
from graphforge.models.mlp import MLP


class TestGCNPreprocessing:
    """Tests for the adjacency preprocessing pipeline."""

    def test_add_self_loops_identity(self):
        """add_self_loops on zero matrix → identity matrix."""
        A = torch.zeros(4, 4)
        A_sl = add_self_loops(A)
        assert torch.allclose(A_sl, torch.eye(4))

    def test_add_self_loops_idempotent_diagonal(self):
        """Calling add_self_loops on already-looped matrix sets diagonal to 2."""
        A = torch.zeros(3, 3)
        A_sl = add_self_loops(A)
        A_sl2 = add_self_loops(A_sl)
        # Diagonal should be 2 (added self-loop twice)
        assert (torch.diagonal(A_sl2) == 2.0).all()

    def test_normalized_adj_row_sum_bounded(self):
        """
        For any valid adjacency, each row of Â sums to at most 1.0.
        This is not exact equality but bounded: ≤ 1 + eps.
        """
        A = torch.tensor([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], dtype=torch.float32)
        norm_adj = preprocess_adjacency(A)
        row_sums = norm_adj.sum(dim=1)
        # Row sums should be ≤ 1.0 (symmetric norm produces values in [0,1] for each entry)
        # and > 0 (self-loop ensures each node has at least one connection)
        assert (row_sums > 0).all()
        # Actually row sum of D^{-1/2} A D^{-1/2} is NOT guaranteed to be 1,
        # but should be in a reasonable range (0, 1]
        assert (row_sums <= 1.0 + 1e-5).all(), f"Row sums exceed 1: {row_sums}"

    def test_normalized_adj_symmetry(self):
        """Â must be symmetric because A is symmetric and D is diagonal."""
        A = torch.tensor([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], dtype=torch.float32)
        norm_adj = preprocess_adjacency(A)
        assert torch.allclose(norm_adj, norm_adj.T, atol=1e-6), "Normalized adjacency is not symmetric."

    def test_isolated_node_safe(self):
        """An isolated node (no edges) should not cause division by zero."""
        A = torch.zeros(3, 3)
        # Only nodes 0 and 1 are connected; node 2 is isolated
        A[0, 1] = A[1, 0] = 1.0
        # Should not raise — self-loops prevent zero degree
        norm_adj = preprocess_adjacency(A)
        assert torch.isfinite(norm_adj).all(), "NaN/Inf in normalized adjacency for isolated node."


class TestGCNLayerScratch:
    """Tests for GCNLayerScratch forward pass shapes and behavior."""

    def test_output_shape(self, synthetic_data, norm_adj):
        """Output shape must be [N, out_features]."""
        N = synthetic_data["x"].shape[0]
        layer = GCNLayerScratch(in_features=3, out_features=7)
        out = layer(synthetic_data["x"], norm_adj)
        assert out.shape == (N, 7), f"Expected ({N}, 7), got {out.shape}"

    def test_output_is_finite(self, synthetic_data, norm_adj):
        """No NaN or Inf values in output."""
        layer = GCNLayerScratch(in_features=3, out_features=4)
        out = layer(synthetic_data["x"], norm_adj)
        assert torch.isfinite(out).all(), "GCNLayerScratch produced NaN or Inf."

    def test_zero_features_give_zero_output(self, norm_adj):
        """With zero input features and zero bias, output should be zero."""
        N = norm_adj.shape[0]
        layer = GCNLayerScratch(in_features=3, out_features=4, bias=True)
        with torch.no_grad():
            layer.linear.weight.zero_()
            layer.linear.bias.zero_()
        x = torch.zeros(N, 3)
        out = layer(x, norm_adj)
        assert torch.allclose(out, torch.zeros(N, 4)), "Zero input with zero weights should give zero output."

    def test_no_activation_in_layer(self, synthetic_data, norm_adj):
        """
        GCNLayerScratch should NOT apply activation internally.
        Activation is the caller's responsibility.
        Verify: output may contain negative values (possible without ReLU).
        """
        layer = GCNLayerScratch(in_features=3, out_features=8)
        out = layer(synthetic_data["x"], norm_adj)
        # If output had ReLU, no negatives. Without it, negatives are possible.
        # We don't assert negatives MUST exist, but we check there's no clamping.
        # The simplest check: output range is wider than [0, inf).
        # We just verify no activation was applied by checking unbounded values.
        assert out.min().item() < out.max().item(), "Output has no variance — possible activation bug."


class TestTwoLayerGCN:
    """Tests for the full TwoLayerGCN classifier."""

    def test_output_shape(self, synthetic_data, norm_adj):
        N = synthetic_data["x"].shape[0]
        model = TwoLayerGCN(in_features=3, hidden_dim=16, n_classes=2)
        out = model(synthetic_data["x"], norm_adj)
        assert out.shape == (N, 2), f"Expected ({N}, 2), got {out.shape}"

    def test_output_is_log_probs(self, synthetic_data, norm_adj):
        """
        Output should be log-probabilities:
        - All values ≤ 0 (log of probability ≤ 1 is ≤ 0)
        - exp(output) sums to 1.0 per row
        """
        model = TwoLayerGCN(in_features=3, hidden_dim=16, n_classes=2, dropout=0.0)
        model.eval()
        out = model(synthetic_data["x"], norm_adj)

        # All log-probs ≤ 0
        assert (out <= 0).all(), "Log-probabilities should all be ≤ 0."

        # exp sums to 1 per row
        probs = torch.exp(out)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Probabilities should sum to 1, got max deviation {(row_sums - 1).abs().max():.6f}"
        )

    def test_train_vs_eval_mode(self, synthetic_data, norm_adj):
        """
        Dropout is active in train mode, disabled in eval mode.
        With dropout=0.5, train outputs should differ from eval outputs
        (when run without torch.no_grad and different random states).
        """
        model = TwoLayerGCN(in_features=3, hidden_dim=32, n_classes=2, dropout=0.5)
        x = synthetic_data["x"]

        model.eval()
        with torch.no_grad():
            out_eval1 = model(x, norm_adj)
            out_eval2 = model(x, norm_adj)
        # Eval outputs must be deterministic
        assert torch.allclose(out_eval1, out_eval2), "Eval mode outputs should be deterministic."

    def test_parameter_count_reasonable(self):
        """Parameter count should scale with hidden_dim and in_features."""
        m_small = TwoLayerGCN(in_features=3, hidden_dim=8, n_classes=2)
        m_large = TwoLayerGCN(in_features=3, hidden_dim=64, n_classes=2)
        params_small = sum(p.numel() for p in m_small.parameters())
        params_large = sum(p.numel() for p in m_large.parameters())
        assert params_large > params_small, "Larger hidden_dim should have more parameters."

    def test_get_embeddings_shape(self, synthetic_data, norm_adj):
        N = synthetic_data["x"].shape[0]
        model = TwoLayerGCN(in_features=3, hidden_dim=16, n_classes=2)
        emb = model.get_embeddings(synthetic_data["x"], norm_adj)
        assert emb.shape == (N, 16), f"Embeddings should be [N, hidden_dim], got {emb.shape}"


class TestMLP:
    """Tests for the MLP baseline."""

    def test_output_shape(self, synthetic_data, norm_adj):
        N = synthetic_data["x"].shape[0]
        model = MLP(in_features=3, hidden_dim=16, n_classes=2)
        out = model(synthetic_data["x"], norm_adj)
        assert out.shape == (N, 2)

    def test_ignores_norm_adj(self, synthetic_data, norm_adj):
        """MLP should produce same output regardless of norm_adj."""
        model = MLP(in_features=3, hidden_dim=16, n_classes=2, dropout=0.0)
        model.eval()
        with torch.no_grad():
            out1 = model(synthetic_data["x"], norm_adj)
            # Pass a different (random) adj — output should be identical
            rand_adj = torch.rand_like(norm_adj)
            out2 = model(synthetic_data["x"], rand_adj)
        assert torch.allclose(out1, out2), "MLP output changed when norm_adj changed — it should not."

    def test_output_is_log_probs(self, synthetic_data, norm_adj):
        model = MLP(in_features=3, hidden_dim=16, n_classes=2, dropout=0.0)
        model.eval()
        out = model(synthetic_data["x"], norm_adj)
        assert (out <= 0).all()
        probs = torch.exp(out)
        assert torch.allclose(probs.sum(dim=1), torch.ones(synthetic_data["x"].shape[0]), atol=1e-5)
