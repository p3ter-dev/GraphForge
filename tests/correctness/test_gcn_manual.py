"""
Correctness Test: Manual GCN Layer Verification
================================================
Assignment 1, requirement 5: Manually calculate one node's output from
one GCN layer and verify it matches the implementation.

Approach:
1. Create a tiny, fixed graph (5 nodes) with known features.
2. Create a fixed weight matrix W (not random — we set it explicitly).
3. Run the GCNLayerScratch forward pass.
4. Manually compute the same output step-by-step using NumPy.
5. Assert the two results match to 1e-5 tolerance.

This test proves that our implementation is mathematically correct,
not just that it runs without errors.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from graphforge.models.gcn_scratch import (
    GCNLayerScratch,
    add_self_loops,
    preprocess_adjacency,
    symmetric_normalize,
    compute_degree,
)


class TestManualGCNVerification:
    """
    Manual step-by-step verification that GCNLayerScratch computes
    exactly what the GCN equation says it should.
    """

    def setup_method(self):
        """Build a tiny fixed graph for all tests in this class."""
        # ── 5-node graph ─────────────────────────────────────────────────────
        # Edges: 0-1, 0-2, 1-3, 2-3, 3-4
        self.N = 5
        self.F_in = 3
        self.F_out = 2

        A_np = np.array(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=np.float32,
        )

        # Fixed node features (deterministic)
        X_np = np.array(
            [
                [1.0, 0.5, -0.3],   # node 0
                [-1.0, 0.2, 0.8],   # node 1
                [0.0, -0.4, 1.1],   # node 2
                [0.6, 0.9, -0.5],   # node 3
                [-0.2, -0.8, 0.4],  # node 4
            ],
            dtype=np.float32,
        )

        # Fixed weight matrix (NOT random — we set it to known values)
        W_np = np.array(
            [
                [0.5, -0.3],
                [-0.2, 0.7],
                [0.1, 0.4],
            ],
            dtype=np.float32,
        )

        # Fixed bias (all zeros for simplicity — easier to verify by hand)
        b_np = np.array([0.0, 0.0], dtype=np.float32)

        self.A_np = A_np
        self.X_np = X_np
        self.W_np = W_np
        self.b_np = b_np

        # PyTorch tensors
        self.A = torch.tensor(A_np)
        self.X = torch.tensor(X_np)

    # ── Individual preprocessing tests ────────────────────────────────────────

    def test_add_self_loops(self):
        """Ã = A + I: diagonal should become 1, off-diagonal unchanged."""
        A_sl = add_self_loops(self.A)
        # Diagonal must all be 1
        diag = torch.diagonal(A_sl)
        assert (diag == 1.0).all(), f"Expected all 1s on diagonal, got {diag}"
        # Off-diagonal unchanged
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    assert A_sl[i, j] == self.A[i, j], f"Off-diagonal changed at ({i},{j})"

    def test_degree_computation(self):
        """Degree = row sum of Ã (should include self-loop contribution)."""
        A_sl = add_self_loops(self.A)
        deg = compute_degree(A_sl)
        # Node 0: edges to 1, 2, + self = degree 3
        # Node 1: edges to 0, 3, + self = degree 3
        # Node 2: edges to 0, 3, + self = degree 3
        # Node 3: edges to 1, 2, 4, + self = degree 4
        # Node 4: edge to 3, + self = degree 2
        expected = torch.tensor([3.0, 3.0, 3.0, 4.0, 2.0])
        assert torch.allclose(deg, expected), f"Got degrees {deg}, expected {expected}"

    def test_symmetric_normalization_shape(self):
        """Normalized adjacency must have shape [N, N]."""
        A_sl = add_self_loops(self.A)
        norm = symmetric_normalize(A_sl)
        assert norm.shape == (self.N, self.N)

    def test_symmetric_normalization_values_node0(self):
        """
        Manually verify Â[0,1].

        Â[i,j] = Ã[i,j] / sqrt(D̃[i,i] * D̃[j,j])
        Â[0,1] = 1 / sqrt(3 * 3) = 1/3 ≈ 0.3333
        """
        A_sl = add_self_loops(self.A)
        norm = symmetric_normalize(A_sl)
        expected_01 = 1.0 / np.sqrt(3.0 * 3.0)  # = 1/3
        assert abs(float(norm[0, 1]) - expected_01) < 1e-6, (
            f"Â[0,1] = {float(norm[0,1]):.6f}, expected {expected_01:.6f}"
        )
        # Â[0,0] = 1/sqrt(3*3) = 1/3 (self-loop)
        expected_00 = 1.0 / np.sqrt(3.0 * 3.0)
        assert abs(float(norm[0, 0]) - expected_00) < 1e-6

    def test_symmetric_normalization_values_node3(self):
        """
        Node 3 has degree 4 (connects to 1, 2, 4, and itself).
        Â[3,4] = 1 / sqrt(4 * 2) = 1/sqrt(8) ≈ 0.3536
        """
        A_sl = add_self_loops(self.A)
        norm = symmetric_normalize(A_sl)
        expected_34 = 1.0 / np.sqrt(4.0 * 2.0)
        assert abs(float(norm[3, 4]) - expected_34) < 1e-6

    # ── Full layer verification ───────────────────────────────────────────────

    def test_gcn_layer_matches_manual_numpy(self):
        """
        Core correctness test: GCNLayerScratch output matches NumPy manual calculation.

        We verify node 0's output, but the test computes ALL nodes
        so we can check the full [N, F_out] output matrix.

        GCN Layer formula:
            H = Â · X · W + bias   (before activation)

        Step-by-step NumPy:
            1. Ã = A + I
            2. D̃ = diag(row_sums(Ã))
            3. Â = D̃^{-0.5} · Ã · D̃^{-0.5}
            4. XW = X @ W              [N, F_out]
            5. H = Â @ XW + bias       [N, F_out]
        """
        # ── NumPy manual computation ──────────────────────────────────────────
        A_sl = self.A_np + np.eye(self.N, dtype=np.float32)
        deg = A_sl.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        # D^{-0.5} @ A_sl @ D^{-0.5} via broadcasting
        norm_adj_np = d_inv_sqrt[:, None] * A_sl * d_inv_sqrt[None, :]

        XW_np = self.X_np @ self.W_np   # [N, F_out]
        H_manual = norm_adj_np @ XW_np + self.b_np  # [N, F_out]

        # ── PyTorch GCNLayerScratch ───────────────────────────────────────────
        layer = GCNLayerScratch(in_features=self.F_in, out_features=self.F_out)

        # Override random weights with our fixed weights so results are comparable.
        with torch.no_grad():
            layer.linear.weight.copy_(torch.tensor(self.W_np.T))  # Linear stores W.T
            layer.linear.bias.copy_(torch.tensor(self.b_np))

        norm_adj_torch = preprocess_adjacency(self.A)
        H_torch = layer(self.X, norm_adj_torch).detach().numpy()

        # ── Assert match ──────────────────────────────────────────────────────
        np.testing.assert_allclose(
            H_torch, H_manual, rtol=1e-4, atol=1e-5,
            err_msg=(
                f"GCNLayerScratch output does not match manual NumPy calculation.\n"
                f"Torch:\n{H_torch}\nManual:\n{H_manual}"
            ),
        )

    def test_node0_output_by_hand(self):
        """
        Spot-check: manually compute node 0's output and verify.

        Node 0's neighbors (in Ã, including self): {0, 1, 2}
        degrees: node0=3, node1=3, node2=3

        Â[0,0] = 1/sqrt(3·3) = 1/3
        Â[0,1] = 1/sqrt(3·3) = 1/3
        Â[0,2] = 1/sqrt(3·3) = 1/3
        Â[0,3] = 0  (no edge 0-3)
        Â[0,4] = 0  (no edge 0-4)

        H[0] = (1/3) · X[0]·W + (1/3) · X[1]·W + (1/3) · X[2]·W + bias

        Let's compute X[i]·W for i in {0,1,2}:
          X[0]·W = [1.0, 0.5, -0.3] @ [[0.5,-0.3],[-0.2,0.7],[0.1,0.4]]
                 = [1·0.5 + 0.5·(-0.2) + (-0.3)·0.1, 1·(-0.3) + 0.5·0.7 + (-0.3)·0.4]
                 = [0.5 - 0.10 - 0.03, -0.3 + 0.35 - 0.12]
                 = [0.37, -0.07]

          X[1]·W = [-1, 0.2, 0.8] @ W
                 = [-1·0.5 + 0.2·(-0.2) + 0.8·0.1, -1·(-0.3) + 0.2·0.7 + 0.8·0.4]
                 = [-0.5 - 0.04 + 0.08, 0.3 + 0.14 + 0.32]
                 = [-0.46, 0.76]

          X[2]·W = [0, -0.4, 1.1] @ W
                 = [0 + (-0.4)·(-0.2) + 1.1·0.1, 0 + (-0.4)·0.7 + 1.1·0.4]
                 = [0.08 + 0.11, -0.28 + 0.44]
                 = [0.19, 0.16]

        H[0] = (1/3)·([0.37,-0.07] + [-0.46,0.76] + [0.19,0.16]) + [0,0]
             = (1/3)·[0.10, 0.85]
             = [0.0333..., 0.2833...]
        """
        expected_node0 = np.array([0.10 / 3.0, 0.85 / 3.0], dtype=np.float32)

        layer = GCNLayerScratch(in_features=self.F_in, out_features=self.F_out)
        with torch.no_grad():
            layer.linear.weight.copy_(torch.tensor(self.W_np.T))
            layer.linear.bias.copy_(torch.tensor(self.b_np))

        norm_adj_torch = preprocess_adjacency(self.A)
        H_torch = layer(self.X, norm_adj_torch).detach().numpy()

        np.testing.assert_allclose(
            H_torch[0], expected_node0, rtol=1e-4, atol=1e-5,
            err_msg=(
                f"Node 0 output mismatch.\n"
                f"Got: {H_torch[0]}\nExpected: {expected_node0}"
            ),
        )
