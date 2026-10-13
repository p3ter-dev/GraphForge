"""
Correctness Test: Permutation Equivariance
==========================================
Assignment 1, requirement 6.

What is permutation equivariance?
----------------------------------
A GNN is said to be *permutation equivariant* if reordering the nodes
produces correspondingly reordered outputs:

    f(PX, PAP^T) = P · f(X, A)

where P is any permutation matrix (a matrix with exactly one 1 per row
and per column, all others 0).

In plain English:
    "If you shuffle the rows of X and relabel the edges consistently,
     the model's predictions get shuffled in exactly the same way."

Why does this matter?
----------------------
Nodes in a graph have no inherent ordering — node "0" is an arbitrary
label. A correct GNN must produce the same predictions regardless of
how we number the nodes. If it didn't, the model's output would depend
on an arbitrary implementation detail (node numbering) rather than the
true graph structure.

How we test it:
---------------
1. Run model on original (X, A) → get predictions P_orig.
2. Apply a random permutation π to all node indices:
   - X_perm = X[π, :]              (reorder rows)
   - A_perm = A[π, :][:, π]       (reorder rows AND columns)
   - y_perm = y[π]                 (reorder labels)
3. Run model on (X_perm, A_perm) → get predictions P_perm.
4. Assert: P_perm == P_orig[π]     (permuted predictions match original permuted)

Note: We use the SAME trained model weights for both runs.
The permutation affects data layout, not model parameters.
"""

import numpy as np
import pytest
import torch

from graphforge.data.synthetic import make_two_community_graph
from graphforge.models.gcn_scratch import TwoLayerGCN, preprocess_adjacency
from graphforge.training.trainer import TrainingConfig, train


class TestPermutationEquivariance:
    """
    Verify that the GCN from scratch is permutation equivariant.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Build and train a GCN on the synthetic graph."""
        torch.manual_seed(0)
        data = make_two_community_graph(n_nodes=60, seed=0)

        self.x = data["x"]
        self.y = data["y"]
        self.adj = data["adj"]
        self.norm_adj = preprocess_adjacency(self.adj)
        self.train_mask = data["train_mask"]
        self.val_mask = data["val_mask"]
        self.test_mask = data["test_mask"]
        self.N = self.x.shape[0]

        # Train for a short time (just need a non-trivial model, not best accuracy)
        self.model = TwoLayerGCN(
            in_features=3, hidden_dim=16, n_classes=2, dropout=0.0
        )
        cfg = TrainingConfig(n_epochs=100, patience=100, seed=0, hidden_dim=16, dropout=0.0)
        self.model, _ = train(
            self.model, self.x, self.norm_adj, self.y,
            self.train_mask, self.val_mask, cfg
        )
        self.model.eval()

    def _apply_permutation(self, perm: torch.Tensor):
        """
        Apply permutation perm to all graph tensors.

        perm[i] = j means: new position i contains original node j.

        Returns (x_perm, adj_perm, norm_adj_perm, y_perm).
        """
        x_perm = self.x[perm]               # reorder node features
        adj_perm = self.adj[perm][:, perm]  # reorder rows AND columns of adjacency
        norm_adj_perm = preprocess_adjacency(adj_perm)
        y_perm = self.y[perm]               # reorder labels (for sanity check)
        return x_perm, adj_perm, norm_adj_perm, y_perm

    @torch.no_grad()
    def test_equivariance_random_permutation(self):
        """
        After permuting nodes, predictions must permute identically.
        Uses argmax predictions (class indices) for a discrete, robust check.
        """
        # Original predictions
        log_probs_orig = self.model(self.x, self.norm_adj)   # [N, C]
        pred_orig = log_probs_orig.argmax(dim=1)             # [N]

        # Random permutation
        perm = torch.randperm(self.N, generator=torch.Generator().manual_seed(999))
        x_perm, _, norm_adj_perm, y_perm = self._apply_permutation(perm)

        # Predictions after permutation
        log_probs_perm = self.model(x_perm, norm_adj_perm)   # [N, C]
        pred_perm = log_probs_perm.argmax(dim=1)             # [N]

        # Equivariance check: pred_perm[i] should equal pred_orig[perm[i]]
        # Equivalently: pred_perm == pred_orig[perm]
        expected = pred_orig[perm]

        mismatches = (pred_perm != expected).sum().item()
        assert mismatches == 0, (
            f"Equivariance FAILED: {mismatches}/{self.N} nodes have mismatched predictions.\n"
            f"pred_orig: {pred_orig.tolist()}\n"
            f"pred_perm: {pred_perm.tolist()}\n"
            f"expected:  {expected.tolist()}"
        )

    @torch.no_grad()
    def test_equivariance_log_probs_match(self):
        """
        Stronger check: the full log-probability distributions must also match,
        not just the argmax. Tolerates small floating-point differences.
        """
        log_probs_orig = self.model(self.x, self.norm_adj)   # [N, C]

        perm = torch.randperm(self.N, generator=torch.Generator().manual_seed(42))
        x_perm, _, norm_adj_perm, _ = self._apply_permutation(perm)
        log_probs_perm = self.model(x_perm, norm_adj_perm)   # [N, C]

        expected = log_probs_orig[perm]  # permuted original output

        assert torch.allclose(log_probs_perm, expected, rtol=1e-3, atol=1e-4), (
            "Log-probability distributions do not match after permutation.\n"
            f"Max absolute difference: {(log_probs_perm - expected).abs().max():.6f}"
        )

    @torch.no_grad()
    def test_equivariance_identity_permutation(self):
        """
        Sanity check: identity permutation should reproduce identical outputs.
        """
        identity = torch.arange(self.N)
        x_perm, _, norm_adj_perm, _ = self._apply_permutation(identity)
        log_probs_orig = self.model(self.x, self.norm_adj)
        log_probs_perm = self.model(x_perm, norm_adj_perm)

        assert torch.allclose(log_probs_orig, log_probs_perm, rtol=1e-5, atol=1e-6), (
            "Identity permutation produced different outputs — something is fundamentally wrong."
        )

    @torch.no_grad()
    def test_adjacency_reordering_is_consistent(self):
        """
        Verify that adj[π, :][:, π] correctly relabels edges.
        Edge (i, j) in original → edge (π⁻¹[i], π⁻¹[j]) in permuted.
        We check a specific edge pair.
        """
        perm = torch.tensor([2, 0, 1, 4, 3] + list(range(5, self.N)))
        _, adj_perm, _, _ = self._apply_permutation(perm)

        # For every original edge (i, j), the permuted adj should have an edge
        # at (where perm maps i, where perm maps j).
        # We build an inverse permutation: inv_perm[perm[i]] = i
        inv_perm = torch.argsort(perm)

        for i in range(self.N):
            for j in range(self.N):
                if self.adj[i, j] > 0:
                    new_i = inv_perm[i].item()
                    new_j = inv_perm[j].item()
                    assert adj_perm[new_i, new_j] > 0, (
                        f"Edge ({i},{j}) should appear at ({new_i},{new_j}) after permutation."
                    )
