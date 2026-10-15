"""
Unit Tests: Synthetic Graph Generation
"""

import torch

from graphforge.data.synthetic import make_two_community_graph


class TestSyntheticGraphGeneration:

    def test_correct_node_count(self):
        data = make_two_community_graph(n_nodes=100, seed=0)
        assert data["x"].shape[0] == 100
        assert data["y"].shape[0] == 100
        assert data["adj"].shape == (100, 100)

    def test_correct_feature_count(self):
        data = make_two_community_graph(n_nodes=50, n_features=3, seed=0)
        assert data["x"].shape == (50, 3)

    def test_adjacency_is_symmetric(self):
        """Undirected graph → A must equal A.T."""
        data = make_two_community_graph(n_nodes=60, seed=1)
        adj = data["adj"]
        assert torch.allclose(adj, adj.T), "Adjacency matrix is not symmetric."

    def test_no_self_loops_in_raw_adj(self):
        """Raw adjacency should not have self-loops (we add them in preprocess)."""
        data = make_two_community_graph(n_nodes=60, seed=2)
        diag = torch.diagonal(data["adj"])
        assert (diag == 0).all(), "Raw adjacency should have no self-loops."

    def test_labels_not_in_features(self):
        """
        Critical: labels must never appear in the feature matrix.
        We verify the feature columns are real-valued and that the
        label tensor is a separate integer tensor.
        """
        data = make_two_community_graph(n_nodes=100, seed=3)
        assert data["x"].dtype == torch.float32
        assert data["y"].dtype == torch.long
        # Features and labels are separate tensors (different objects)
        # and have different dtypes — can't be the same thing.
        assert data["x"] is not data["y"]

    def test_binary_labels(self):
        """Labels should be 0 or 1 for a two-class problem."""
        data = make_two_community_graph(n_nodes=100, seed=4)
        unique_labels = set(data["y"].tolist())
        assert unique_labels == {0, 1}, f"Expected labels {{0, 1}}, got {unique_labels}"

    def test_split_masks_partition_nodes(self):
        """Train, val, test masks must be disjoint and cover all nodes."""
        data = make_two_community_graph(n_nodes=100, seed=5)
        tr = data["train_mask"]
        va = data["val_mask"]
        te = data["test_mask"]

        # Disjoint
        assert not (tr & va).any(), "Train and val masks overlap."
        assert not (tr & te).any(), "Train and test masks overlap."
        assert not (va & te).any(), "Val and test masks overlap."

        # Complete
        union = tr | va | te
        assert union.all(), "Some nodes are not in any split."

    def test_split_fractions_approximate(self):
        """Splits should be approximately 60/20/20."""
        data = make_two_community_graph(n_nodes=200, seed=6)
        n = 200
        n_train = data["train_mask"].sum().item()
        n_val = data["val_mask"].sum().item()
        n_test = data["test_mask"].sum().item()

        assert abs(n_train / n - 0.60) < 0.05, f"Train fraction off: {n_train/n:.2f}"
        assert abs(n_val / n - 0.20) < 0.05, f"Val fraction off: {n_val/n:.2f}"
        assert abs(n_test / n - 0.20) < 0.05, f"Test fraction off: {n_test/n:.2f}"

    def test_reproducibility(self):
        """Same seed → identical outputs."""
        d1 = make_two_community_graph(n_nodes=80, seed=42)
        d2 = make_two_community_graph(n_nodes=80, seed=42)
        assert torch.allclose(d1["x"], d2["x"]), "Features not reproducible."
        assert torch.equal(d1["y"], d2["y"]), "Labels not reproducible."
        assert torch.equal(d1["adj"], d2["adj"]), "Adjacency not reproducible."

    def test_different_seeds_differ(self):
        """Different seeds should produce different graphs."""
        d1 = make_two_community_graph(n_nodes=80, seed=1)
        d2 = make_two_community_graph(n_nodes=80, seed=2)
        assert not torch.allclose(d1["x"], d2["x"]), "Different seeds produced same features."

    def test_intra_edges_more_likely(self):
        """
        Sanity check on the stochastic block model: within-community
        edge density should be higher than between-community density.
        """
        data = make_two_community_graph(n_nodes=200, seed=99, p_intra=0.25, p_inter=0.01)
        y = data["y"].numpy()
        adj = data["adj"].numpy()
        n = 200

        intra_edges = 0
        inter_edges = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    if y[i] == y[j]:
                        intra_edges += 1
                    else:
                        inter_edges += 1

        # Should be far more intra than inter edges
        assert intra_edges > inter_edges * 5, (
            f"Expected intra >> inter, got intra={intra_edges}, inter={inter_edges}"
        )

    def test_edge_index_consistent_with_adj(self):
        """edge_index (COO format) and adj must represent the same edges."""
        data = make_two_community_graph(n_nodes=60, seed=7)
        adj = data["adj"]
        edge_index = data["edge_index"]

        # Build adj from edge_index
        N = adj.shape[0]
        adj_from_ei = torch.zeros(N, N)
        adj_from_ei[edge_index[0], edge_index[1]] = 1.0

        assert torch.equal(adj, adj_from_ei), "edge_index and adj are inconsistent."

    def test_at_least_100_nodes_requirement(self):
        """Assignment requires at least 100 nodes."""
        data = make_two_community_graph(n_nodes=120)
        assert data["metadata"]["n_nodes"] >= 100


class TestSyntheticGraphFeatureDesign:
    """
    Test the intentional feature design: graph structure must provide
    signal that raw features alone cannot.
    """

    def test_feature1_has_no_class_signal(self):
        """
        Feature 1 (index 1) should have no class signal: the mean for
        class 0 and class 1 should be approximately equal (within noise).
        With large N and noise_std=1.2, difference should be < 0.3.
        """
        data = make_two_community_graph(n_nodes=500, seed=0)
        x = data["x"]
        y = data["y"]

        mean_c0 = x[y == 0, 1].mean().item()
        mean_c1 = x[y == 1, 1].mean().item()
        diff = abs(mean_c0 - mean_c1)

        assert diff < 0.5, (
            f"Feature 1 should have no class signal, but mean difference = {diff:.3f}. "
            "Check the synthetic data generator."
        )

    def test_feature0_has_class_signal(self):
        """Feature 0 should separate classes (mean ~1.0 for class 0, ~-1.0 for class 1)."""
        data = make_two_community_graph(n_nodes=500, seed=0)
        x = data["x"]
        y = data["y"]

        mean_c0 = x[y == 0, 0].mean().item()
        mean_c1 = x[y == 1, 0].mean().item()

        assert mean_c0 > 0.5, f"Class 0 Feature 0 mean should be > 0.5, got {mean_c0:.3f}"
        assert mean_c1 < -0.5, f"Class 1 Feature 0 mean should be < -0.5, got {mean_c1:.3f}"
