"""
Automated Data Leakage and Decoder Symmetry Tests for Link Prediction
=====================================================================
Assignment 3 Requirement 9:
Automated tests proving that held-out positive links and their reverse copies
are strictly absent from the message-passing graph.
"""

import pytest
import torch

from graphforge.data.link_split import create_link_prediction_splits
from graphforge.models.link_prediction import (
    DotProductDecoder,
    SymmetricMLPDecoder,
)


@pytest.fixture
def sample_splits():
    torch.manual_seed(42)
    N = 100
    F_in = 32
    x = torch.randn(N, F_in)

    # Random connected graph
    src = torch.randint(0, N, (300,))
    dst = torch.randint(0, N, (300,))
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    splits = create_link_prediction_splits(edge_index, x, val_ratio=0.10, test_ratio=0.15, seed=42)
    return splits


class TestDataLeakageGuards:

    def test_held_out_positive_edges_absent_from_message_passing(self, sample_splits):
        """
        CRITICAL LEAKAGE TEST:
        Neither held-out validation edges nor held-out test edges (nor their reverse copies)
        may appear in the message-passing graph E_train!
        """
        msg_edges = sample_splits.train_message_edge_index.cpu().numpy()
        msg_edge_set = set(zip(msg_edges[0], msg_edges[1]))

        # Test validation positives
        val_pos = sample_splits.val_pos_edge_index.cpu().numpy()
        for u, v in zip(val_pos[0], val_pos[1]):
            assert (u, v) not in msg_edge_set, f"LEAKAGE DETECTED: Val link ({u}, {v}) found in message passing graph!"
            assert (v, u) not in msg_edge_set, f"LEAKAGE DETECTED: Reverse val link ({v}, {u}) found in message passing graph!"

        # Test test positives
        test_pos = sample_splits.test_pos_edge_index.cpu().numpy()
        for u, v in zip(test_pos[0], test_pos[1]):
            assert (u, v) not in msg_edge_set, f"LEAKAGE DETECTED: Test link ({u}, {v}) found in message passing graph!"
            assert (v, u) not in msg_edge_set, f"LEAKAGE DETECTED: Reverse test link ({v}, {u}) found in message passing graph!"

    def test_positive_splits_have_zero_overlap(self, sample_splits):
        """Train, val, and test positive splits must be mutually disjoint."""
        def to_canonical_set(edges):
            arr = edges.cpu().numpy()
            return set((min(u, v), max(u, v)) for u, v in zip(arr[0], arr[1]))

        train_set = to_canonical_set(sample_splits.train_pos_edge_index)
        val_set = to_canonical_set(sample_splits.val_pos_edge_index)
        test_set = to_canonical_set(sample_splits.test_pos_edge_index)

        assert len(train_set.intersection(val_set)) == 0, "Train and Val positive overlap!"
        assert len(train_set.intersection(test_set)) == 0, "Train and Test positive overlap!"
        assert len(val_set.intersection(test_set)) == 0, "Val and Test positive overlap!"

    def test_negative_pairs_never_contain_observed_edges(self, sample_splits):
        """Negatives must strictly exclude all observed positive edges."""
        msg_edges = sample_splits.train_message_edge_index.cpu().numpy()
        all_observed_set = set((min(u, v), max(u, v)) for u, v in zip(msg_edges[0], msg_edges[1]))

        # Include val and test in all observed
        for pos in [sample_splits.val_pos_edge_index, sample_splits.test_pos_edge_index]:
            arr = pos.cpu().numpy()
            for u, v in zip(arr[0], arr[1]):
                all_observed_set.add((min(u, v), max(u, v)))

        for neg in [sample_splits.train_neg_edge_index, sample_splits.val_neg_edge_index, sample_splits.test_neg_edge_index]:
            arr = neg.cpu().numpy()
            for u, v in zip(arr[0], arr[1]):
                assert u != v, f"Negative sample contains self-loop ({u}, {v})"
                pair = (min(u, v), max(u, v))
                assert pair not in all_observed_set, f"Negative sample ({u}, {v}) is an actual positive edge!"

    def test_negative_splits_mutually_disjoint(self, sample_splits):
        """Evaluation negative pairs must not overlap between splits."""
        def to_set(tensor):
            arr = tensor.cpu().numpy()
            return set((min(u, v), max(u, v)) for u, v in zip(arr[0], arr[1]))

        train_neg = to_set(sample_splits.train_neg_edge_index)
        val_neg = to_set(sample_splits.val_neg_edge_index)
        test_neg = to_set(sample_splits.test_neg_edge_index)

        assert len(train_neg.intersection(val_neg)) == 0
        assert len(train_neg.intersection(test_neg)) == 0
        assert len(val_neg.intersection(test_neg)) == 0


class TestDecoderSymmetry:

    def test_symmetric_mlp_decoder_invariant_to_pair_order(self):
        """
        Requirement 5: Reversing an undirected node pair must produce exactly the same score.
        """
        torch.manual_seed(42)
        N = 20
        emb_dim = 16
        z = torch.randn(N, emb_dim)

        decoder = SymmetricMLPDecoder(emb_dim=emb_dim, hidden_dim=32)
        decoder.eval()

        u = torch.tensor([0, 5, 12, 19])
        v = torch.tensor([3, 8, 15, 1])

        forward_pairs = torch.stack([u, v], dim=0)
        reverse_pairs = torch.stack([v, u], dim=0)

        with torch.no_grad():
            scores_fwd = decoder(z, forward_pairs)
            scores_rev = decoder(z, reverse_pairs)

        diff = (scores_fwd - scores_rev).abs().max().item()
        assert diff < 1e-6, f"SymmetricMLPDecoder symmetry violated! Diff: {diff}"

    def test_dot_product_decoder_symmetry(self):
        """Dot product decoder is also symmetric."""
        z = torch.randn(10, 16)
        decoder = DotProductDecoder()
        pairs_fwd = torch.tensor([[0, 2], [1, 3]])
        # Reversing (0, 1) and (2, 3) means row 0 and row 1 are swapped: [[1, 3], [0, 2]]
        pairs_rev = torch.stack([pairs_fwd[1], pairs_fwd[0]], dim=0)

        s_fwd = decoder(z, pairs_fwd)
        s_rev = decoder(z, pairs_rev)
        assert torch.allclose(s_fwd, s_rev, atol=1e-6)
