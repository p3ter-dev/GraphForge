"""
Unit Tests: Training Loop
"""

import pytest
import torch

from graphforge.data.synthetic import make_two_community_graph
from graphforge.models.gcn_scratch import TwoLayerGCN, preprocess_adjacency
from graphforge.models.mlp import MLP
from graphforge.training.trainer import TrainingConfig, accuracy, evaluate, train


class TestTrainingLoop:
    """Tests that the training loop trains correctly and respects data hygiene."""

    @pytest.fixture
    def tiny_data(self):
        """Very small graph for fast training tests."""
        return make_two_community_graph(n_nodes=40, seed=42)

    @pytest.fixture
    def tiny_norm_adj(self, tiny_data):
        return preprocess_adjacency(tiny_data["adj"])

    def test_loss_decreases_over_training(self, tiny_data, tiny_norm_adj):
        """
        After 100 epochs of training, training loss should decrease.
        This is a basic sanity check that the optimizer and backprop work.
        """
        model = TwoLayerGCN(in_features=3, hidden_dim=16, n_classes=2, dropout=0.0)
        cfg = TrainingConfig(n_epochs=100, patience=100, seed=0, hidden_dim=16, dropout=0.0)

        model, history = train(
            model, tiny_data["x"], tiny_norm_adj, tiny_data["y"],
            tiny_data["train_mask"], tiny_data["val_mask"], cfg
        )

        initial_loss = history.train_loss[0]
        final_loss = history.train_loss[-1]
        assert final_loss < initial_loss, (
            f"Training loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_train_accuracy_above_chance(self, tiny_data, tiny_norm_adj):
        """
        A trained GCN should achieve > 60% training accuracy on this
        separable synthetic dataset (chance level = 50% for balanced classes).
        """
        model = TwoLayerGCN(in_features=3, hidden_dim=16, n_classes=2, dropout=0.0)
        cfg = TrainingConfig(n_epochs=200, patience=200, seed=0, hidden_dim=16, dropout=0.0)

        model, history = train(
            model, tiny_data["x"], tiny_norm_adj, tiny_data["y"],
            tiny_data["train_mask"], tiny_data["val_mask"], cfg
        )

        final_train_acc = history.train_acc[-1]
        assert final_train_acc > 0.6, (
            f"Training accuracy {final_train_acc:.3f} is not above chance (0.5)."
        )

    def test_test_labels_never_affect_loss(self, tiny_data, tiny_norm_adj):
        """
        Critical data hygiene: test labels must NEVER be used during training.

        We verify this structurally: the train() function only receives
        train_mask and val_mask — the test_mask is never passed to it.
        If you look at the trainer.py code, loss is only computed on
        train_mask nodes. This test makes that explicit.

        Strategy: train two models — one where test labels are
        correctly set, one where test labels are scrambled. If the
        models produce identical training trajectories, test labels
        were not used.
        """
        # Model A: correct labels everywhere
        model_a = TwoLayerGCN(in_features=3, hidden_dim=8, n_classes=2, dropout=0.0)

        # Model B: scramble test labels
        y_scrambled = tiny_data["y"].clone()
        test_mask = tiny_data["test_mask"]
        y_scrambled[test_mask] = 1 - y_scrambled[test_mask]   # flip test labels

        model_b = TwoLayerGCN(in_features=3, hidden_dim=8, n_classes=2, dropout=0.0)
        # Copy same initial weights so comparison is fair
        model_b.load_state_dict(model_a.state_dict())

        cfg = TrainingConfig(n_epochs=50, patience=50, seed=0, hidden_dim=8, dropout=0.0)

        _, hist_a = train(
            model_a, tiny_data["x"], tiny_norm_adj, tiny_data["y"],
            tiny_data["train_mask"], tiny_data["val_mask"], cfg
        )
        _, hist_b = train(
            model_b, tiny_data["x"], tiny_norm_adj, y_scrambled,
            tiny_data["train_mask"], tiny_data["val_mask"], cfg
        )

        # Training loss trajectories must be identical — test labels had no effect.
        for i, (la, lb) in enumerate(zip(hist_a.train_loss, hist_b.train_loss)):
            assert abs(la - lb) < 1e-5, (
                f"Training loss differed at epoch {i}: model_a={la:.6f}, model_b={lb:.6f}\n"
                "This suggests test labels leaked into the training loss."
            )

    def test_early_stopping_triggered(self, tiny_data, tiny_norm_adj):
        """
        With a very small patience, early stopping should trigger before max epochs.
        """
        model = TwoLayerGCN(in_features=3, hidden_dim=8, n_classes=2, dropout=0.0)
        cfg = TrainingConfig(n_epochs=500, patience=5, seed=0, hidden_dim=8, dropout=0.0)

        model, history = train(
            model, tiny_data["x"], tiny_norm_adj, tiny_data["y"],
            tiny_data["train_mask"], tiny_data["val_mask"], cfg
        )

        # Should have stopped well before 500 epochs
        assert len(history.train_loss) < 400, (
            f"Expected early stopping, but ran {len(history.train_loss)} epochs."
        )

    def test_best_checkpoint_is_restored(self, tiny_data, tiny_norm_adj):
        """
        After training, the model should be at the best validation checkpoint,
        not the last epoch's weights.
        Best val accuracy should equal history.best_val_acc.
        """
        model = TwoLayerGCN(in_features=3, hidden_dim=16, n_classes=2, dropout=0.0)
        cfg = TrainingConfig(n_epochs=200, patience=200, seed=42, hidden_dim=16, dropout=0.0)

        model, history = train(
            model, tiny_data["x"], tiny_norm_adj, tiny_data["y"],
            tiny_data["train_mask"], tiny_data["val_mask"], cfg
        )

        # Evaluate the returned model on the val set
        _, actual_val_acc = evaluate(
            model, tiny_data["x"], tiny_norm_adj, tiny_data["y"], tiny_data["val_mask"]
        )

        assert abs(actual_val_acc - history.best_val_acc) < 1e-4, (
            f"Returned model val_acc={actual_val_acc:.4f} != "
            f"history.best_val_acc={history.best_val_acc:.4f}. "
            "Model was not restored to best checkpoint."
        )

    def test_reproducibility_same_seed(self, tiny_data, tiny_norm_adj):
        """Same seed → identical training trajectories."""
        def run(seed):
            model = TwoLayerGCN(in_features=3, hidden_dim=8, n_classes=2, dropout=0.0)
            cfg = TrainingConfig(n_epochs=30, patience=30, seed=seed, hidden_dim=8, dropout=0.0)
            _, hist = train(
                model, tiny_data["x"], tiny_norm_adj, tiny_data["y"],
                tiny_data["train_mask"], tiny_data["val_mask"], cfg
            )
            return hist.train_loss

        losses_a = run(seed=123)
        losses_b = run(seed=123)

        for i, (la, lb) in enumerate(zip(losses_a, losses_b)):
            assert abs(la - lb) < 1e-6, f"Loss diverged at epoch {i}: {la} vs {lb}"


class TestAccuracyFunction:
    """Tests for the accuracy helper function."""

    def test_all_correct(self):
        log_probs = torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]))
        labels = torch.tensor([0, 1, 0])
        mask = torch.ones(3, dtype=torch.bool)
        assert accuracy(log_probs, labels, mask) == 1.0

    def test_all_wrong(self):
        log_probs = torch.log(torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]))
        labels = torch.tensor([0, 1, 0])
        mask = torch.ones(3, dtype=torch.bool)
        assert accuracy(log_probs, labels, mask) == 0.0

    def test_mask_applied(self):
        """Accuracy should only consider masked nodes."""
        log_probs = torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]))
        labels = torch.tensor([0, 0, 0])   # node 1 is wrong
        mask = torch.tensor([True, False, True])   # ignore node 1
        # Nodes 0 and 2 are both predicted correctly
        assert accuracy(log_probs, labels, mask) == 1.0
