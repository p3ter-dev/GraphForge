"""
Unit Tests for MLOps: Packaging, Registry, Promotion Gate, and Drift Monitoring
==============================================================================
"""

import pytest
import torch
from pathlib import Path

from graphforge.models.cora_models import CoraGCN
from graphforge.serving.packaging import package_model, load_packaged_model
from graphforge.serving.registry import ModelRegistry, ModelRegistrationRecord
from graphforge.monitoring.drift import (
    compute_graph_structure_metrics,
    compute_embedding_distribution_metrics,
    detect_distribution_drift,
)


@pytest.fixture
def tmp_m4_dir(tmp_path):
    return tmp_path / "mlops_test"


class TestModelPackaging:

    def test_package_and_reload_exact_reproducibility(self, tmp_m4_dir):
        """Packaging model and reloading must yield identical tensor predictions."""
        torch.manual_seed(42)
        model = CoraGCN(in_features=10, hidden_dim=16, num_classes=3)
        model.eval()

        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

        with torch.no_grad():
            out_before = model(x, edge_index)

        # Package
        pkg_dir = package_model(
            model=model,
            model_name="test_gcn",
            architecture_name="CoraGCN",
            in_features=10,
            num_classes=3,
            hidden_dim=16,
            output_dir=tmp_m4_dir,
            version="0.1.0",
            metrics={"val_acc": 0.85},
        )

        # Reload
        reloaded, meta = load_packaged_model(pkg_dir)
        with torch.no_grad():
            out_after = reloaded(x, edge_index)

        diff = (out_before - out_after).abs().max().item()
        assert diff < 1e-6, f"Reloaded model outputs diverged: {diff}"
        assert meta["version"] == "0.1.0"
        assert meta["architecture"] == "CoraGCN"


class TestPromotionGate:

    def test_first_model_is_automatically_promoted(self, tmp_m4_dir):
        reg = ModelRegistry(registry_file=tmp_m4_dir / "reg.json")
        rec = ModelRegistrationRecord(
            version="1.0.0",
            model_name="gcn",
            architecture="CoraGCN",
            package_dir=str(tmp_m4_dir),
            metrics={"val_acc": 0.80},
            status="candidate",
            registered_at="",
        )
        promoted, msg = reg.evaluate_and_promote(rec, metric_name="val_acc", threshold_delta=0.005)
        assert promoted is True
        assert reg.get_champion()["version"] == "1.0.0"

    def test_promotion_gate_blocks_regression(self, tmp_m4_dir):
        reg = ModelRegistry(registry_file=tmp_m4_dir / "reg.json")
        # Champion: 0.80
        rec1 = ModelRegistrationRecord("1.0.0", "gcn", "CoraGCN", str(tmp_m4_dir), {"val_acc": 0.80}, "candidate", "")
        reg.evaluate_and_promote(rec1, metric_name="val_acc")

        # Candidate with inferior performance: 0.78
        rec_inferior = ModelRegistrationRecord("1.0.1", "gcn", "CoraGCN", str(tmp_m4_dir), {"val_acc": 0.78}, "candidate", "")
        promoted, msg = reg.evaluate_and_promote(rec_inferior, metric_name="val_acc", threshold_delta=0.005)
        assert promoted is False
        assert "Blocked Regression" in msg
        assert reg.get_champion()["version"] == "1.0.0"

    def test_tie_leaves_champion_in_place(self, tmp_m4_dir):
        reg = ModelRegistry(registry_file=tmp_m4_dir / "reg.json")
        rec1 = ModelRegistrationRecord("1.0.0", "gcn", "CoraGCN", str(tmp_m4_dir), {"val_acc": 0.80}, "candidate", "")
        reg.evaluate_and_promote(rec1, metric_name="val_acc")

        # Tie
        rec_tie = ModelRegistrationRecord("1.0.1", "gcn", "CoraGCN", str(tmp_m4_dir), {"val_acc": 0.80}, "candidate", "")
        promoted, msg = reg.evaluate_and_promote(rec_tie, metric_name="val_acc", threshold_delta=0.005)
        assert promoted is False
        assert reg.get_champion()["version"] == "1.0.0"

    def test_missing_metric_rejected(self, tmp_m4_dir):
        reg = ModelRegistry(registry_file=tmp_m4_dir / "reg.json")
        rec = ModelRegistrationRecord("1.0.0", "gcn", "CoraGCN", str(tmp_m4_dir), {"loss": 0.5}, "candidate", "")
        promoted, msg = reg.evaluate_and_promote(rec, metric_name="val_acc")
        assert promoted is False
        assert "missing from candidate" in msg

    def test_rollback_restores_previous_champion(self, tmp_m4_dir):
        reg = ModelRegistry(registry_file=tmp_m4_dir / "reg.json")
        rec1 = ModelRegistrationRecord("1.0.0", "gcn", "CoraGCN", str(tmp_m4_dir), {"val_acc": 0.80}, "candidate", "")
        reg.evaluate_and_promote(rec1, metric_name="val_acc")

        rec2 = ModelRegistrationRecord("1.1.0", "gcn", "CoraGCN", str(tmp_m4_dir), {"val_acc": 0.85}, "candidate", "")
        reg.evaluate_and_promote(rec2, metric_name="val_acc")
        assert reg.get_champion()["version"] == "1.1.0"

        # Rollback
        success, msg = reg.rollback()
        assert success is True
        assert reg.get_champion()["version"] == "1.0.0"


class TestDriftMonitoring:

    def test_graph_structure_metrics(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        metrics = compute_graph_structure_metrics(edge_index, num_nodes=4)
        assert metrics["num_nodes"] == 4.0
        assert metrics["num_edges_directed"] == 3.0
        assert metrics["avg_degree"] == 0.75
        assert metrics["isolated_node_ratio"] == 0.25  # node 3 is isolated

    def test_embedding_drift_detection(self):
        torch.manual_seed(42)
        base = torch.randn(100, 32)
        # Identical distribution -> no drift
        res_normal = detect_distribution_drift(base, base.clone(), alpha=0.05, wasserstein_threshold=0.30)
        assert res_normal["drift_detected"] is False

        # Shifted distribution (shifted mean by 5.0) -> drift detected
        shifted = base + 5.0
        res_drift = detect_distribution_drift(base, shifted, alpha=0.05, wasserstein_threshold=0.30)
        assert res_drift["drift_detected"] is True
