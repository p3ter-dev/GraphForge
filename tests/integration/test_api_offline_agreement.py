"""
Integration Tests: Offline vs API Agreement & Error Handling
============================================================
Assignment 4:
- Verify offline Python predictions and FastAPI responses agree when inputs are identical.
- Test edge cases: isolated nodes, duplicate target IDs, invalid edge endpoints,
  incorrect feature dimensions, unknown target IDs, and model-loading failures.
"""

import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from api.main import SERVICE_STATE, app
from graphforge.data.cora import CORA_CLASSES
from graphforge.models.cora_models import CoraGCN
from graphforge.serving.packaging import package_model
from graphforge.serving.registry import ModelRegistry


@pytest.fixture(scope="module")
def setup_test_model(tmp_path_factory):
    """Train a tiny model, package it, and register as champion for API tests."""
    temp_dir = tmp_path_factory.mktemp("test_m4_artifacts")
    torch.manual_seed(42)

    in_feat = 1433
    num_classes = 7
    model = CoraGCN(in_features=in_feat, hidden_dim=32, num_classes=num_classes)
    model.eval()

    pkg_dir = package_model(
        model=model,
        model_name="cora_gcn",
        architecture_name="CoraGCN",
        in_features=in_feat,
        num_classes=num_classes,
        hidden_dim=32,
        output_dir=temp_dir / "packaged",
        version="1.0.0-test",
        metrics={"val_acc": 0.82},
    )

    reg_file = temp_dir / "registry.json"
    registry = ModelRegistry(registry_file=reg_file)
    registry.data["current_champion"] = "1.0.0-test"
    registry.data["models"]["1.0.0-test"] = {
        "version": "1.0.0-test",
        "model_name": "cora_gcn",
        "architecture": "CoraGCN",
        "package_dir": str(pkg_dir),
        "metrics": {"val_acc": 0.82},
        "status": "champion",
        "registered_at": "2026-10-14T12:00:00Z",
    }
    registry._save()

    # Inject into API state
    SERVICE_STATE["model"] = model
    SERVICE_STATE["metadata"] = {
        "model_name": "cora_gcn",
        "version": "1.0.0-test",
        "architecture": "CoraGCN",
        "feature_schema": {"in_features": in_feat},
        "output_mapping": {"num_classes": 7, "class_names": CORA_CLASSES},
        "hyperparameters": {"hidden_dim": 32, "in_features": in_feat, "num_classes": 7},
    }

    return model, in_feat, num_classes


class TestOfflineApiAgreement:

    def test_offline_and_api_predictions_match_exactly(self, setup_test_model):
        """
        Requirement 18: Verify offline and API predictions agree when preprocessing,
        graph context, model version, and inputs are identical.
        """
        model, in_feat, num_classes = setup_test_model
        client = TestClient(app)

        # Create sample graph: 5 nodes, 4 edges
        torch.manual_seed(123)
        N = 5
        x = torch.randn(N, in_feat)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        target_ids = [0, 2, 4]

        # 1. Offline forward pass
        model.eval()
        with torch.no_grad():
            out_offline = model(x, edge_index)
            probs_offline = F.softmax(out_offline, dim=-1).cpu().numpy()

        # 2. API Call
        payload = {
            "x": x.tolist(),
            "edge_index": edge_index.tolist(),
            "target_node_ids": target_ids,
        }
        res = client.post("/predict/node", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()

        assert data["model_version"] == "1.0.0-test"
        assert len(data["predictions"]) == len(target_ids)

        # 3. Check exact numerical agreement (< 1e-4)
        for item in data["predictions"]:
            nid = item["node_id"]
            api_probs = item["probabilities"]
            expected_probs = probs_offline[nid]

            for c in range(num_classes):
                diff = abs(api_probs[c] - expected_probs[c])
                assert diff < 1e-4, f"Node {nid} class {c} probability diverged: API={api_probs[c]}, Offline={expected_probs[c]}"


class TestApiEdgeCasesAndErrors:

    def test_isolated_node_handling(self, setup_test_model):
        """Node with 0 incident edges must be classified gracefully."""
        client = TestClient(app)
        x = [[0.1] * 1433, [0.2] * 1433]
        # Empty graph (0 edges)
        payload = {
            "x": x,
            "edge_index": [[], []],
            "target_node_ids": [0, 1],
        }
        res = client.post("/predict/node", json=payload)
        assert res.status_code == 200
        preds = res.json()["predictions"]
        assert len(preds) == 2

    def test_duplicate_target_ids(self, setup_test_model):
        """Duplicate target IDs in request return predictions for each requested ID."""
        client = TestClient(app)
        x = [[0.1] * 1433, [0.2] * 1433]
        payload = {
            "x": x,
            "edge_index": [[0], [1]],
            "target_node_ids": [1, 1, 0],
        }
        res = client.post("/predict/node", json=payload)
        assert res.status_code == 200
        preds = res.json()["predictions"]
        assert len(preds) == 3
        assert preds[0]["node_id"] == 1
        assert preds[1]["node_id"] == 1
        assert preds[2]["node_id"] == 0

    def test_incorrect_feature_dimensions(self, setup_test_model):
        """HTTP 422 when feature dimension != 1433."""
        client = TestClient(app)
        x = [[0.1] * 10]  # Only 10 features instead of 1433
        payload = {
            "x": x,
            "edge_index": [[], []],
            "target_node_ids": [0],
        }
        res = client.post("/predict/node", json=payload)
        assert res.status_code == 422
        assert "Feature dimension mismatch" in res.json()["detail"]

    def test_invalid_edge_endpoints(self, setup_test_model):
        """HTTP 422 when edge references non-existent node."""
        client = TestClient(app)
        x = [[0.1] * 1433, [0.2] * 1433]  # N=2 (valid IDs: 0, 1)
        payload = {
            "x": x,
            "edge_index": [[0], [99]],  # 99 is invalid!
            "target_node_ids": [0],
        }
        res = client.post("/predict/node", json=payload)
        assert res.status_code == 422
        assert "Invalid edge endpoint" in res.json()["detail"]

    def test_unknown_target_id(self, setup_test_model):
        """HTTP 422 when target node ID is out of bounds."""
        client = TestClient(app)
        x = [[0.1] * 1433, [0.2] * 1433]  # N=2
        payload = {
            "x": x,
            "edge_index": [[0], [1]],
            "target_node_ids": [5],  # 5 is unknown!
        }
        res = client.post("/predict/node", json=payload)
        assert res.status_code == 422
        assert "Target node ID 5 is unknown" in res.json()["detail"]
