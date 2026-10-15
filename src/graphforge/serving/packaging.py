"""
Model Packaging, Serialization, and Artifact Bundling
=====================================================
GraphForge Assignment 4: Reproducible GNN MLOps Pipeline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from graphforge.data.cora import CORA_CLASSES
from graphforge.models.cora_models import CoraGAT, CoraGCN, CoraGraphSAGE, CoraMLP

SUPPORTED_ARCHITECTURES = {
    "CoraGCN": CoraGCN,
    "CoraMLP": CoraMLP,
    "CoraGraphSAGE": CoraGraphSAGE,
    "CoraGAT": CoraGAT,
}


def package_model(
    model: nn.Module,
    model_name: str,
    architecture_name: str,
    in_features: int,
    num_classes: int,
    hidden_dim: int,
    output_dir: str | Path,
    version: str = "1.0.0",
    metrics: Optional[Dict[str, float]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Package model weights, architecture configuration, feature schema,
    label mapping, and performance metrics into a self-contained artifact directory.
    """
    output_path = Path(output_dir) / model_name / f"v_{version}"
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Save weights
    weights_path = output_path / "model_weights.pt"
    torch.save(model.state_dict(), weights_path)

    # 2. Save metadata & schema
    metadata = {
        "model_name": model_name,
        "version": version,
        "architecture": architecture_name,
        "feature_schema": {
            "in_features": in_features,
            "feature_type": "binary_bag_of_words",
            "expected_shape": [None, in_features],
        },
        "output_mapping": {
            "num_classes": num_classes,
            "class_names": CORA_CLASSES[:num_classes],
        },
        "hyperparameters": {
            "hidden_dim": hidden_dim,
            "in_features": in_features,
            "num_classes": num_classes,
        },
        "metrics": metrics or {},
        "extra_metadata": extra_metadata or {},
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def load_packaged_model(package_dir: str | Path) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a packaged model from directory, re-instantiate its architecture,
    load weights, and return metadata.
    """
    package_path = Path(package_dir)
    metadata_path = package_path / "metadata.json"
    weights_path = package_path / "model_weights.pt"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found in package directory: {metadata_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found in package directory: {weights_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    arch_name = metadata["architecture"]
    if arch_name not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch_name}'. Supported: {list(SUPPORTED_ARCHITECTURES.keys())}")

    hp = metadata["hyperparameters"]
    arch_cls = SUPPORTED_ARCHITECTURES[arch_name]
    model = arch_cls(
        in_features=hp["in_features"],
        hidden_dim=hp["hidden_dim"],
        num_classes=hp["num_classes"],
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, metadata
