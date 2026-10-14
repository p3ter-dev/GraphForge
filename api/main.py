"""
GraphForge FastAPI Serving Application
======================================
Production-grade GNN inference service for Node Classification and Link Prediction.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from graphforge.serving.registry import ModelRegistry
from graphforge.serving.packaging import load_packaged_model
from graphforge.data.cora import CORA_CLASSES


app = FastAPI(
    title="GraphForge GNN Model Service",
    description="Production API serving Graph Neural Networks for node classification and link prediction.",
    version="1.0.0",
)

# Global model state
SERVICE_STATE: Dict[str, Any] = {
    "model": None,
    "metadata": None,
    "start_time": time.time(),
}


def get_active_model() -> Tuple[Any, Dict[str, Any]]:
    """Retrieve currently active champion model."""
    if SERVICE_STATE["model"] is not None:
        return SERVICE_STATE["model"], SERVICE_STATE["metadata"]

    registry = ModelRegistry()
    champ = registry.get_champion()

    if champ is not None and Path(champ["package_dir"]).exists():
        model, meta = load_packaged_model(champ["package_dir"])
        SERVICE_STATE["model"] = model
        SERVICE_STATE["metadata"] = meta
        return model, meta

    # Fallback search in artifacts
    art_dir = Path("artifacts")
    pt_files = list(art_dir.glob("**/model_weights.pt"))
    if pt_files:
        pkg_dir = pt_files[0].parent
        model, meta = load_packaged_model(pkg_dir)
        SERVICE_STATE["model"] = model
        SERVICE_STATE["metadata"] = meta
        return model, meta

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="No approved model available in registry or artifact store. Train and register a model first.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Request and Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class NodePredictRequest(BaseModel):
    x: List[List[float]] = Field(..., description="Node feature matrix [N, in_features]")
    edge_index: List[List[int]] = Field(..., description="Graph edge index [2, M] (directed edges)")
    target_node_ids: List[int] = Field(..., description="Target node IDs to return predictions for")


class NodePredictionItem(BaseModel):
    node_id: int
    predicted_class_id: int
    predicted_class_name: str
    probabilities: List[float]


class NodePredictResponse(BaseModel):
    predictions: List[NodePredictionItem]
    model_version: str
    model_architecture: str


class LinkPairItem(BaseModel):
    u: int
    v: int


class LinkPredictRequest(BaseModel):
    x: List[List[float]] = Field(..., description="Node feature matrix [N, in_features]")
    edge_index: List[List[int]] = Field(..., description="Graph edge index [2, M]")
    pairs: List[List[int]] = Field(..., description="List of [u, v] node pairs to score")


class LinkPredictionItem(BaseModel):
    u: int
    v: int
    link_probability: float
    predicted_link: bool


class LinkPredictResponse(BaseModel):
    predictions: List[LinkPredictionItem]
    model_version: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    uptime_sec = time.time() - SERVICE_STATE["start_time"]
    try:
        model, meta = get_active_model()
        model_loaded = True
        version = meta.get("version", "unknown")
        arch = meta.get("architecture", "unknown")
    except Exception:
        model_loaded = False
        version = "none"
        arch = "none"

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "active_model_version": version,
        "architecture": arch,
        "uptime_seconds": round(uptime_sec, 2),
    }


@app.get("/model/metadata", tags=["Model Governance"])
def get_model_metadata() -> Dict[str, Any]:
    _, meta = get_active_model()
    return meta


@app.post("/predict/node", response_model=NodePredictResponse, tags=["Inference"])
def predict_node(req: NodePredictRequest) -> NodePredictResponse:
    model, meta = get_active_model()

    N = len(req.x)
    if N == 0:
        raise HTTPException(status_code=422, detail="Feature matrix x cannot be empty.")

    # 1. Feature Dimension Validation
    expected_dim = meta["feature_schema"]["in_features"]
    for idx, row in enumerate(req.x):
        if len(row) != expected_dim:
            raise HTTPException(
                status_code=422,
                detail=f"Feature dimension mismatch at node {idx}: expected {expected_dim}, got {len(row)}.",
            )

    # 2. Edge Index Validation
    if len(req.edge_index) != 2:
        raise HTTPException(status_code=422, detail="edge_index must have shape [2, M].")

    src, dst = req.edge_index[0], req.edge_index[1]
    if len(src) != len(dst):
        raise HTTPException(status_code=422, detail="edge_index source and destination lengths must match.")

    for i, (u, v) in enumerate(zip(src, dst)):
        if u < 0 or u >= N or v < 0 or v >= N:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid edge endpoint at index {i}: ({u}, {v}) references non-existent node outside [0, {N-1}].",
            )

    # 3. Target Node Validation
    for target in req.target_node_ids:
        if target < 0 or target >= N:
            raise HTTPException(
                status_code=422,
                detail=f"Target node ID {target} is unknown (outside valid range [0, {N-1}]).",
            )

    # Convert to Tensors
    x_tensor = torch.tensor(req.x, dtype=torch.float32)
    edge_tensor = torch.tensor(req.edge_index, dtype=torch.long)

    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(x_tensor, edge_tensor)
        probs = F.softmax(out, dim=-1)

    class_names = meta["output_mapping"]["class_names"]
    results = []

    for target in req.target_node_ids:
        node_probs = probs[target].tolist()
        pred_cls = int(torch.argmax(probs[target]).item())
        results.append(
            NodePredictionItem(
                node_id=target,
                predicted_class_id=pred_cls,
                predicted_class_name=class_names[pred_cls] if pred_cls < len(class_names) else f"Class_{pred_cls}",
                probabilities=[round(p, 4) for p in node_probs],
            )
        )

    return NodePredictResponse(
        predictions=results,
        model_version=meta.get("version", "1.0.0"),
        model_architecture=meta.get("architecture", "CoraGCN"),
    )


@app.post("/predict/link", response_model=LinkPredictResponse, tags=["Inference"])
def predict_link(req: LinkPredictRequest) -> LinkPredictResponse:
    model, meta = get_active_model()
    N = len(req.x)
    if N == 0:
        raise HTTPException(status_code=422, detail="Feature matrix x cannot be empty.")

    # Validation
    for p in req.pairs:
        if len(p) != 2:
            raise HTTPException(status_code=422, detail="Each pair must contain exactly 2 node IDs [u, v].")
        u, v = p[0], p[1]
        if u < 0 or u >= N or v < 0 or v >= N:
            raise HTTPException(status_code=422, detail=f"Pair [{u}, {v}] contains out-of-bounds node index for N={N}.")

    x_tensor = torch.tensor(req.x, dtype=torch.float32)
    edge_tensor = torch.tensor(req.edge_index, dtype=torch.long)

    # Compute node embeddings
    model.eval()
    with torch.no_grad():
        if hasattr(model, "get_embeddings"):
            z = model.get_embeddings(x_tensor, edge_tensor)
        else:
            z = model(x_tensor, edge_tensor)

        pairs_u = [p[0] for p in req.pairs]
        pairs_v = [p[1] for p in req.pairs]
        z_u = z[pairs_u]
        z_v = z[pairs_v]

        # Inner product link scoring
        scores = (z_u * z_v).sum(dim=-1)
        probs = torch.sigmoid(scores).tolist()

    results = []
    for (u, v), prob in zip(req.pairs, probs):
        results.append(
            LinkPredictionItem(
                u=u,
                v=v,
                link_probability=round(float(prob), 4),
                predicted_link=bool(prob >= 0.5),
            )
        )

    return LinkPredictResponse(
        predictions=results,
        model_version=meta.get("version", "1.0.0"),
    )
