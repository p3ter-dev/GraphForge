"""
Evaluation utilities for node classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

try:
    from sklearn.metrics import f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@torch.no_grad()
def node_accuracy(
    model: nn.Module,
    x: Tensor,
    norm_adj: Tensor,
    y: Tensor,
    mask: Tensor,
) -> float:
    """Classification accuracy on masked nodes."""
    model.eval()
    log_probs = model(x, norm_adj)
    pred = log_probs[mask].argmax(dim=1)
    return float((pred == y[mask]).sum()) / float(mask.sum())


@torch.no_grad()
def node_f1(
    model: nn.Module,
    x: Tensor,
    norm_adj: Tensor,
    y: Tensor,
    mask: Tensor,
    average: str = "macro",
) -> float:
    """Macro (or specified) F1 score on masked nodes. Requires scikit-learn."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for F1 computation.")
    model.eval()
    log_probs = model(x, norm_adj)
    pred = log_probs[mask].argmax(dim=1).cpu().numpy()
    true = y[mask].cpu().numpy()
    return float(f1_score(true, pred, average=average, zero_division=0))


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    x: Tensor,
    norm_adj: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Get predicted class and probability for all nodes.

    Returns
    -------
    (pred_classes [N], probs [N, C])
    """
    model.eval()
    log_probs = model(x, norm_adj)
    probs = torch.exp(log_probs)       # convert log-probs to probabilities
    pred_classes = probs.argmax(dim=1)
    return pred_classes, probs
