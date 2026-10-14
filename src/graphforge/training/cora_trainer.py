"""
Standardized Trainer and Evaluator for Cora Models
===================================================
Provides reproducible training with validation-based early stopping,
checkpoint restoration, timing, parameter counting, and test evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, classification_report


@dataclass
class CoraTrainingConfig:
    lr: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 300
    patience: int = 40
    seed: int = 42
    hidden_dim: int = 64
    dropout: float = 0.5
    heads: int = 8  # for GAT


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_acc: float = 0.0
    training_time_seconds: float = 0.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_cora_model(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    cfg: CoraTrainingConfig,
) -> Tuple[nn.Module, TrainingHistory]:
    """
    Train a model with early stopping based on validation loss/accuracy.
    Restores the best validation checkpoint before returning.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history = TrainingHistory()
    best_val_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0

    start_time = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        # ── Train Step ──
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Compute train accuracy
        train_pred = out[train_mask].argmax(dim=-1)
        train_acc = (train_pred == y[train_mask]).float().mean().item()

        # ── Validation Step ──
        model.eval()
        with torch.no_grad():
            val_out = model(x, edge_index)
            val_loss = F.nll_loss(val_out[val_mask], y[val_mask]).item()
            val_pred = val_out[val_mask].argmax(dim=-1)
            val_acc = (val_pred == y[val_mask]).float().mean().item()

        history.train_loss.append(float(loss.item()))
        history.val_loss.append(float(val_loss))
        history.train_acc.append(float(train_acc))
        history.val_acc.append(float(val_acc))

        # Checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history.best_val_acc = val_acc
            history.best_epoch = epoch
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= cfg.patience:
                break

    history.training_time_seconds = time.time() - start_time

    # Restore champion weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def evaluate_cora_model(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, Any]:
    """
    Evaluate model on a given mask (val or test).
    Computes accuracy, macro F1, and confusion matrix.
    """
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        loss = F.nll_loss(out[mask], y[mask]).item()
        preds = out[mask].argmax(dim=-1).cpu().numpy()
        targets = y[mask].cpu().numpy()

        acc = float((preds == targets).mean())
        macro_f1 = float(f1_score(targets, preds, average="macro"))
        cm = confusion_matrix(targets, preds)

        return {
            "loss": loss,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "confusion_matrix": cm,
            "preds": preds,
            "targets": targets,
            "full_logits": out.cpu(),
        }
