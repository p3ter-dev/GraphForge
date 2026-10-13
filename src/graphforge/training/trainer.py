"""
Generic training loop for Assignment 1 node classification.

Supports any model with signature: model(x, norm_adj) -> log_probs [N, C].
Works for both the GCN and the MLP baseline.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    """Hyperparameters for a single training run."""
    lr: float = 1e-2
    weight_decay: float = 5e-4
    n_epochs: int = 300
    patience: int = 50        # early stopping: stop if val loss doesn't improve
    seed: int = 42
    hidden_dim: int = 64
    dropout: float = 0.5


@dataclass
class TrainingHistory:
    """Records per-epoch metrics during training."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    elapsed_seconds: float = 0.0


def accuracy(log_probs: Tensor, labels: Tensor, mask: Tensor) -> float:
    """
    Compute classification accuracy on masked nodes.

    Parameters
    ----------
    log_probs : Tensor [N, C]  — model output (log-softmax)
    labels    : Tensor [N]     — ground-truth class indices
    mask      : BoolTensor [N] — True for nodes to include in calculation

    Returns
    -------
    float — fraction of correctly classified nodes in the masked set
    """
    pred = log_probs[mask].argmax(dim=1)      # [M] predicted class indices
    correct = (pred == labels[mask]).sum()    # number of correct predictions
    return float(correct) / float(mask.sum())


def train_one_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    x: Tensor,
    norm_adj: Tensor,
    y: Tensor,
    train_mask: Tensor,
) -> tuple[float, float]:
    """
    Run one epoch of training.

    Returns (train_loss, train_accuracy).

    The loss is computed only over TRAINING nodes (train_mask).
    This is the standard semi-supervised GCN training protocol:
    - ALL nodes participate in message passing (forward pass uses all of x)
    - BUT loss gradients only come from labeled (training) nodes.
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass: use ALL node features and ALL edges.
    # Even test nodes contribute to message passing — their features flow
    # through the graph but their labels are never used in the loss.
    log_probs = model(x, norm_adj)     # [N, C]

    # Loss: NLLLoss expects log-probabilities (which log_softmax produces).
    # We only compute loss on training nodes.
    loss = nn.functional.nll_loss(log_probs[train_mask], y[train_mask])

    loss.backward()     # compute gradients via backpropagation
    optimizer.step()    # update weights using Adam

    train_acc = accuracy(log_probs.detach(), y, train_mask)
    return float(loss.item()), train_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: Tensor,
    norm_adj: Tensor,
    y: Tensor,
    mask: Tensor,
) -> tuple[float, float]:
    """
    Evaluate the model on a given node mask (val or test).

    @torch.no_grad() disables gradient computation during evaluation,
    saving memory and computation (we don't need gradients to evaluate).

    Returns (loss, accuracy).
    """
    model.eval()
    log_probs = model(x, norm_adj)
    loss = nn.functional.nll_loss(log_probs[mask], y[mask])
    acc = accuracy(log_probs, y, mask)
    return float(loss.item()), acc


def train(
    model: nn.Module,
    x: Tensor,
    norm_adj: Tensor,
    y: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    cfg: TrainingConfig,
) -> tuple[nn.Module, TrainingHistory]:
    """
    Full training loop with early stopping.

    Parameters
    ----------
    model      : any model with forward(x, norm_adj) → log_probs [N, C]
    x          : Tensor [N, F]    — node features
    norm_adj   : Tensor [N, N]    — normalized adjacency
    y          : Tensor [N]       — labels
    train_mask : BoolTensor [N]   — training nodes
    val_mask   : BoolTensor [N]   — validation nodes
    cfg        : TrainingConfig

    Returns
    -------
    (best_model_state_model, history)
    The returned model has been restored to the checkpoint with the best
    validation accuracy.
    """
    set_seed(cfg.seed)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    history = TrainingHistory()
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict: dict = {}
    start = time.time()

    for epoch in range(cfg.n_epochs):
        # ── training step ──────────────────────────────────────────────────
        tr_loss, tr_acc = train_one_epoch(model, optimizer, x, norm_adj, y, train_mask)

        # ── validation step ────────────────────────────────────────────────
        val_loss, val_acc = evaluate(model, x, norm_adj, y, val_mask)

        history.train_loss.append(tr_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(tr_acc)
        history.val_acc.append(val_acc)

        # ── early stopping & checkpoint ────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history.best_val_acc = val_acc
            history.best_epoch = epoch
            patience_counter = 0
            # Save a copy of model weights at the best validation point.
            # We use deepcopy via state_dict so the checkpoint is independent
            # of subsequent training steps.
            import copy
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch} (no val improvement for {cfg.patience} epochs)")
            break

        if (epoch + 1) % 50 == 0:
            print(
                f"  Epoch {epoch+1:4d} | "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    history.elapsed_seconds = time.time() - start

    # Restore the best checkpoint.
    model.load_state_dict(best_state_dict)
    model.eval()

    print(
        f"\n  Best val_acc={history.best_val_acc:.4f} at epoch {history.best_epoch}"
        f"  ({history.elapsed_seconds:.1f}s)"
    )
    return model, history
