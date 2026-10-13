"""
Assignment 1: GCN from Scratch — Training Script
=================================================
Runs all required experiments:
  1. Train GCN on synthetic two-community graph
  2. Train MLP baseline (same features, no edges)
  3. Randomized-edge ablation (shuffle adjacency, break community structure)

Produces:
  - Console output with results table
  - reports/tables/a1_results.csv
  - reports/figures/a1_training_curves.png
  - reports/figures/a1_embeddings.png

Usage:
  uv run python scripts/train_gcn_scratch.py
  uv run python scripts/train_gcn_scratch.py --seed 123 --n_nodes 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── allow running from repo root without installing the package ──────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn

from graphforge.data.synthetic import make_two_community_graph, print_graph_summary
from graphforge.models.gcn_scratch import TwoLayerGCN, preprocess_adjacency
from graphforge.models.mlp import MLP
from graphforge.training.trainer import TrainingConfig, train, evaluate


def make_randomized_adjacency(adj: torch.Tensor, seed: int = 999) -> torch.Tensor:
    """
    Destroy the community structure by randomly shuffling edges.

    We keep the same number of edges and the same degree sequence,
    but break the community correlation. This is an *edge rewiring*.

    Method:
    - Collect all edges (i, j) with i < j (upper triangle).
    - Randomly shuffle node assignments by permuting node labels.
    - Rebuild the adjacency matrix.

    The resulting graph has the same number of edges but they no longer
    correlate with class membership — so the GCN cannot use them to
    predict the correct class better than chance.
    """
    rng = np.random.default_rng(seed)
    N = adj.shape[0]
    adj_np = adj.numpy()

    # Extract edge list
    rows, cols = np.where(np.triu(adj_np, k=1) > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    n_edges = len(edges)

    # Random permutation of node labels — this breaks community structure
    perm = rng.permutation(N)

    # Rebuild adjacency with permuted node labels
    new_adj = np.zeros((N, N), dtype=np.float32)
    for u, v in edges:
        new_u, new_v = int(perm[u]), int(perm[v])
        new_adj[new_u, new_v] = 1.0
        new_adj[new_v, new_u] = 1.0

    return torch.tensor(new_adj)


def run_experiment(
    model: nn.Module,
    data: dict,
    cfg: TrainingConfig,
    norm_adj: torch.Tensor,
    name: str,
) -> dict:
    """
    Train a model and return a result dictionary.

    Parameters
    ----------
    model    : GCN or MLP (both accept same forward signature)
    data     : output of make_two_community_graph()
    cfg      : training hyperparameters
    norm_adj : normalized adjacency to use for message passing
               (may be original or randomized)
    name     : experiment name (for printing)

    Returns
    -------
    dict with keys: name, test_acc, val_acc, n_params, history
    """
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"{'='*60}")

    x = data["x"]
    y = data["y"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    # Count trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params}")

    # Train
    model, history = train(model, x, norm_adj, y, train_mask, val_mask, cfg)

    # Final test evaluation (only done ONCE, after training is complete)
    test_loss, test_acc = evaluate(model, x, norm_adj, y, test_mask)
    print(f"  Test accuracy: {test_acc:.4f}")

    return {
        "name": name,
        "test_acc": test_acc,
        "val_acc": history.best_val_acc,
        "n_params": n_params,
        "history": history,
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser(description="Assignment 1: GCN from Scratch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_nodes", type=int, default=120, help="Number of synthetic nodes")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--n_epochs", type=int, default=400, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--output_dir", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    SEED = args.seed
    N = args.n_nodes
    H = args.hidden_dim

    # ── Setup output directories ──────────────────────────────────────────────
    fig_dir = Path(args.output_dir) / "figures"
    tab_dir = Path(args.output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate synthetic graph ──────────────────────────────────────────────
    print("\n[1/4] Generating synthetic two-community graph...")
    data = make_two_community_graph(n_nodes=N, seed=SEED)
    print_graph_summary(data)

    # Pre-compute normalized adjacency (original graph)
    norm_adj_orig = preprocess_adjacency(data["adj"])

    # Pre-compute normalized adjacency (randomized graph — edges shuffled)
    adj_random = make_randomized_adjacency(data["adj"], seed=SEED + 100)
    norm_adj_rand = preprocess_adjacency(adj_random)

    # Training config (same for all experiments for fair comparison)
    cfg = TrainingConfig(
        lr=args.lr,
        weight_decay=5e-4,
        n_epochs=args.n_epochs,
        patience=50,
        seed=SEED,
        hidden_dim=H,
        dropout=args.dropout,
    )

    n_features = data["x"].shape[1]  # 3
    n_classes = int(data["y"].max().item()) + 1  # 2

    # ── Experiment 1: GCN with original graph ─────────────────────────────────
    print("\n[2/4] Running experiments...")
    gcn_model = TwoLayerGCN(n_features, H, n_classes, dropout=args.dropout)
    result_gcn = run_experiment(gcn_model, data, cfg, norm_adj_orig, "GCN (original graph)")

    # ── Experiment 2: MLP baseline (no edges at all) ──────────────────────────
    # The MLP ignores norm_adj entirely — we pass it for API consistency.
    mlp_model = MLP(n_features, H, n_classes, dropout=args.dropout)
    result_mlp = run_experiment(mlp_model, data, cfg, norm_adj_orig, "MLP (no graph, features only)")

    # ── Experiment 3: GCN with randomized edges ────────────────────────────────
    gcn_rand_model = TwoLayerGCN(n_features, H, n_classes, dropout=args.dropout)
    result_rand = run_experiment(
        gcn_rand_model, data, cfg, norm_adj_rand, "GCN (randomized edges)"
    )

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<30} {'Val Acc':>9} {'Test Acc':>9} {'Params':>8}")
    print("  " + "-" * 58)
    for r in [result_gcn, result_mlp, result_rand]:
        print(
            f"  {r['name']:<30} {r['val_acc']:>9.4f} {r['test_acc']:>9.4f} {r['n_params']:>8}"
        )
    print("=" * 60)

    # Interpretation
    gcn_vs_mlp = result_gcn["test_acc"] - result_mlp["test_acc"]
    gcn_vs_rand = result_gcn["test_acc"] - result_rand["test_acc"]
    print(f"\n  GCN vs MLP:           {gcn_vs_mlp:+.4f}")
    print(f"  GCN vs Randomized:    {gcn_vs_rand:+.4f}")

    if gcn_vs_mlp > 0.02:
        print("  → Graph structure provides clear benefit over features alone.")
    elif gcn_vs_mlp > -0.02:
        print("  → Graph structure provides marginal benefit; features dominate.")
    else:
        print("  → WARNING: MLP outperforms GCN — investigate over-smoothing or bugs.")

    if gcn_vs_rand > 0.05:
        print("  → Community structure in the graph is crucial for performance.")
    else:
        print("  → Randomizing edges had little effect — investigate if graph is used.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    import csv
    csv_path = tab_dir / "a1_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "val_acc", "test_acc", "n_params"])
        writer.writeheader()
        for r in [result_gcn, result_mlp, result_rand]:
            writer.writerow({
                "model": r["name"],
                "val_acc": f"{r['val_acc']:.4f}",
                "test_acc": f"{r['test_acc']:.4f}",
                "n_params": r["n_params"],
            })
    print(f"\n  Results saved to: {csv_path}")

    # ── Save training curves ──────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")  # headless rendering

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colors = ["#2E86AB", "#E84855", "#A23B72"]
        results = [result_gcn, result_mlp, result_rand]

        for ax_idx, (result, color) in enumerate(zip(results, colors)):
            h = result["history"]
            axes[0].plot(h.train_loss, label=result["name"], color=color, alpha=0.8)
            axes[1].plot(h.val_acc, label=result["name"], color=color, alpha=0.8)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Training Loss Curves")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Accuracy")
        axes[1].set_title("Validation Accuracy Curves")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        fig_path = fig_dir / "a1_training_curves.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Training curves saved to: {fig_path}")

    except ImportError:
        print("  matplotlib not available — skipping plots.")

    print("\n[Done] Assignment 1 training script complete.")


if __name__ == "__main__":
    main()
