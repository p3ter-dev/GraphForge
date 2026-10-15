"""
Assignment 3: Link Prediction and Data Leakage Training & Evaluation Script
===========================================================================
Runs all required experiments:
  1. Valid Link Prediction: Dot-Product Decoder vs Symmetric MLP Decoder vs Common Neighbors
  2. Data Leakage Experiment: Deliberately leaky message-passing comparison
  3. ROC-AUC and Average Precision (AP) evaluation
  4. Top 10 candidate unobserved link predictions (Open-World Assumption analysis)
  5. Plots: ROC / PR curves, Leaky vs Valid comparison

Outputs:
  - reports/tables/a3_results.csv
  - reports/tables/a3_candidate_links.csv
  - reports/figures/a3_roc_pr_curves.png
  - reports/figures/a3_leaky_vs_valid.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # noqa: E402
# isort: split

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from graphforge.data.cora import load_cora_dataset
from graphforge.data.link_split import LinkDataSplits, create_link_prediction_splits
from graphforge.models.link_prediction import (
    DotProductDecoder,
    GNNEncoder,
    LinkPredictor,
    SymmetricMLPDecoder,
    compute_common_neighbors_score,
)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_link_predictor(
    model: LinkPredictor,
    splits: LinkDataSplits,
    message_edge_index: torch.Tensor,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    max_epochs: int = 200,
    patience: int = 30,
) -> Tuple[LinkPredictor, Dict[str, Any]]:
    """
    Train a link prediction model using binary cross-entropy on positive and negative links.
    Early stopping on validation ROC-AUC.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = 0.0
    best_weights = None
    no_improve = 0

    x = splits.x
    train_pos = splits.train_pos_edge_index
    train_neg = splits.train_neg_edge_index

    val_pos = splits.val_pos_edge_index
    val_neg = splits.val_neg_edge_index

    # Labels for BCE
    train_y = torch.cat([torch.ones(train_pos.shape[1]), torch.zeros(train_neg.shape[1])])
    val_y = torch.cat([torch.ones(val_pos.shape[1]), torch.zeros(val_neg.shape[1])]).numpy()

    for epoch in range(1, max_epochs + 1):
        # ── Train ──
        model.train()
        optimizer.zero_grad()

        z = model.encoder(x, message_edge_index)
        pos_scores = model.decoder(z, train_pos)
        neg_scores = model.decoder(z, train_neg)
        all_scores = torch.cat([pos_scores, neg_scores])

        loss = F.binary_cross_entropy(all_scores, train_y)
        loss.backward()
        optimizer.step()

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            z_val = model.encoder(x, message_edge_index)
            v_pos_scores = model.decoder(z_val, val_pos).cpu().numpy()
            v_neg_scores = model.decoder(z_val, val_neg).cpu().numpy()
            v_all_scores = np.concatenate([v_pos_scores, v_neg_scores])

            val_auc = float(roc_auc_score(val_y, v_all_scores))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, {"best_val_auc": best_val_auc}


def evaluate_link_predictions(
    model: LinkPredictor,
    splits: LinkDataSplits,
    message_edge_index: torch.Tensor,
) -> Dict[str, Any]:
    """
    Evaluate test ROC-AUC, Average Precision, and ROC/PR curve points.
    """
    model.eval()
    with torch.no_grad():
        z = model.encoder(splits.x, message_edge_index)
        pos_scores = model.decoder(z, splits.test_pos_edge_index).cpu().numpy()
        neg_scores = model.decoder(z, splits.test_neg_edge_index).cpu().numpy()

        y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        y_scores = np.concatenate([pos_scores, neg_scores])

        auc = float(roc_auc_score(y_true, y_scores))
        ap = float(average_precision_score(y_true, y_scores))
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        prec, rec, _ = precision_recall_curve(y_true, y_scores)

        return {
            "roc_auc": auc,
            "ap": ap,
            "fpr": fpr,
            "tpr": tpr,
            "precision": prec,
            "recall": rec,
            "z": z,
            "y_true": y_true,
            "y_scores": y_scores,
        }


def main():
    parser = argparse.ArgumentParser(description="Assignment 3: Link Prediction and Data Leakage")
    parser.add_argument("--output_dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    tab_dir = Path(args.output_dir) / "tables"
    fig_dir = Path(args.output_dir) / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading Cora dataset (excluding document classes y)...")
    _, data = load_cora_dataset(root="data/Cora")
    set_seed(args.seed)

    print("\n[2/5] Splitting positive and negative links with leakage guards...")
    splits = create_link_prediction_splits(data.edge_index, data.x, val_ratio=0.05, test_ratio=0.10, seed=args.seed)

    n_train = splits.train_pos_edge_index.shape[1]
    n_val = splits.val_pos_edge_index.shape[1]
    n_test = splits.test_pos_edge_index.shape[1]
    print(f"  Positive links (undirected): Train={n_train}, Val={n_val}, Test={n_test}")
    print(f"  Message-passing graph edges: {splits.train_message_edge_index.shape[1]} (strictly {2 * n_train} directed)")
    print(f"  Negative samples: Train={splits.train_neg_edge_index.shape[1]}, Val={splits.val_neg_edge_index.shape[1]}, Test={splits.test_neg_edge_index.shape[1]}")

    results = {}

    # ── 1. Common Neighbors Baseline ─────────────────────────────────────────
    print("\n[3/5] Evaluating Common-Neighbors baseline (on E_train strictly)...")
    cn_pos = compute_common_neighbors_score(splits.train_message_edge_index, splits.num_nodes, splits.test_pos_edge_index).numpy()
    cn_neg = compute_common_neighbors_score(splits.train_message_edge_index, splits.num_nodes, splits.test_neg_edge_index).numpy()
    cn_y_true = np.concatenate([np.ones_like(cn_pos), np.zeros_like(cn_neg)])
    cn_scores = np.concatenate([cn_pos, cn_neg])

    cn_auc = float(roc_auc_score(cn_y_true, cn_scores))
    cn_ap = float(average_precision_score(cn_y_true, cn_scores))
    cn_fpr, cn_tpr, _ = roc_curve(cn_y_true, cn_scores)
    cn_prec, cn_rec, _ = precision_recall_curve(cn_y_true, cn_scores)

    results["Common Neighbors (Baseline)"] = {
        "roc_auc": cn_auc,
        "ap": cn_ap,
        "fpr": cn_fpr,
        "tpr": cn_tpr,
        "precision": cn_prec,
        "recall": cn_rec,
        "leaky": False,
    }

    # ── 2. Learned Model: GNN + Dot Product Decoder (Valid) ───────────────────
    print("\nTraining GNN + Dot-Product Decoder (Valid message passing)...")
    set_seed(args.seed)
    dot_encoder = GNNEncoder(in_features=data.num_features, hidden_dim=64, out_dim=64, dropout=0.3)
    dot_model = LinkPredictor(dot_encoder, DotProductDecoder())
    dot_model, _ = train_link_predictor(dot_model, splits, splits.train_message_edge_index, lr=0.01, max_epochs=200, patience=30)
    dot_eval = evaluate_link_predictions(dot_model, splits, splits.train_message_edge_index)
    dot_eval["leaky"] = False
    results["GNN + Dot-Product Decoder (Valid)"] = dot_eval

    # ── 3. Learned Model: GNN + Symmetric MLP Decoder (Valid) ─────────────────
    print("\nTraining GNN + Symmetric MLP Decoder (Valid message passing)...")
    set_seed(args.seed)
    mlp_encoder = GNNEncoder(in_features=data.num_features, hidden_dim=64, out_dim=64, dropout=0.3)
    mlp_decoder = SymmetricMLPDecoder(emb_dim=64, hidden_dim=64, dropout=0.2)
    mlp_model = LinkPredictor(mlp_encoder, mlp_decoder)
    mlp_model, _ = train_link_predictor(mlp_model, splits, splits.train_message_edge_index, lr=0.01, max_epochs=200, patience=30)
    mlp_eval = evaluate_link_predictions(mlp_model, splits, splits.train_message_edge_index)
    mlp_eval["leaky"] = False
    results["GNN + Symmetric MLP Decoder (Valid)"] = mlp_eval

    # ── 4. Deliberately Leaky Experiment ──────────────────────────────────────
    print("\n[4/5] Running DELIBERATELY LEAKY experiment (Full graph message passing)...")
    print("  WARNING: This model cheats by passing messages over held-out val & test edges!")
    set_seed(args.seed)
    leaky_encoder = GNNEncoder(in_features=data.num_features, hidden_dim=64, out_dim=64, dropout=0.3)
    leaky_model = LinkPredictor(leaky_encoder, DotProductDecoder())
    # Notice we pass data.edge_index (FULL GRAPH) into message passing!
    leaky_model, _ = train_link_predictor(leaky_model, splits, data.edge_index, lr=0.01, max_epochs=200, patience=30)
    leaky_eval = evaluate_link_predictions(leaky_model, splits, data.edge_index)
    leaky_eval["leaky"] = True
    results["GNN + Dot-Product (LEAKY - Invalid)"] = leaky_eval

    # Print Results Summary
    print("\n" + "=" * 70)
    print("  ASSIGNMENT 3: LINK PREDICTION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  {'Model / Methodology':<36} {'Test ROC-AUC':>14} {'Test AP':>12}")
    print("  " + "-" * 66)

    csv_rows = []
    for name, r in results.items():
        print(f"  {name:<36} {r['roc_auc']:>14.4f} {r['ap']:>12.4f}")
        csv_rows.append({
            "model": name,
            "roc_auc": f"{r['roc_auc']:.4f}",
            "average_precision": f"{r['ap']:.4f}",
            "is_leaky": r["leaky"],
        })
    print("=" * 70)

    # Save benchmark table
    with open(tab_dir / "a3_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "roc_auc", "average_precision", "is_leaky"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Results saved to: {tab_dir / 'a3_results.csv'}")

    # ── 5. Generate Candidate Link Predictions (Open-World Assumption) ────────
    print("\n[5/5] Extracting top 10 candidate unobserved links...")
    # Evaluate 2,000 unobserved candidate pairs
    rng = np.random.default_rng(args.seed + 999)
    cand_list = []
    seen_all_pos = set(
        (min(int(data.edge_index[0, i]), int(data.edge_index[1, i])),
         max(int(data.edge_index[0, i]), int(data.edge_index[1, i])))
        for i in range(data.edge_index.shape[1])
    )

    while len(cand_list) < 2000:
        u = int(rng.integers(0, splits.num_nodes))
        v = int(rng.integers(0, splits.num_nodes))
        if u != v:
            p = (min(u, v), max(u, v))
            if p not in seen_all_pos:
                cand_list.append(p)

    cand_tensor = torch.tensor(cand_list, dtype=torch.long).t()
    with torch.no_grad():
        z_cand = mlp_model.encoder(splits.x, splits.train_message_edge_index)
        cand_probs = mlp_model.decoder(z_cand, cand_tensor).cpu().numpy()

    top10_idx = np.argsort(cand_probs)[::-1][:10]
    candidate_rows = []

    print("\n" + "=" * 80)
    print("  TOP 10 CANDIDATE PREDICTED CITATIONS (UNOBSERVED LINKS)")
    print("=" * 80)
    print(f"  {'Rank':<5} {'Node Pair (u, v)':<22} {'Predicted Prob':<16} {'Common Neighbors (Train)':<25}")
    print("  " + "-" * 76)

    cn_cand = compute_common_neighbors_score(splits.train_message_edge_index, splits.num_nodes, cand_tensor).numpy()

    for rank, idx in enumerate(top10_idx, start=1):
        u, v = cand_tensor[0, idx].item(), cand_tensor[1, idx].item()
        prob = cand_probs[idx]
        cn = int(cn_cand[idx])
        print(f"  {rank:<5} ({u}, {v}){'':<12} {prob:<16.4f} {cn:<25}")

        candidate_rows.append({
            "rank": rank,
            "node_u": u,
            "node_v": v,
            "predicted_probability": f"{prob:.4f}",
            "common_neighbors_train": cn,
            "open_world_justification": (
                "Unobserved link in citation networks is NOT confirmed negative; "
                "represents high-confidence literature cross-reference candidate."
            ),
        })

    with open(tab_dir / "a3_candidate_links.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_rows)
    print(f"  Candidate predictions saved to: {tab_dir / 'a3_candidate_links.csv'}")

    # ── 6. Visualizations ─────────────────────────────────────────────────────
    print("\nGenerating ROC / PR curves and Leaky vs Valid comparison plots...")

    # Plot 1: ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_models = [
        ("Common Neighbors", results["Common Neighbors (Baseline)"], "#757575", "--"),
        ("GNN + DotProduct (Valid)", results["GNN + Dot-Product Decoder (Valid)"], "#2E86AB", "-"),
        ("GNN + Symmetric MLP (Valid)", results["GNN + Symmetric MLP Decoder (Valid)"], "#3BB273", "-"),
        ("GNN + DotProduct (LEAKY)", results["GNN + Dot-Product (LEAKY - Invalid)"], "#E84855", ":"),
    ]

    for label, r, color, ls in plot_models:
        axes[0].plot(r["fpr"], r["tpr"], label=f"{label} (AUC={r['roc_auc']:.3f})", color=color, linestyle=ls, linewidth=2)
        axes[1].plot(r["recall"], r["precision"], label=f"{label} (AP={r['ap']:.3f})", color=color, linestyle=ls, linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC Curves (Test Links)", fontsize=11)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Precision-Recall Curves (Test Links)", fontsize=11)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "a3_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Valid vs Leaky bar comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["Common Neighbors", "Dot-Product (Valid)", "Symmetric MLP (Valid)", "Dot-Product (LEAKY)"]
    aucs = [results["Common Neighbors (Baseline)"]["roc_auc"],
            results["GNN + Dot-Product Decoder (Valid)"]["roc_auc"],
            results["GNN + Symmetric MLP Decoder (Valid)"]["roc_auc"],
            results["GNN + Dot-Product (LEAKY - Invalid)"]["roc_auc"]]
    bar_colors = ["#757575", "#2E86AB", "#3BB273", "#E84855"]

    bars = ax.bar(names, [a * 100 for a in aucs], color=bar_colors, width=0.55)
    ax.set_ylabel("Test ROC-AUC (%)")
    ax.set_ylim(50, 100)
    ax.set_title("Link Prediction: Valid Evaluation vs Deliberately Leaky Evaluation", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_dir / "a3_leaky_vs_valid.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n[Done] Assignment 3 link prediction experiments completed successfully!")


if __name__ == "__main__":
    main()
