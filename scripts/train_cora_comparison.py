"""
Assignment 2: Cora Architecture Comparison Training & Evaluation Script
========================================================================
Runs all required experiments:
  1. Main benchmark comparison: MLP vs GCN vs GraphSAGE vs GAT
  2. 3-seed robustness evaluation (reporting mean ± std)
  3. Depth and over-smoothing ablation (depths 2, 3, 4, 6 with cosine similarity)
  4. Edge removal ablation (drop rates 0%, 20%, 40%, 60%, 80%, 100%)
  5. Detailed error analysis on >=5 misclassified test nodes
  6. Visualizations: training curves, t-SNE embeddings, confusion matrices

Outputs:
  - reports/tables/a2_results.csv
  - reports/tables/a2_error_analysis.csv
  - reports/figures/a2_training_curves.png
  - reports/figures/a2_depth_oversmoothing.png
  - reports/figures/a2_edge_ablation.png
  - reports/figures/a2_tsne_embeddings.png
  - reports/figures/a2_confusion_matrices.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from graphforge.data.cora import (
    CORA_CLASSES,
    get_cora_statistics,
    load_cora_dataset,
    print_cora_summary,
)
from graphforge.models.cora_models import (
    CoraGAT,
    CoraGCN,
    CoraGraphSAGE,
    CoraMLP,
    compute_mean_pairwise_cosine_similarity,
)
from graphforge.training.cora_trainer import (
    CoraTrainingConfig,
    evaluate_cora_model,
    set_seed,
    train_cora_model,
)


def get_model_instance(name: str, in_feat: int, num_classes: int, depth: int = 2, hidden_dim: int = 64, dropout: float = 0.5):
    name_lower = name.lower()
    if "mlp" in name_lower:
        return CoraMLP(in_features=in_feat, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=depth, dropout=dropout)
    elif "sage" in name_lower or "graphsage" in name_lower:
        return CoraGraphSAGE(in_features=in_feat, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=depth, dropout=dropout)
    elif "gat" in name_lower:
        return CoraGAT(in_features=in_feat, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=depth, heads=8, dropout=0.6)
    else:  # default GCN
        return CoraGCN(in_features=in_feat, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=depth, dropout=dropout)


def drop_random_edges(edge_index: torch.Tensor, drop_rate: float, seed: int = 42) -> torch.Tensor:
    """Randomly drop edges uniformly at random."""
    if drop_rate <= 0.0:
        return edge_index
    if drop_rate >= 1.0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    num_edges = edge_index.shape[1]
    keep_prob = 1.0 - drop_rate
    rng = torch.Generator().manual_seed(seed)
    mask = torch.rand(num_edges, generator=rng) < keep_prob
    return edge_index[:, mask]


def main():
    parser = argparse.ArgumentParser(description="Assignment 2: Cora Architecture Comparison")
    parser.add_argument("--output_dir", type=str, default="reports", help="Output directory for reports")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds for evaluation")
    args = parser.parse_args()

    tab_dir = Path(args.output_dir) / "tables"
    fig_dir = Path(args.output_dir) / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Loading and inspecting Cora dataset...")
    _, data = load_cora_dataset(root="data/Cora")
    stats = get_cora_statistics(data)
    print_cora_summary(stats)

    in_feat = data.num_features
    num_classes = stats["num_classes"]
    model_names = ["MLP", "GCN", "GraphSAGE", "GAT"]

    # ── 1. Benchmark Across Seeds ─────────────────────────────────────────────
    print(f"\n[2/6] Running benchmark across {len(args.seeds)} seeds: {args.seeds}...")
    benchmark_results: Dict[str, Dict[str, List[float]]] = {
        m: {"acc": [], "f1": [], "time": [], "params": []} for m in model_names
    }
    champion_models: Dict[str, Any] = {}
    champion_histories: Dict[str, Any] = {}
    champion_evals: Dict[str, Any] = {}

    for seed in args.seeds:
        for name in model_names:
            set_seed(seed)
            lr = 0.005 if name == "GAT" else 0.01
            dropout = 0.6 if name == "GAT" else 0.5
            cfg = CoraTrainingConfig(lr=lr, max_epochs=200, patience=40, seed=seed, hidden_dim=64, dropout=dropout)

            model = get_model_instance(name, in_feat, num_classes, depth=2, hidden_dim=64, dropout=dropout)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            model, hist = train_cora_model(model, data.x, data.edge_index, data.y, data.train_mask, data.val_mask, cfg)
            eval_res = evaluate_cora_model(model, data.x, data.edge_index, data.y, data.test_mask)

            benchmark_results[name]["acc"].append(eval_res["accuracy"])
            benchmark_results[name]["f1"].append(eval_res["macro_f1"])
            benchmark_results[name]["time"].append(hist.training_time_seconds)
            benchmark_results[name]["params"].append(n_params)

            if seed == args.seeds[0]:
                champion_models[name] = model
                champion_histories[name] = hist
                champion_evals[name] = eval_res

    print("\n" + "=" * 75)
    print(f"  BENCHMARK RESULTS SUMMARY ({len(args.seeds)} Seeds Mean ± Std)")
    print("=" * 75)
    print(f"  {'Model':<12} {'Test Acc (%)':<20} {'Macro-F1':<16} {'Time (s)':<12} {'Params':<10}")
    print("  " + "-" * 71)

    csv_rows = []
    for name in model_names:
        accs = np.array(benchmark_results[name]["acc"]) * 100
        f1s = np.array(benchmark_results[name]["f1"])
        times = np.array(benchmark_results[name]["time"])
        params = benchmark_results[name]["params"][0]

        acc_str = f"{accs.mean():.2f} ± {accs.std():.2f}"
        f1_str = f"{f1s.mean():.4f} ± {f1s.std():.4f}"
        time_str = f"{times.mean():.2f}s"

        print(f"  {name:<12} {acc_str:<20} {f1_str:<16} {time_str:<12} {params:<10}")

        csv_rows.append({
            "model": name,
            "test_acc_mean": f"{accs.mean():.4f}",
            "test_acc_std": f"{accs.std():.4f}",
            "macro_f1_mean": f"{f1s.mean():.4f}",
            "macro_f1_std": f"{f1s.std():.4f}",
            "time_mean_sec": f"{times.mean():.2f}",
            "params": params,
        })
    print("=" * 75)

    # Save benchmark CSV
    with open(tab_dir / "a2_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved benchmark table to {tab_dir / 'a2_results.csv'}")

    # ── 2. Training Curves & Confusion Matrices ───────────────────────────────
    print("\n[3/6] Generating training curves and confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"MLP": "#E84855", "GCN": "#2E86AB", "GraphSAGE": "#3BB273", "GAT": "#A23B72"}

    for name in model_names:
        h = champion_histories[name]
        axes[0].plot(h.train_loss, label=name, color=colors[name], alpha=0.85)
        axes[1].plot(h.val_acc, label=name, color=colors[name], alpha=0.85)

    axes[0].set_title("Cora Training Loss Curves", fontsize=11)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Cora Validation Accuracy Curves", fontsize=11)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "a2_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for idx, name in enumerate(model_names):
        cm = champion_evals[name]["confusion_matrix"]
        axes[idx].imshow(cm, cmap="Blues", interpolation="nearest")
        axes[idx].set_title(f"{name} (Acc: {champion_evals[name]['accuracy']*100:.1f}%)", fontsize=11)
        axes[idx].set_xlabel("Predicted")
        if idx == 0:
            axes[idx].set_ylabel("True Class")
        axes[idx].set_xticks(range(num_classes))
        axes[idx].set_yticks(range(num_classes))
    plt.tight_layout()
    plt.savefig(fig_dir / "a2_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. Depth & Over-Smoothing Ablation ───────────────────────────────────
    print("\n[4/6] Investigating GNN depth & over-smoothing (depths: 2, 3, 4, 6)...")
    depths = [2, 3, 4, 6]
    depth_accs: Dict[str, List[float]] = {m: [] for m in model_names}
    depth_sims: Dict[str, List[float]] = {m: [] for m in model_names}

    for d in depths:
        for name in model_names:
            set_seed(42)
            lr = 0.005 if name == "GAT" else 0.01
            cfg = CoraTrainingConfig(lr=lr, max_epochs=150, patience=35, seed=42, hidden_dim=64, dropout=0.5)
            model = get_model_instance(name, in_feat, num_classes, depth=d, hidden_dim=64, dropout=0.5)
            model, _ = train_cora_model(model, data.x, data.edge_index, data.y, data.train_mask, data.val_mask, cfg)
            ev = evaluate_cora_model(model, data.x, data.edge_index, data.y, data.test_mask)

            # Measure embedding similarity across nodes
            emb = model.get_embeddings(data.x, data.edge_index)
            sim = compute_mean_pairwise_cosine_similarity(emb, sample_size=500, seed=42)

            depth_accs[name].append(ev["accuracy"] * 100)
            depth_sims[name].append(sim)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in model_names:
        axes[0].plot(depths, depth_accs[name], marker="o", label=name, color=colors[name])
        axes[1].plot(depths, depth_sims[name], marker="s", label=name, color=colors[name])

    axes[0].set_title("Test Accuracy vs GNN Depth", fontsize=11)
    axes[0].set_xlabel("Number of Layers (Depth)")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_xticks(depths)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Embedding Cosine Similarity (Over-smoothing)", fontsize=11)
    axes[1].set_xlabel("Number of Layers (Depth)")
    axes[1].set_ylabel("Mean Pairwise Cosine Similarity (1.0 = collapsed)")
    axes[1].set_xticks(depths)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "a2_depth_oversmoothing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved depth/over-smoothing plot to {fig_dir / 'a2_depth_oversmoothing.png'}")

    # ── 4. Edge Sparsification Ablation ───────────────────────────────────────
    print("\n[5/6] Investigating edge removal ablation (drop rates: 0% to 100%)...")
    drop_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ablation_accs: Dict[str, List[float]] = {m: [] for m in model_names}

    for rate in drop_rates:
        sparse_edges = drop_random_edges(data.edge_index, rate, seed=42)
        for name in model_names:
            set_seed(42)
            lr = 0.005 if name == "GAT" else 0.01
            cfg = CoraTrainingConfig(lr=lr, max_epochs=120, patience=30, seed=42, hidden_dim=64, dropout=0.5)
            model = get_model_instance(name, in_feat, num_classes, depth=2, hidden_dim=64, dropout=0.5)
            model, _ = train_cora_model(model, data.x, sparse_edges, data.y, data.train_mask, data.val_mask, cfg)
            ev = evaluate_cora_model(model, data.x, sparse_edges, data.y, data.test_mask)
            ablation_accs[name].append(ev["accuracy"] * 100)

    plt.figure(figsize=(8, 5))
    for name in model_names:
        plt.plot(drop_rates, ablation_accs[name], marker="o", label=name, color=colors[name], linewidth=2)
    plt.title("Effect of Edge Removal on Model Accuracy (Cora)", fontsize=12)
    plt.xlabel("Edge Removal Fraction (1.0 = No Edges)")
    plt.ylabel("Test Accuracy (%)")
    plt.xticks(drop_rates, [f"{int(r*100)}%" for r in drop_rates])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "a2_edge_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved edge ablation plot to {fig_dir / 'a2_edge_ablation.png'}")

    # ── 5. Error Analysis on >= 5 Misclassified Test Nodes ────────────────────
    print("\n[6/6] Conducting failure analysis on misclassified test nodes...")
    gcn_preds = champion_evals["GCN"]["preds"]
    test_indices = torch.where(data.test_mask)[0].cpu().numpy()
    test_targets = data.y[data.test_mask].cpu().numpy()

    incorrect_mask = (gcn_preds != test_targets)
    incorrect_test_node_ids = test_indices[incorrect_mask]

    error_analysis_rows = []
    # Analyze first 5 failure cases
    src, dst = data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy()
    labels = data.y.cpu().numpy()

    for node_id in incorrect_test_node_ids[:6]:
        true_c = int(labels[node_id])
        pred_c = int(gcn_preds[np.where(test_indices == node_id)[0][0]])

        # Find neighbors
        neighbor_mask = (src == node_id)
        nbrs = dst[neighbor_mask]
        degree = len(nbrs)

        if degree > 0:
            nbr_labels = labels[nbrs]
            homophily = float((nbr_labels == true_c).mean())
            pred_ratio = float((nbr_labels == pred_c).mean())
        else:
            homophily = 0.0
            pred_ratio = 0.0

        # Non-zero feature count
        active_words = int((data.x[node_id] > 0).sum().item())

        # Diagnosis
        if degree == 0:
            diag = "Isolated node; GCN receives zero neighbor signal and degrades to feature projection."
        elif homophily == 0.0:
            diag = f"Extreme heterophily: 0% of neighbors belong to true class {true_c}; {pred_ratio*100:.0f}% belong to predicted class {pred_c} (misled by neighbor voting)."
        elif homophily < 0.4:
            diag = f"Boundary/inter-disciplinary paper: minority homophily ({homophily*100:.0f}%). Neighborhood dominated by class {pred_c}."
        elif active_words < 5:
            diag = f"Sparse feature vector ({active_words} active words); insufficient text signal."
        else:
            diag = "Complex boundary case; high cross-citation between closely related subject areas."

        error_analysis_rows.append({
            "node_id": int(node_id),
            "true_class": f"{true_c} ({CORA_CLASSES[true_c]})",
            "pred_class": f"{pred_c} ({CORA_CLASSES[pred_c]})",
            "degree": degree,
            "neighborhood_homophily": f"{homophily * 100:.1f}%",
            "pred_class_neighbor_pct": f"{pred_ratio * 100:.1f}%",
            "active_words": active_words,
            "diagnosis": diag,
        })

    print("\n" + "=" * 90)
    print("  ERROR ANALYSIS ON MISCLASSIFIED TEST NODES (GCN)")
    print("=" * 90)
    for r in error_analysis_rows[:5]:
        print(f"  Node {r['node_id']} | True: {r['true_class']} | Pred: {r['pred_class']} | Degree: {r['degree']} | Nbr Homophily: {r['neighborhood_homophily']}")
        print(f"    Diagnosis: {r['diagnosis']}\n")

    with open(tab_dir / "a2_error_analysis.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(error_analysis_rows[0].keys()))
        writer.writeheader()
        writer.writerows(error_analysis_rows)
    print(f"  Saved error analysis to {tab_dir / 'a2_error_analysis.csv'}")

    # ── 6. t-SNE Embeddings Plot ──────────────────────────────────────────────
    print("\nGenerating t-SNE 2D latent space visualizations...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    try:
        cora_cmap = matplotlib.colormaps["tab10"].resampled(num_classes)
    except Exception:
        cora_cmap = "tab10"
    y_test = data.y[data.test_mask].cpu().numpy()

    for idx, name in enumerate(model_names):
        m = champion_models[name]
        emb = m.get_embeddings(data.x, data.edge_index)[data.test_mask].cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(emb)

        axes[idx].scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_test, cmap=cora_cmap, alpha=0.75, s=25)
        axes[idx].set_title(f"{name} Latent Space (t-SNE)", fontsize=11)
        axes[idx].axis("off")

    plt.suptitle("Cora Test Set Embeddings: Feature Projection (MLP) vs Graph Representations (GNNs)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "a2_tsne_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved t-SNE embeddings plot to {fig_dir / 'a2_tsne_embeddings.png'}")

    print("\n[Done] Assignment 2 Cora benchmark and experimental suite complete!")


if __name__ == "__main__":
    main()
