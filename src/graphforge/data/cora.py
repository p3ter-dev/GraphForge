"""
Cora Dataset Loader, Graph Statistics, and Inspection Utilities
================================================================
GraphForge Assignment 2: Architecture Comparison on Cora
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

CORA_CLASSES = [
    "Theory",
    "Reinforcement_Learning",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Case_Based",
    "Rule_Learning",
]


def load_cora_dataset(root: str = "data/Cora") -> Tuple[Planetoid, Data]:
    """
    Load the Cora citation network dataset via PyTorch Geometric.

    Returns
    -------
    dataset : Planetoid dataset object
    data    : Data object containing x, edge_index, y, train_mask, val_mask, test_mask
    """
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    return dataset, data


def compute_graph_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute edge homophily ratio h:
        h = |{ (u, v) in E : y_u == y_v }| / |E|

    Homophily > 0.5 indicates strong community alignment where
    connected papers share the same subject category.
    """
    src, dst = edge_index[0], edge_index[1]
    # Filter self-loops if any for pure topological homophily
    mask = src != dst
    if mask.sum() == 0:
        return 1.0
    same_label = (y[src[mask]] == y[dst[mask]]).float()
    return same_label.mean().item()


def get_cora_statistics(data: Data) -> Dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics for the Cora graph.
    """
    num_nodes = data.num_nodes
    num_edges = data.num_edges  # directed edge count in PyG
    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

    # Degree distribution
    src = data.edge_index[0]
    degrees = torch.bincount(src, minlength=num_nodes).float()
    isolated_nodes = int((degrees == 0).sum().item())

    # Homophily
    homophily = compute_graph_homophily(data.edge_index, data.y)

    # Split sizes
    n_train = int(data.train_mask.sum().item())
    n_val = int(data.val_mask.sum().item())
    n_test = int(data.test_mask.sum().item())

    # Feature sparsity
    total_entries = data.x.numel()
    non_zero = int((data.x > 0).sum().item())
    sparsity = 1.0 - (non_zero / total_entries)
    mean_words_per_doc = float(data.x.sum(dim=1).mean().item())

    # Class distribution
    class_counts = torch.bincount(data.y, minlength=num_classes).tolist()

    return {
        "num_nodes": num_nodes,
        "num_edges_directed": num_edges,
        "num_edges_undirected": num_edges // 2,
        "num_features": num_features,
        "num_classes": num_classes,
        "mean_degree": float(degrees.mean().item()),
        "min_degree": int(degrees.min().item()),
        "max_degree": int(degrees.max().item()),
        "median_degree": float(degrees.median().item()),
        "isolated_nodes": isolated_nodes,
        "edge_homophily": homophily,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "feature_sparsity": sparsity,
        "mean_words_per_doc": mean_words_per_doc,
        "class_counts": class_counts,
    }


def print_cora_summary(stats: Dict[str, Any]) -> None:
    """Pretty-print Cora dataset inspection summary."""
    print("=" * 60)
    print("  CORA CITATION NETWORK — DATASET INSPECTION")
    print("=" * 60)
    print(f"  Nodes (Documents)          : {stats['num_nodes']}")
    print(f"  Edges (Directed / Undir)   : {stats['num_edges_directed']} / {stats['num_edges_undirected']}")
    print(f"  Node Features (Vocab size) : {stats['num_features']} (binary word indicators)")
    print(f"  Document Classes           : {stats['num_classes']}")
    print(f"  Average Degree             : {stats['mean_degree']:.2f}")
    print(f"  Degree Min / Med / Max     : {stats['min_degree']} / {stats['median_degree']} / {stats['max_degree']}")
    print(f"  Isolated Nodes             : {stats['isolated_nodes']}")
    print(f"  Edge Homophily Ratio       : {stats['edge_homophily'] * 100:.2f}% (intra-class citation probability)")
    print(f"  Feature Sparsity           : {stats['feature_sparsity'] * 100:.2f}% (avg {stats['mean_words_per_doc']:.1f} unique words/doc)")
    print(f"  Dataset Splits             : Train={stats['n_train']}, Val={stats['n_val']}, Test={stats['n_test']}")
    print("=" * 60)
