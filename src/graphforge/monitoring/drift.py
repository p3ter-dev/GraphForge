"""
Graph Topology and Embedding Drift Monitoring Module
====================================================
GraphForge Assignment 4: Reproducible GNN MLOps Pipeline

Monitors:
  1. Graph Structure Statistics: Average degree, density, clustering coefficient, isolated nodes.
  2. Embedding Representation Statistics: L2 norm distribution, pairwise cosine similarity,
     and distribution drift (Wasserstein distance & Kolmogorov-Smirnov test).
  3. Mathematical explanation of latent coordinate rotation / permutation ambiguity.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp, wasserstein_distance


def compute_graph_structure_metrics(edge_index: torch.Tensor, num_nodes: int) -> Dict[str, float]:
    """
    Compute structural topological statistics of a graph snapshot.
    """
    if edge_index.numel() == 0:
        return {
            "num_nodes": float(num_nodes),
            "num_edges_directed": 0.0,
            "avg_degree": 0.0,
            "graph_density": 0.0,
            "isolated_node_ratio": 1.0,
        }

    src = edge_index[0]
    degrees = torch.bincount(src, minlength=num_nodes).float()
    num_edges = float(edge_index.shape[1])

    avg_degree = float(degrees.mean().item())
    max_possible_directed_edges = float(num_nodes * (num_nodes - 1))
    density = num_edges / max_possible_directed_edges if max_possible_directed_edges > 0 else 0.0
    isolated_ratio = float((degrees == 0).sum().item()) / float(num_nodes)

    return {
        "num_nodes": float(num_nodes),
        "num_edges_directed": num_edges,
        "avg_degree": avg_degree,
        "graph_density": density,
        "isolated_node_ratio": isolated_ratio,
    }


def compute_embedding_distribution_metrics(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Compute summary statistics of latent embedding representations.
    """
    # L2 norms
    norms = torch.norm(embeddings, p=2, dim=-1).cpu().numpy()
    
    # Mean pairwise cosine similarity
    normed = F.normalize(embeddings, p=2, dim=-1)
    # Subsample if large for efficiency
    if embeddings.shape[0] > 400:
        indices = torch.randperm(embeddings.shape[0])[:400]
        normed = normed[indices]

    sim_mat = torch.mm(normed, normed.t())
    mask = ~torch.eye(normed.shape[0], dtype=torch.bool, device=embeddings.device)
    mean_cos_sim = float(sim_mat[mask].mean().item())

    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "mean_cosine_similarity": mean_cos_sim,
    }


def detect_distribution_drift(
    baseline_embeddings: torch.Tensor,
    current_embeddings: torch.Tensor,
    alpha: float = 0.05,
    wasserstein_threshold: float = 0.30,
) -> Dict[str, Any]:
    """
    Compare embedding norm distributions between baseline snapshot and current snapshot.
    Uses Kolmogorov-Smirnov two-sample test and Wasserstein distance.
    """
    base_norms = torch.norm(baseline_embeddings, p=2, dim=-1).detach().cpu().numpy()
    curr_norms = torch.norm(current_embeddings, p=2, dim=-1).detach().cpu().numpy()

    # 1. Wasserstein distance (Earth Mover's Distance)
    w_dist = float(wasserstein_distance(base_norms, curr_norms))

    # 2. Kolmogorov-Smirnov test
    ks_stat, p_value = ks_2samp(base_norms, curr_norms)

    drift_detected = (w_dist > wasserstein_threshold) or (p_value < alpha)

    return {
        "wasserstein_distance": w_dist,
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "drift_detected": bool(drift_detected),
        "alert_reason": (
            f"Embedding drift detected: Wasserstein distance {w_dist:.4f} > {wasserstein_threshold} "
            f"or KS p-value {p_value:.2e} < {alpha}"
            if drift_detected else "No significant embedding drift detected."
        ),
    }


def explain_embedding_incomparability() -> str:
    """
    Detailed explanation of why raw embedding coordinates from different
    model versions cannot be directly compared.
    """
    return (
        "WHY RAW EMBEDDING COORDINATES FROM DIFFERENT MODEL VERSIONS CANNOT BE DIRECTLY COMPARED:\n"
        "---------------------------------------------------------------------------------------\n"
        "Neural network training with stochastic gradient descent (SGD) optimizes representations up to\n"
        "an arbitrary orthogonal transformation and permutation ambiguity.\n\n"
        "1. Orthogonal Invariance: In linear layers and inner-product decoders, for any orthogonal matrix\n"
        "   Q in O(d) (where Q^T Q = I), the inner product is strictly preserved:\n"
        "       <Q z_u, Q z_v> = z_u^T Q^T Q z_v = z_u^T z_v\n"
        "   Two identically trained GCN models with different initial random seeds or epochs will discover\n"
        "   latent representations that are rotated or reflected relative to one another in R^D.\n\n"
        "2. Coordinate Permutation: The ordering of hidden neuron dimensions is arbitrary. Neuron 4 in\n"
        "   Model Version 1 might encode the semantic concept learned by Neuron 27 in Model Version 2.\n\n"
        "3. Correct Comparison Methodology:\n"
        "   - Monitor invariant scalar statistics (e.g. L2 norm distributions, pairwise cosine similarities,\n"
        "     internal neighborhood clustering coefficients).\n"
        "   - If direct vector comparison across versions is required, align the latent spaces first\n"
        "     using Orthogonal Procrustes analysis on shared anchor nodes:\n"
        "         min_{Q in O(d)} || Z_1 - Z_2 Q ||_F\n"
    )
