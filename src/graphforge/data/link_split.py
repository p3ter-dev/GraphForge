"""
Link Prediction Dataset Splitting and Negative Sampling with Leakage Guards
============================================================================
GraphForge Assignment 3: Link Prediction and Data Leakage

Key principles implemented:
1. Positives are split at the unique undirected edge level.
2. Both directions (u, v) and (v, u) are placed strictly into the same split.
3. Message-passing graph E_train contains ONLY training positive edges.
   Held-out validation and test positive edges are strictly excised.
4. Negatives are sampled to exclude self-loops and any positive edge in the full graph.
5. Negative sets are fixed and disjoint across train, val, and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Tuple

import numpy as np
import torch


@dataclass
class LinkDataSplits:
    # Message-passing graph (bidirectional) containing ONLY training edges
    train_message_edge_index: torch.Tensor  # [2, 2 * num_train_edges]

    # Evaluation links (edge pairs [2, num_pairs])
    train_pos_edge_index: torch.Tensor
    train_neg_edge_index: torch.Tensor

    val_pos_edge_index: torch.Tensor
    val_neg_edge_index: torch.Tensor

    test_pos_edge_index: torch.Tensor
    test_neg_edge_index: torch.Tensor

    # Node features (labels y are excluded)
    x: torch.Tensor
    num_nodes: int


def get_unique_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Extract unique undirected edges where u < v, strictly deduplicated.
    """
    src, dst = edge_index[0], edge_index[1]
    # Keep strictly u < v
    mask = src < dst
    filtered_src = src[mask]
    filtered_dst = dst[mask]

    # Deduplicate unique pairs
    stacked = torch.stack([filtered_src, filtered_dst], dim=0)
    unique_pairs = torch.unique(stacked, dim=1)
    return unique_pairs


def sample_negative_pairs(
    num_nodes: int,
    num_samples: int,
    all_positive_set: Set[Tuple[int, int]],
    existing_neg_set: Set[Tuple[int, int]],
    seed: int = 42,
) -> Tuple[torch.Tensor, Set[Tuple[int, int]]]:
    """
    Sample disjoint negative pairs (u, v) where u != v and (u, v) is not in positive set.
    """
    rng = np.random.default_rng(seed)
    neg_list = []
    new_neg_set = set(existing_neg_set)

    # Rejection sampling
    while len(neg_list) < num_samples:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u == v:
            continue
        # Canonical undirected pair
        pair = (min(u, v), max(u, v))
        if pair not in all_positive_set and pair not in new_neg_set:
            new_neg_set.add(pair)
            neg_list.append(pair)

    neg_tensor = torch.tensor(neg_list, dtype=torch.long).t().contiguous()
    return neg_tensor, new_neg_set


def create_link_prediction_splits(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> LinkDataSplits:
    """
    Create rigorously separated positive and negative splits for link prediction.
    """
    rng = np.random.default_rng(seed)
    num_nodes = x.shape[0]

    # Step 1: Extract unique undirected edges (u < v)
    unique_edges = get_unique_undirected_edges(edge_index)
    num_edges = unique_edges.shape[1]

    # Create master set of all observed positive links (canonical undirected)
    all_pos_set = set(
        (int(unique_edges[0, i]), int(unique_edges[1, i]))
        for i in range(num_edges)
    )

    # Step 2: Permute edges and split
    perm = rng.permutation(num_edges)
    n_val = int(num_edges * val_ratio)
    n_test = int(num_edges * test_ratio)

    val_indices = perm[:n_val]
    test_indices = perm[n_val : n_val + n_test]
    train_indices = perm[n_val + n_test :]

    train_pos = unique_edges[:, train_indices]
    val_pos = unique_edges[:, val_indices]
    test_pos = unique_edges[:, test_indices]

    # Step 3: Build bidirectional message-passing graph with ONLY training positive edges
    # E_train contains (u, v) and (v, u) for (u, v) in train_pos
    train_msg_src = torch.cat([train_pos[0], train_pos[1]])
    train_msg_dst = torch.cat([train_pos[1], train_pos[0]])
    train_message_edge_index = torch.stack([train_msg_src, train_msg_dst], dim=0)

    # Step 4: Sample negative pairs for train, val, and test
    neg_seen: Set[Tuple[int, int]] = set()

    train_neg, neg_seen = sample_negative_pairs(num_nodes, train_pos.shape[1], all_pos_set, neg_seen, seed=seed)
    val_neg, neg_seen = sample_negative_pairs(num_nodes, val_pos.shape[1], all_pos_set, neg_seen, seed=seed + 1)
    test_neg, neg_seen = sample_negative_pairs(num_nodes, test_pos.shape[1], all_pos_set, neg_seen, seed=seed + 2)

    return LinkDataSplits(
        train_message_edge_index=train_message_edge_index,
        train_pos_edge_index=train_pos,
        train_neg_edge_index=train_neg,
        val_pos_edge_index=val_pos,
        val_neg_edge_index=val_neg,
        test_pos_edge_index=test_pos,
        test_neg_edge_index=test_neg,
        x=x,
        num_nodes=num_nodes,
    )
