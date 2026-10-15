"""
Demonstration of Inductive Inference on Previously Unseen Nodes
==============================================================
Assignment 4 Requirement 8:
Demonstrate inference on existing nodes and on previously unseen nodes
introduced after training. Exclude unseen nodes and their incident edges
from the training graph. Explain the required graph context and inference limitations.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F

from graphforge.data.cora import CORA_CLASSES, load_cora_dataset
from graphforge.models.cora_models import CoraGCN
from graphforge.training.cora_trainer import CoraTrainingConfig, set_seed, train_cora_model


def main():
    print("=" * 80)
    print("  ASSIGNMENT 4: INDUCTIVE INFERENCE ON UNSEEN NODES DEMONSTRATION")
    print("=" * 80)

    # 1. Load Cora
    _, data = load_cora_dataset(root="data/Cora")
    set_seed(42)

    total_nodes = data.num_nodes  # 2708
    num_unseen = 100
    num_train_nodes = total_nodes - num_unseen  # 2608

    print("\n[1/4] Partitioning Graph:")
    print(f"  Training Graph Nodes (Seen)   : [0, {num_train_nodes - 1}] ({num_train_nodes} nodes)")
    print(f"  Unseen Nodes (Post-training) : [{num_train_nodes}, {total_nodes - 1}] ({num_unseen} nodes)")

    # 2. Strict Exclusion of Unseen Nodes from Training Graph
    # Filter edges: keep ONLY edges where BOTH endpoints are in [0, num_train_nodes - 1]
    src, dst = data.edge_index[0], data.edge_index[1]
    train_edge_mask = (src < num_train_nodes) & (dst < num_train_nodes)
    train_edge_index = data.edge_index[:, train_edge_mask]

    train_x = data.x[:num_train_nodes]
    train_y = data.y[:num_train_nodes]

    # Sub-split train/val masks
    train_mask = data.train_mask[:num_train_nodes]
    val_mask = data.val_mask[:num_train_nodes]

    print(f"  Total Directed Edges in Full Graph    : {data.num_edges}")
    print(f"  Strict Training Graph Directed Edges  : {train_edge_index.shape[1]}")
    print(f"  Incident Edges Withheld During Train  : {data.num_edges - train_edge_index.shape[1]}")

    # 3. Train Model on Seen Graph Only
    print("\n[2/4] Training GCN on Seen Nodes Only...")
    cfg = CoraTrainingConfig(lr=0.01, max_epochs=120, patience=30, seed=42, hidden_dim=64)
    model = CoraGCN(in_features=data.num_features, hidden_dim=64, num_classes=7, dropout=0.5)
    model, _ = train_cora_model(model, train_x, train_edge_index, train_y, train_mask, val_mask, cfg)
    print("  Training finished!")

    # 4. Inference on Existing (Seen) Nodes
    model.eval()
    with torch.no_grad():
        out_seen = model(train_x, train_edge_index)
        preds_seen = out_seen.argmax(dim=-1)
        # Evaluate on valid test nodes within seen graph
        test_mask_seen = data.test_mask[:num_train_nodes]
        acc_seen = (preds_seen[test_mask_seen] == train_y[test_mask_seen]).float().mean().item()

    print("\n[3/4] Inference on Existing (Seen) Nodes:")
    print(f"  Test Accuracy on Seen Nodes: {acc_seen * 100:.2f}%")

    # 5. Inductive Inference on Previously Unseen Nodes
    # We now present the unseen nodes along with their local citation context (edges to seen nodes)
    print("\n[4/4] Inductive Inference on Previously Unseen Nodes:")
    with torch.no_grad():
        # Full graph includes all nodes and all connections
        out_full = model(data.x, data.edge_index)
        preds_full = out_full.argmax(dim=-1)

        unseen_indices = torch.arange(num_train_nodes, total_nodes)
        unseen_preds = preds_full[unseen_indices]
        unseen_targets = data.y[unseen_indices]
        acc_unseen = (unseen_preds == unseen_targets).float().mean().item()

    print(f"  Accuracy on Previously Unseen Nodes: {acc_unseen * 100:.2f}%")

    # Sample predictions
    print("\n  Sample Predictions on Unseen Documents:")
    for i in range(5):
        node_id = num_train_nodes + i
        true_c = int(unseen_targets[i])
        pred_c = int(unseen_preds[i])
        prob = F.softmax(out_full[node_id], dim=-1)[pred_c].item()
        match = "✓" if true_c == pred_c else "✗"
        print(f"    Node {node_id} | True: {CORA_CLASSES[true_c]:<22} | Pred: {CORA_CLASSES[pred_c]:<22} | Conf: {prob*100:.1f}% [{match}]")

    print("\n" + "=" * 80)
    print("  GRAPH CONTEXT REQUIREMENTS AND INFERENCE LIMITATIONS:")
    print("  -----------------------------------------------------")
    print("  1. Required Graph Context:")
    print("     GNNs require local neighborhood connectivity to propagate messages. To classify an")
    print("     unseen node inductively, the inference service must receive its incident edges")
    print("     connecting it to existing nodes or other candidate nodes.")
    print("  2. Isolated Node Degradation:")
    print("     If an unseen document arrives with zero citations (an isolated node), the GCN receives")
    print("     no neighborhood messages and must rely solely on the node's self-loop (Ã_ii = 1).")
    print("     Its performance gracefully degrades to that of a feature projection (MLP-equivalent).")
    print("  3. Scalability / Subgraph Extraction:")
    print("     In production with massive graphs (millions of nodes), full-graph message passing is")
    print("     infeasible. The serving layer must extract an ego-network (k-hop neighborhood)")
    print("     around the target nodes using PyG NeighborLoader before calling model.forward().")
    print("=" * 80)


if __name__ == "__main__":
    main()
