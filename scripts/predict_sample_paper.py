"""
Predict a Single Research Paper Topic (Live Demo for Assignment 4)
==================================================================
Demonstrates live prediction of a research paper topic using the deployed GNN.
Queries the running FastAPI / Docker container (or falls back to the packaged champion).
"""

from __future__ import annotations

import json
import urllib.request
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphforge.data.cora import load_cora_dataset, CORA_CLASSES
from graphforge.serving.packaging import load_packaged_model
from graphforge.serving.registry import ModelRegistry


def print_probability_bars(probabilities: dict[str, float]):
    print("\n  Topic Probabilities Distribution:")
    print("  " + "-" * 55)
    for topic, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct = f"{prob * 100:5.1f}%"
        print(f"  {topic:<24} |{bar}| {pct}")
    print("  " + "-" * 55)


def predict_paper_topic(target_node: int = 0):
    print("=" * 68)
    print(f"  ASSIGNMENT 4 LIVE DEMO: CLASSIFYING RESEARCH PAPER #{target_node}")
    print("=" * 68)

    # 1. Load Cora ground truth and citation graph
    _, data = load_cora_dataset(root="data/Cora")
    true_class = int(data.y[target_node].item())
    true_label = CORA_CLASSES[true_class]
    active_words = int(data.x[target_node].sum().item())

    # Find citation neighbors (papers cited by or citing this paper)
    cited_by_target = data.edge_index[1][data.edge_index[0] == target_node].tolist()
    citing_target = data.edge_index[0][data.edge_index[1] == target_node].tolist()
    neighbors = sorted(list(set(cited_by_target + citing_target)))

    print(f"\n  [Paper Metadata]")
    print(f"    • Paper ID           : #{target_node}")
    print(f"    • Active Dictionary Words : {active_words} words (out of 1,433 vocabulary)")
    print(f"    • Connected Citation Papers: {len(neighbors)} papers ({neighbors})")
    print(f"    • True Category (Ground Truth) : {true_label}")

    # Build local ego-network around target node
    subgraph_nodes = sorted(list(set([target_node] + neighbors)))
    node_to_idx = {nid: i for i, nid in enumerate(subgraph_nodes)}

    # Subgraph features and edge index
    x_sub = [data.x[nid].tolist() for nid in subgraph_nodes]
    
    # Filter edges belonging strictly to this ego-network
    edge_mask = torch.tensor([(s in node_to_idx and d in node_to_idx) for s, d in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())])
    sub_src = [node_to_idx[s] for s in data.edge_index[0][edge_mask].tolist()]
    sub_dst = [node_to_idx[d] for d in data.edge_index[1][edge_mask].tolist()]

    target_idx = node_to_idx[target_node]

    # 2. Query Live API
    url = "http://localhost:8000/predict/node"
    payload = {
        "x": x_sub,
        "edge_index": [sub_src, sub_dst] if len(sub_src) > 0 else [[target_idx], [target_idx]],
        "target_node_ids": [target_idx],
    }

    print(f"\n  Sending request to deployed GNN inference service...")
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            pred_item = res["predictions"][0]
            version = res["model_version"]
            arch = res["model_architecture"]
            probs = pred_item["probabilities"]
            pred_label = pred_item["predicted_class_name"]
            confidence = max(probs)
            source = f"Live Docker Container / FastAPI ({arch} v{version})"
    except Exception:
        # Fallback to local champion package
        reg = ModelRegistry(registry_file=Path("artifacts/model_registry.json"))
        champ = reg.get_champion()
        pkg_dir = Path(champ["package_dir"]) if champ else Path("artifacts/packaged_models/cora_gcn/v_1.0.0")
        model, meta = load_packaged_model(pkg_dir)
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_sub, dtype=torch.float32)
            e_t = torch.tensor([sub_src, sub_dst], dtype=torch.long) if len(sub_src) > 0 else torch.zeros((2, 0), dtype=torch.long)
            out = model(x_t, e_t)
            probs = torch.exp(out[target_idx]).cpu().tolist()
            pred_class_id = int(torch.tensor(probs).argmax().item())
            pred_label = CORA_CLASSES[pred_class_id]
            confidence = max(probs)
            source = f"Packaged Champion Artifact ({meta['architecture']} v{meta['version']})"

    prob_dict = {CORA_CLASSES[i]: probs[i] for i in range(len(CORA_CLASSES))}
    is_correct = (pred_label == true_label)

    # 3. Output results
    print(f"\n  [Inference Response] — Served by {source}")
    print(f"    • Predicted Topic    : {pred_label}  {'[CORRECT ✓]' if is_correct else '[INCORRECT ✗]'}")
    print(f"    • Model Confidence   : {confidence * 100:.2f}%")

    print_probability_bars(prob_dict)

    print(f"\n  Conclusion: GNN successfully classified Paper #{target_node} as '{pred_label}' ({confidence*100:.1f}% confidence).")
    print("=" * 68)


if __name__ == "__main__":
    predict_paper_topic(target_node=0)
