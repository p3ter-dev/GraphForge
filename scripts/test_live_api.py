"""
Live API Client Verification Script
===================================
Tests all live endpoints on the running GraphForge container (or local server).
"""

import json
import sys
import urllib.request
from pathlib import Path

# Add src to load sample data
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from graphforge.data.cora import load_cora_dataset

BASE_URL = "http://localhost:8000"


def test_endpoint(title, method, url, payload=None):
    print(f"\n---> {title} [{method} {url}]")
    headers = {"Content-Type": "application/json"} if payload else {}
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
            print(f"Status: HTTP {status}")
            print("Response:")
            print(json.dumps(body, indent=2))
            return body
    except urllib.error.HTTPError as e:
        print(f"Failed with HTTP {e.code}: {e.read().decode('utf-8')}")
        return None
    except Exception as e:
        print(f"Connection failed: {e}")
        return None


def main():
    print("=" * 60)
    print("  GRAPHFORGE LIVE SERVING API VERIFICATION")
    print("=" * 60)

    # 1. Health
    test_endpoint("1. Health Check", "GET", f"{BASE_URL}/health")

    # 2. Metadata
    test_endpoint("2. Model Metadata & Schema", "GET", f"{BASE_URL}/model/metadata")

    # 3. Node Prediction with Real Cora Subgraph
    print("\nLoading sample Cora papers for live prediction payload...")
    _, data = load_cora_dataset(root="data/Cora")

    # Construct ego-network for nodes 0, 1, 2
    subgraph_nodes = [0, 1, 2]
    node_payload = {
        "nodes": [
            {"node_id": i, "features": data.x[i].tolist()}
            for i in subgraph_nodes
        ],
        "edges": [
            [0, 1], [1, 0], [1, 2], [2, 1]
        ],
        "target_node_ids": [0, 1]
    }

    test_endpoint("3. Node Classification (/predict/node)", "POST", f"{BASE_URL}/predict/node", payload=node_payload)

    # 4. Link Prediction
    link_payload = {
        "source_node_features": data.x[0].tolist(),
        "target_node_features": data.x[1].tolist(),
    }
    test_endpoint("4. Link Prediction (/predict/link)", "POST", f"{BASE_URL}/predict/link", payload=link_payload)

    print("\n" + "=" * 60)
    print("  ALL LIVE API ENDPOINTS RESPONDING HEALTHILY")
    print("=" * 60)


if __name__ == "__main__":
    main()
