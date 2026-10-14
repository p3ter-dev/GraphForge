# GraphForge

**Graph Neural Network Engineering and MLOps** — A complete, reproducible GNN project covering four assignments: custom GCN implementation, architecture comparison on Cora, link prediction with leakage guards, and a full MLOps pipeline.

---

## Quick Start

### Prerequisites
- Linux / macOS
- [`uv`](https://docs.astral.sh/uv/) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Git

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/graphforge.git
cd graphforge

# Create virtual environment with Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install all dependencies (CPU torch; see below for GPU)
uv pip install -e ".[dev]" \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

**GPU setup (optional):**
```bash
# For CUDA 12.1
uv pip install -e ".[dev]" \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple
```

### Verify Installation

```bash
python -c "import torch, torch_geometric, mlflow; print('All imports OK')"
pytest tests/ -v --co -q   # list tests without running
```

---

## Project Structure

```
graphforge/
├── src/graphforge/          # Python package (models, data, training, serving)
├── notebooks/               # Jupyter notebooks with saved outputs
├── configs/                 # YAML experiment configurations
├── tests/                   # Unit, integration, correctness, and leakage tests
├── scripts/                 # Training and evaluation entrypoints
├── reports/                 # Figures, tables, and technical report
├── artifacts/               # Local model artifacts (not committed if large)
├── api/                     # FastAPI serving application
├── docker/                  # Dockerfile and compose
└── .github/workflows/       # CI/CD pipelines
```

---

## Assignments

### Assignment 1: GCN from Scratch

**Goal:** Build and verify a two-layer GCN using only PyTorch tensor operations.

```bash
# Train all experiments (GCN, MLP, randomized edges)
python scripts/train_gcn_scratch.py

# Run correctness and equivariance tests
pytest tests/correctness/ -v

# Open educational notebook
jupyter notebook notebooks/01_gcn_from_scratch.ipynb
```

**What it demonstrates:**
- Self-loops, degree normalization, symmetric Â computation
- Manual layer-output verification (NumPy vs PyTorch, ~1e-5 tolerance)
- Permutation equivariance (a fundamental GNN correctness property)
- GCN vs MLP vs randomized-edge ablation results

### Assignment 2: Architecture Comparison on Cora

**Goal:** Benchmark and analyze MLP, GCN, GraphSAGE, and GAT on document classification.

```bash
# Run complete benchmark suite, ablations, and error analysis
python scripts/train_cora_comparison.py

# Run unit tests
pytest tests/unit/test_cora_models.py -v

# Open educational notebook
jupyter notebook notebooks/02_cora_comparison.ipynb
```

**What it demonstrates:**
- Graph homophily (81.0%) as a source of semi-supervised signal (+23% GNN over MLP)
- Over-smoothing analysis: depth vs accuracy & representation cosine similarity collapse
- Edge sparsification: gradual accuracy decay from 80.6% down to 57.1% as edges are removed
- Diagnostic error analysis on misclassified boundary/heterophilic citations

### Assignment 3: Link Prediction and Data Leakage Guards

**Goal:** Build an unsupervised link prediction system on Cora without category labels, comparing decoders and exposing data leakage.

```bash
# Run link prediction benchmark, leakage demonstration, and candidate extraction
python scripts/train_link_prediction.py

# Run leakage guard automated tests
pytest tests/leakage/test_link_leakage.py -v

# Open educational notebook
jupyter notebook notebooks/03_link_prediction.ipynb
```

**What it demonstrates:**
- Strict message-passing isolation: held-out positive links and reverse copies excised from $E_{\text{train}}$
- Mathematical decoder symmetry: guaranteed $s(u, v) \equiv s(v, u)$ via commutative operator concatenation
- Learned GNN + Symmetric MLP (92.29% ROC-AUC) vs Common Neighbors heuristic (72.12%)
- Deliberately leaky evaluation exposing artificial metric inflation (94.17% ROC-AUC)
- Top 10 candidate citation discoveries under the Open-World Assumption

### Assignment 4: MLOps Pipeline *(coming in Milestone 4)*

```bash
# Start MLflow UI
mlflow ui --port 5000

# Train with tracking
python scripts/train_gcn_scratch.py  # tracked version in M4

# Start API
uvicorn api.main:app --reload

# Docker
docker compose -f docker/docker-compose.yml up
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Just correctness checks
pytest tests/correctness/ -v

# With coverage
pytest tests/ --cov=src/graphforge --cov-report=term-missing
```

---

## Reproducibility

Every experiment uses:
- Explicit random seeds (Python `random`, NumPy, PyTorch)
- Fixed synthetic data generation via `numpy.random.default_rng(seed)`
- Hyperparameters stored in `configs/*.yaml`
- Locked dependency tree via `uv.lock`

To reproduce exactly:
```bash
uv sync                  # install from lockfile
python scripts/train_gcn_scratch.py --seed 42
```

---

## Results Summary

### Assignment 1 (Synthetic Two-Community Graph)

| Model | Val Acc | Test Acc | Multi-seed (Mean ± Std) | Params |
|-------|---------|----------|-------------------------|--------|
| GCN (original graph) | 1.0000 | 1.0000 | **100.00% ± 0.00%** | 386 |
| MLP (features only) | 0.7917 | 0.8333 | **81.94% ± 8.56%** | 386 |
| GCN (randomized edges) | 0.7083 | 0.7500 | **59.72% ± 11.95%** | 386 |

### Assignment 2 (Cora Document Classification)

| Model | Test Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | Training Time (Mean) | Params |
|-------|----------------------------|-----------------------|----------------------|--------|
| **GCN** | **80.60% ± 0.42%** | **0.8006 ± 0.0031** | 8.83s | 92,231 |
| **GAT** | **80.40% ± 0.24%** | **0.7982 ± 0.0018** | 17.14s | 92,373 |
| **GraphSAGE** | **78.83% ± 0.17%** | **0.7829 ± 0.0018** | 11.96s | 184,391 |
| **MLP (Baseline)** | **57.63% ± 0.96%** | **0.5621 ± 0.0087** | 3.24s | 92,231 |

### Assignment 3 (Cora Link Prediction)

| Model / Methodology | Message Passing | Test ROC-AUC | Test Average Precision | Status |
|---------------------|-----------------|--------------|------------------------|--------|
| **Common Neighbors (Baseline)** | $E_{\text{train}}$ | 0.7212 | 0.7192 | Non-parametric Baseline |
| **GNN + Dot-Product Decoder** | $E_{\text{train}}$ | 0.8692 | 0.8723 | Valid Evaluation |
| **GNN + Symmetric MLP Decoder** | $E_{\text{train}}$ | **0.9229** | **0.9344** | **Valid Champion** |
| **GNN + Dot-Product (LEAKY)** | $E_{\text{full}}$ | 0.9417 | 0.9277 | **Fraudulent / Invalid** |

---

## Technical Stack

| Component | Tool | Version |
|-----------|------|---------|
| Language | Python | 3.12 |
| Deep learning | PyTorch | ≥2.3 |
| Graph ML | PyTorch Geometric | ≥2.5 |
| Experiment tracking | MLflow | ≥2.14 |
| API serving | FastAPI | ≥0.111 |
| Container | Docker | — |
| CI | GitHub Actions | — |
| Package manager | uv | ≥0.13 |

---

## Hardware Notes

All experiments are designed to run on CPU in reasonable time (minutes, not hours) since the datasets are small (Cora: 2708 nodes). GPU is supported automatically if CUDA is available and the CUDA wheel is installed.

---

## License

MIT
