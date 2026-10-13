"""
Utility functions used across GraphForge.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers: Python built-in random, NumPy, PyTorch CPU and CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def file_md5(path: str | Path) -> str:
    """
    Compute the MD5 hash of a file.
    Useful for logging dataset identity in experiment tracking.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_md5(t: torch.Tensor) -> str:
    """Compute MD5 of a tensor's bytes — for tracking dataset tensors."""
    return hashlib.md5(t.numpy().tobytes()).hexdigest()
