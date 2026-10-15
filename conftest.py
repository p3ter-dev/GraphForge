"""
Root conftest.py — ensures src/ is on sys.path for all test runs.
This is the standard pattern for src-layout projects.
"""

import sys
from pathlib import Path

# Make sure `src/graphforge` is importable regardless of install mode
sys.path.insert(0, str(Path(__file__).parent / "src"))
