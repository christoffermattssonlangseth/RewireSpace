"""Compatibility shim for legacy imports.

Prefer:
    from rewirespace.spatial_rewiring import ...
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from rewirespace.spatial_rewiring import *  # noqa: F401,F403
except ModuleNotFoundError:
    src_path = Path(__file__).resolve().parent / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from rewirespace.spatial_rewiring import *  # noqa: F401,F403
