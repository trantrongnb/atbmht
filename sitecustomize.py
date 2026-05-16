"""Project-local Python path bootstrap for bundled legacy dependencies."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _prepend(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


for candidate in (
    PROJECT_ROOT,
    PROJECT_ROOT / "adversarialYolo",
    PROJECT_ROOT / "GANLatentDiscovery",
    PROJECT_ROOT / "ultralytics",
):
    _prepend(candidate)
