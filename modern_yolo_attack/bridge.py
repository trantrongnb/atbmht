from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
FALLBACK_OLD_REPO_ROOT = WORKSPACE_ROOT / "Naturalistic-Adversarial-Patch"
FALLBACK_MODERN_SOURCE_REPO = WORKSPACE_ROOT / "NaturalisticAdversarialPatches"


def _resolve_legacy_root() -> Path:
    """Ưu tiên dependency đã được bundle ngay trong project hiện tại."""
    if (PROJECT_ROOT / "adversarialYolo").exists() and (PROJECT_ROOT / "GANLatentDiscovery").exists():
        return PROJECT_ROOT
    return FALLBACK_OLD_REPO_ROOT


def _resolve_modern_source_root() -> Path:
    """Ưu tiên ultralytics local trong project hiện tại."""
    if (PROJECT_ROOT / "ultralytics").exists():
        return PROJECT_ROOT
    return FALLBACK_MODERN_SOURCE_REPO


OLD_REPO_ROOT = _resolve_legacy_root()
MODERN_SOURCE_REPO = _resolve_modern_source_root()


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def bootstrap_paths() -> None:
    _prepend_sys_path(MODERN_SOURCE_REPO)
    _prepend_sys_path(OLD_REPO_ROOT)


def import_yolo_class():
    bootstrap_paths()
    try:
        from ultralytics import YOLO
    except ImportError:
        if not (MODERN_SOURCE_REPO / "ultralytics").exists():
            raise
        from ultralytics import YOLO
    return YOLO
