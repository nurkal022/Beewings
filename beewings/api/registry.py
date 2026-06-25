"""Lazy, cached model registry.

Loading a UNet checkpoint takes several seconds and tens of MB of RAM, so we
load each one once on first use and keep it in memory. Each model gets its own
lock because torch inference on a single module is not thread-safe and the API
serves requests from a thread pool.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict

from ..core.profiles import METHODOLOGIES
from ..ml.inference import LoadedModel, load

# methodology id -> checkpoint filename (from the Methodology definitions)
_CHECKPOINT_NAMES: Dict[str, str] = {
    mid: m.checkpoint_name for mid, m in METHODOLOGIES.items()
}


def checkpoints_dir() -> Path:
    """Folder holding the .pt files. Override with BEEWINGS_CHECKPOINTS."""
    env = os.environ.get("BEEWINGS_CHECKPOINTS")
    if env:
        return Path(env)
    # Default: <repo root>/checkpoints (three levels up from this file).
    return Path(__file__).resolve().parents[2] / "checkpoints"


def _device() -> str:
    return os.environ.get("BEEWINGS_DEVICE", "auto")


class _Registry:
    def __init__(self) -> None:
        self._cache: Dict[str, LoadedModel] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def checkpoint_path(self, methodology: str) -> Path:
        name = _CHECKPOINT_NAMES.get(methodology)
        if name is None:
            raise KeyError(
                f"unknown methodology '{methodology}'; "
                f"expected one of {sorted(_CHECKPOINT_NAMES)}"
            )
        return checkpoints_dir() / name

    def lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def get(self, methodology: str) -> LoadedModel:
        """Return the cached model for a methodology, loading it on first use."""
        path = self.checkpoint_path(methodology)
        return self.get_by_path(path)

    def get_by_path(self, path: Path) -> LoadedModel:
        key = str(Path(path).resolve())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with self.lock_for(key):
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            if not Path(path).exists():
                raise FileNotFoundError(f"checkpoint not found: {path}")
            model = load(Path(path), device=_device())
            self._cache[key] = model
            return model

    def loaded(self) -> Dict[str, str]:
        """Map of loaded checkpoint path -> device, for /health."""
        return {k: str(m.device) for k, m in self._cache.items()}

    def available(self) -> Dict[str, dict]:
        """Describe configured methodologies and whether the file exists."""
        out: Dict[str, dict] = {}
        for mid, name in _CHECKPOINT_NAMES.items():
            path = checkpoints_dir() / name
            out[mid] = {
                "checkpoint": name,
                "path": str(path),
                "exists": path.exists(),
            }
        return out


registry = _Registry()
