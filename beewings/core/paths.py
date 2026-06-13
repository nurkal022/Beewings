"""Resource path resolution that works both from source and a frozen exe.

When packaged with PyInstaller (``sys.frozen``), the working directory is
wherever the user launched the .exe from, not where the app and its bundled
``checkpoints/`` live. Relative paths like ``"checkpoints/alpatov12.pt"`` must
therefore be resolved against the app's own directory, not the CWD.
"""
from __future__ import annotations

import sys
from pathlib import Path


def app_base_dir() -> Path:
    """Directory the app lives in: the .exe folder when frozen, else repo root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resolve_checkpoint(spec) -> Path:
    """Resolve a checkpoint path/spec to a concrete file.

    Tries, in order: an absolute path as-is; the path relative to the current
    directory; then relative to :func:`app_base_dir`. Returns the first that
    exists, otherwise the app-base candidate (so callers get a sensible path to
    report as missing).
    """
    p = Path(spec)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return app_base_dir() / p
