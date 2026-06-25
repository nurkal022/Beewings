"""Image input helpers shared by the routers.

Two input modes are supported everywhere:
  * an uploaded file (multipart), decoded in memory, and
  * a path on the server's filesystem.

Server-side paths are confined to BEEWINGS_DATA_ROOT (when set) to prevent
path-traversal into arbitrary files.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException


def data_root() -> Optional[Path]:
    """Allowed root for server-side paths, or None (no restriction) if unset."""
    env = os.environ.get("BEEWINGS_DATA_ROOT")
    return Path(env).resolve() if env else None


def resolve_path(raw: str) -> Path:
    """Resolve a user-supplied server path, confined to BEEWINGS_DATA_ROOT."""
    p = Path(raw).expanduser().resolve()
    root = data_root()
    if root is not None and root not in p.parents and p != root:
        raise HTTPException(
            status_code=403,
            detail=f"path '{raw}' is outside the allowed data root",
        )
    return p


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR ndarray."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="cannot decode uploaded image")
    return bgr


def read_image(upload_bytes: Optional[bytes], image_path: Optional[str]) -> np.ndarray:
    """Load a BGR image from an upload OR a server path. Exactly one is required."""
    if upload_bytes is not None and image_path:
        raise HTTPException(
            status_code=400,
            detail="provide either an uploaded file or image_path, not both",
        )
    if upload_bytes is not None:
        return decode_image(upload_bytes)
    if image_path:
        p = resolve_path(image_path)
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise HTTPException(status_code=400, detail=f"cannot read image: {p}")
        return bgr
    raise HTTPException(
        status_code=400, detail="no image provided (file upload or image_path)"
    )
