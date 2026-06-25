"""Pydantic request/response models for the API.

These are API-facing DTOs and are intentionally separate from the storage
schema in ``beewings.core.schema`` (which carries timestamps, versions, etc.).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# --- shared ---------------------------------------------------------------

class LandmarkPoint(BaseModel):
    id: int
    x: float
    y: float
    confidence: Optional[float] = None
    uncertain: bool = False


class IndexValue(BaseModel):
    name: str
    value: Optional[float]
    formula: str
    notes: str = ""


# --- /detect --------------------------------------------------------------

class DetectJSONRequest(BaseModel):
    """Body for JSON-mode detection (server-side path)."""
    image_path: str
    methodology: str = "tofilski"
    tta: bool = False
    confidence_min: float = 0.0
    return_confidence: bool = True


class DetectResponse(BaseModel):
    methodology: str
    profile: str
    n_points: int
    image_size: Tuple[int, int]  # (width, height)
    landmarks: List[LandmarkPoint]
    indices: Optional[List[IndexValue]] = None
    ci_subspecies: Optional[List[str]] = None
    notes: str = ""


# --- /segment -------------------------------------------------------------

class SegmentJSONRequest(BaseModel):
    scan_path: str
    margin: float = 0.10
    min_area_frac: float = 0.0003
    rotate: bool = False
    label_frac: Optional[float] = None
    output: str = "inline"          # "inline" | "files"
    out_dir: Optional[str] = None   # required when output == "files"


class WingBox(BaseModel):
    index: int
    x: int
    y: int
    w: int
    h: int
    crop_b64: Optional[str] = None  # JPEG base64, only for output=inline


class SegmentResponse(BaseModel):
    n_wings: int
    output: str
    boxes: List[WingBox] = Field(default_factory=list)
    files: List[str] = Field(default_factory=list)  # only for output=files


# --- /indices -------------------------------------------------------------

class IndicesRequest(BaseModel):
    """Alpatov 12-point landmarks → classical indices."""
    landmarks: List[LandmarkPoint]


class IndicesResponse(BaseModel):
    indices: List[IndexValue]
    ci_subspecies: List[str] = Field(default_factory=list)


# --- /export --------------------------------------------------------------

class ExportWing(BaseModel):
    wing: str
    landmarks: List[LandmarkPoint]


class ExportRequest(BaseModel):
    methodology: str = "alpatov"          # "alpatov" | "tofilski"
    profile: Optional[str] = None         # override profile name (else default per methodology)
    wings: List[ExportWing]
    image_size: Tuple[int, int] = (0, 0)  # (width, height) for TPS metadata
    output: str = "download"              # "download" | "files"
    out_dir: Optional[str] = None         # required when output == "files"


class ExportFilesResponse(BaseModel):
    output: str
    files: Dict[str, str]                 # format -> path
