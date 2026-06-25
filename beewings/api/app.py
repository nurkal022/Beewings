"""FastAPI application for BeeWings.

A thin HTTP layer over the existing core functions. Endpoints:
    GET  /health   — liveness + loaded models + device
    GET  /models   — configured methodologies / checkpoints
    POST /detect   — image -> landmarks (+ indices for Alpatov-12)
    POST /segment  — scan -> per-wing crops (inline base64 or files)
    POST /indices  — landmarks -> classical indices
    POST /export   — landmarks -> TPS / XLSX / JSON
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.profiles import METHODOLOGIES
from .io import data_root
from .registry import checkpoints_dir, registry
from .routers import detect, export, indices, segment

app = FastAPI(
    title="BeeWings API",
    version="0.1.0",
    description="Bee wing morphometry: segmentation, ML landmark detection, "
                "classical indices, and export (TPS/XLSX/JSON).",
)

# CORS — permissive by default; restrict with BEEWINGS_CORS_ORIGINS (comma list).
_origins = os.environ.get("BEEWINGS_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",")] if _origins else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router, tags=["detect"])
app.include_router(segment.router, tags=["segment"])
app.include_router(indices.router, tags=["indices"])
app.include_router(export.router, tags=["export"])


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "checkpoints_dir": str(checkpoints_dir()),
        "data_root": str(data_root()) if data_root() else None,
        "loaded_models": registry.loaded(),
    }


@app.get("/models", tags=["meta"])
def models() -> dict:
    out = {}
    for mid, m in METHODOLOGIES.items():
        info = registry.available()[mid]
        out[mid] = {
            "display_name": m.display_name,
            "reference": m.reference,
            "checkpoint": info["checkpoint"],
            "checkpoint_exists": info["exists"],
        }
    return {"methodologies": out}
