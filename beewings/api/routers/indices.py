"""POST /indices — classical Alpatov-12 morphometric indices from landmarks."""
from __future__ import annotations

from fastapi import APIRouter

from ...core.indices import classify_by_ci, compute_all_alpatov
from ..schemas import IndicesRequest, IndicesResponse, IndexValue

router = APIRouter()


@router.post("/indices", response_model=IndicesResponse)
def indices(body: IndicesRequest) -> IndicesResponse:
    coords = {p.id: (p.x, p.y) for p in body.landmarks}
    results = compute_all_alpatov(coords)
    values = [IndexValue(name=r.name, value=r.value, formula=r.formula, notes=r.notes)
              for r in results]
    ci = next((r.value for r in results if r.name.startswith("CI (Алпатов")), None)
    subspecies = classify_by_ci(ci) if ci is not None else []
    return IndicesResponse(indices=values, ci_subspecies=subspecies)
