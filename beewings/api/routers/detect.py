"""POST /detect — run a trained UNet on a wing image and return landmarks.

Accepts either a multipart upload (``image``) or a JSON body with
``image_path``. Classical indices are computed automatically when the chosen
methodology is Alpatov-12.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ...core.indices import classify_by_ci, compute_all_alpatov
from ...core.profiles import DEFAULT_PROFILE_PER_METHODOLOGY, get_profile
from ...ml.inference import predict
from ..io import read_image
from ..registry import registry
from ..schemas import (DetectJSONRequest, DetectResponse, IndexValue,
                       LandmarkPoint)

router = APIRouter()


def _run_detect(bgr, methodology: str, tta: bool,
                confidence_min: float, return_confidence: bool) -> DetectResponse:
    try:
        lm = registry.get(methodology)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    h, w = bgr.shape[:2]
    lock = registry.lock_for(str(registry.checkpoint_path(methodology).resolve()))
    with lock:
        coords, confs = predict(lm, bgr, tta=tta, return_confidence=True)

    profile_name = DEFAULT_PROFILE_PER_METHODOLOGY.get(methodology, "")
    points = []
    kept_coords = {}
    for lid in sorted(coords):
        x, y = coords[lid]
        c = confs[lid]
        if c < confidence_min:
            continue
        kept_coords[lid] = (x, y)
        points.append(LandmarkPoint(
            id=lid, x=x, y=y,
            confidence=(c if return_confidence else None),
            uncertain=(c < 0.5),
        ))

    indices = None
    subspecies = None
    if methodology == "alpatov" and lm.n_points == 12:
        results = compute_all_alpatov(kept_coords)
        indices = [IndexValue(name=r.name, value=r.value,
                              formula=r.formula, notes=r.notes) for r in results]
        ci = next((r.value for r in results if r.name.startswith("CI (Алпатов")), None)
        if ci is not None:
            subspecies = classify_by_ci(ci)

    return DetectResponse(
        methodology=methodology,
        profile=profile_name,
        n_points=lm.n_points,
        image_size=(w, h),
        landmarks=points,
        indices=indices,
        ci_subspecies=subspecies,
        notes=f"{len(points)}/{lm.n_points} landmarks kept "
              f"(confidence_min={confidence_min}, tta={tta})",
    )


@router.post("/detect", response_model=DetectResponse)
async def detect(
    request: Request,
    image: Optional[UploadFile] = File(default=None),
    methodology: str = Form(default="tofilski"),
    tta: bool = Form(default=False),
    confidence_min: float = Form(default=0.0),
    return_confidence: bool = Form(default=True),
) -> DetectResponse:
    # JSON body (server-side path) vs multipart upload.
    if image is None and request.headers.get("content-type", "").startswith("application/json"):
        body = DetectJSONRequest(**(await request.json()))
        bgr = read_image(None, body.image_path)
        return _run_detect(bgr, body.methodology, body.tta,
                           body.confidence_min, body.return_confidence)

    if image is None:
        raise HTTPException(status_code=400,
                            detail="provide an uploaded 'image' or a JSON body with image_path")
    data = await image.read()
    bgr = read_image(data, None)
    return _run_detect(bgr, methodology, tta, confidence_min, return_confidence)
