"""Test helper: write a landmark annotation for a scan's first crop."""
from __future__ import annotations

from pathlib import Path

from beewings.core.profiles import get_profile
from beewings.core.schema import (Landmark, WingAnnotation, annotation_path,
                                  save_annotation)


def annotate_first_crop(project, entry) -> None:
    """Save a full-profile annotation on the first crop of `entry`."""
    prof = get_profile(project.settings.profile)
    cdir = project.crops_dir(entry)
    crops = sorted(p for p in cdir.glob("*.jpg")
                   if "_label" not in p.stem and "_debug" not in p.stem)
    cp = crops[0]
    lms = [Landmark(id=i, x=float(i * 5 + 1), y=float(i * 3 + 1)) for i in prof.ids]
    ann = WingAnnotation(image=cp.name, image_size=(80, 40),
                         profile=prof.name, landmarks=lms)
    save_annotation(ann, annotation_path(cp, cdir, prof.methodology_id))
