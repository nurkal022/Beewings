"""Pure geometry for editing rectangular boxes (no Qt)."""
from __future__ import annotations

from typing import List, Optional, Tuple

Box = Tuple[int, int, int, int]
Handle = Optional[str]  # None|'move'|'nw'|'ne'|'sw'|'se'|'n'|'s'|'e'|'w'


def normalize_box(box: Box) -> Box:
    """Return an equivalent box with non-negative width and height."""
    x, y, w, h = box
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    return (x, y, w, h)


def _corners(box: Box):
    x, y, w, h = box
    return {"nw": (x, y), "ne": (x + w, y), "sw": (x, y + h), "se": (x + w, y + h)}


def _sides(box: Box):
    x, y, w, h = box
    return {"n": (x + w / 2, y), "s": (x + w / 2, y + h),
            "w": (x, y + h / 2), "e": (x + w, y + h / 2)}


def hit_test(boxes: List[Box], px: int, py: int, handle_px: int = 8):
    """Return (index, handle) for the topmost box hit at (px, py), else (None, None).

    A corner within `handle_px` yields that corner handle; inside the box yields
    'move'. Iterates last-to-first so the most recently drawn box wins.
    """
    for idx in range(len(boxes) - 1, -1, -1):
        box = normalize_box(boxes[idx])
        for name, (cx, cy) in _corners(box).items():
            if abs(px - cx) <= handle_px and abs(py - cy) <= handle_px:
                return idx, name
        for name, (cx, cy) in _sides(box).items():
            if abs(px - cx) <= handle_px and abs(py - cy) <= handle_px:
                return idx, name
        x, y, w, h = box
        if x <= px <= x + w and y <= py <= y + h:
            return idx, "move"
    return None, None


def resize_box(box: Box, handle: Handle, dx: int, dy: int) -> Box:
    """Apply a drag delta to a box given the grabbed handle. Returns a new box."""
    x, y, w, h = box
    if handle == "move":
        return (x + dx, y + dy, w, h)
    if handle == "nw":
        return normalize_box((x + dx, y + dy, w - dx, h - dy))
    if handle == "ne":
        return normalize_box((x, y + dy, w + dx, h - dy))
    if handle == "sw":
        return normalize_box((x + dx, y, w - dx, h + dy))
    if handle == "se":
        return normalize_box((x, y, w + dx, h + dy))
    if handle == "n":
        return normalize_box((x, y + dy, w, h - dy))
    if handle == "s":
        return normalize_box((x, y, w, h + dy))
    if handle == "w":
        return normalize_box((x + dx, y, w - dx, h))
    if handle == "e":
        return normalize_box((x, y, w + dx, h))
    return box
