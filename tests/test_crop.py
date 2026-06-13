from __future__ import annotations

import numpy as np

from beewings.segment.crop import crop_box, crop_wing


def test_crop_box_adds_margin_and_clips():
    img = np.zeros((100, 200, 3), np.uint8)
    out = crop_box(img, (50, 40, 20, 20), margin=0.5)
    # 0.5 margin on a 20px box -> +10px each side -> 40x40 region.
    assert out.shape[0] == 40 and out.shape[1] == 40


def test_crop_box_clips_at_image_edge():
    img = np.zeros((100, 200, 3), np.uint8)
    out = crop_box(img, (0, 0, 20, 20), margin=0.5)
    # Cannot extend past the top-left corner; region is clipped, not padded.
    assert out.shape[0] == 30 and out.shape[1] == 30


def test_crop_wing_rotate_makes_landscape():
    # A tall vertical bar should come out wider-than-tall after canonical rotate.
    img = np.full((200, 200, 3), 255, np.uint8)
    img[40:160, 90:110] = 30  # vertical dark bar
    box = (90, 40, 20, 120)
    out = crop_wing(img, box, margin=0.1, rotate=True, bg=255.0)
    assert out.shape[1] >= out.shape[0]  # width >= height


def test_crop_wing_puts_dark_base_on_left():
    # Horizontal wing-like bar (mid-gray) with a dark, dense "base" on the RIGHT.
    # Canonical rotate must flip it so the dark base ends up on the LEFT.
    img = np.full((120, 300, 3), 255, np.uint8)
    img[55:65, 40:260] = 120          # faint wing blade
    img[45:75, 220:258] = 20          # dark dense base on the right
    box = (40, 45, 220, 30)
    out = crop_wing(img, box, margin=0.05, rotate=True, bg=255.0)
    gray = out[:, :, 0]
    half = out.shape[1] // 2
    dark = gray < (255 - 70)
    assert dark[:, :half].sum() > dark[:, half:].sum()  # base now on the left


from beewings.segment.debug import render_overlay


def test_render_overlay_returns_same_shape_without_mutating():
    img = np.full((120, 240, 3), 255, np.uint8)
    boxes = [(20, 20, 40, 30), (120, 60, 40, 30)]
    order = [0, 1]
    before = img.copy()
    out = render_overlay(img, boxes, order, label_box=(0, 0, 15, 100))
    assert out.shape == img.shape
    assert np.array_equal(img, before)  # input not mutated
    assert not np.array_equal(out, before)  # something was drawn
