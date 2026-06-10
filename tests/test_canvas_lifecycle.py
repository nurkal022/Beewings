from __future__ import annotations


def test_scene_is_parentless(qapp):
    """Scene must be Python-owned (no Qt parent) so the view's parent-chain
    teardown cannot delete it out from under a surviving Python reference."""
    from beewings.annotator.canvas import WingCanvas
    c = WingCanvas()
    assert c._scene.parent() is None


def test_refresh_landmarks_survives_deleted_scene(qapp):
    """Even if the scene's C++ object is gone, scene-touching methods must
    no-op instead of raising (which PyQt6 would turn into a SIGABRT)."""
    from PyQt6 import sip
    from beewings.annotator.canvas import WingCanvas
    from beewings.core.profiles import get_profile
    from beewings.core.schema import Landmark, WingAnnotation
    c = WingCanvas()
    prof = get_profile("Алпатов 12 точек")
    c._profile = prof
    c._ann = WingAnnotation(image="w.jpg", image_size=(60, 30), profile=prof.name,
                            landmarks=[Landmark(id=1, x=5.0, y=5.0)])
    sip.delete(c._scene)                 # simulate Qt deleting the scene
    c.refresh_landmarks()                # must not raise
    c.set_template([])                   # must not raise
    c.clear_template()                   # must not raise
    c._refresh_pixmap()                  # must not raise
