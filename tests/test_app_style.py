from __future__ import annotations


def test_apply_style_sets_stylesheet(qapp):
    from beewings.app.style import apply_style
    apply_style(qapp)
    assert "QPushButton" in qapp.styleSheet()
