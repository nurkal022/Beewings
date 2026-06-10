"""Application-wide Qt stylesheet for a clean, consistent look."""
from __future__ import annotations

QSS = """
QWidget { font-size: 13px; }
QMainWindow, QWidget#Home { background: #f4f5f7; }
QPushButton {
    background: #2d6cdf; color: white; border: none;
    border-radius: 6px; padding: 7px 14px;
}
QPushButton:hover { background: #2559c0; }
QPushButton:disabled { background: #b8c2d0; color: #eef; }
QPushButton#ghost { background: transparent; color: #2d6cdf; }
QPushButton#ghost:hover { background: #e7eefc; }
QTabBar::tab {
    padding: 8px 18px; margin-right: 2px;
    background: #e3e6ea; border-top-left-radius: 6px; border-top-right-radius: 6px;
}
QTabBar::tab:selected { background: #ffffff; font-weight: bold; color: #2d6cdf; }
QListWidget, QTreeWidget { background: white; border: 1px solid #dce0e6; border-radius: 6px; }
QFrame#card {
    background: white; border: 1px solid #dce0e6; border-radius: 10px;
}
QFrame#card:hover { border: 1px solid #2d6cdf; }
QProgressBar { border: 1px solid #dce0e6; border-radius: 6px; text-align: center; height: 16px; }
QProgressBar::chunk { background: #2d6cdf; border-radius: 6px; }
"""


def apply_style(app) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
