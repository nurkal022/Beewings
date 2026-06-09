"""Entry point: `python -m beewings.annotator` or installed `beewings`."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
