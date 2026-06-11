"""BeeWings unified entry point: Home <-> Project."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow

from ..pipeline.project import CropProject
from .home import HomeScreen
from .project_window import ProjectWindow
from .recents import add_recent, default_store, load_recents, save_recents
from .style import apply_style


class AppController:
    def __init__(self):
        self.store = default_store()
        self.home_win = QMainWindow()
        self.home_win.setWindowTitle("BeeWings")
        self.home_win.resize(900, 640)
        self.project_win = None
        self._show_home()

    def _show_home(self) -> None:
        recents = load_recents(self.store)
        home = HomeScreen(recents, on_open=self._open_project)
        self.home_win.setCentralWidget(home)
        self.home_win.show()
        self.home_win.raise_()

    def _open_project(self, folder: str) -> None:
        proj = CropProject.open_folder(Path(folder))
        items = add_recent(load_recents(self.store), str(proj.root),
                           proj.progress(), datetime.now().isoformat(timespec="seconds"))
        save_recents(items, self.store)
        self.home_win.hide()
        old = self.project_win
        self.project_win = ProjectWindow(proj, on_home=self._back_home)
        self.project_win.show()
        if old is not None:
            # Destroy the previous project window so its (now stale) pages/widgets
            # can't be reached by lingering signal connections.
            old.close()
            old.deleteLater()

    def _back_home(self) -> None:
        if self.project_win is not None:
            proj = self.project_win.project
            items = add_recent(load_recents(self.store), str(proj.root),
                               proj.progress(), datetime.now().isoformat(timespec="seconds"))
            save_recents(items, self.store)
        self._show_home()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_style(app)
    AppController()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
