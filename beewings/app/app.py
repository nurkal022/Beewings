"""BeeWings unified entry point: Home <-> Project."""
from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox

from ..pipeline.project import CropProject
from .home import HomeScreen
from .project_window import ProjectWindow
from .recents import add_recent, default_store, load_recents, save_recents
from .style import apply_style

ERROR_LOG = Path.home() / ".beewings" / "errors.log"


def _install_excepthook() -> None:
    """Keep the app alive on an unhandled exception (e.g. in a Qt slot) instead
    of crashing: log it and show a dialog. In PyQt6, overriding sys.excepthook
    suppresses the default abort, so a single bug no longer kills the program."""
    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        try:
            ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
            with ERROR_LOG.open("a", encoding="utf-8") as f:
                f.write(f"\n--- {datetime.now().isoformat(timespec='seconds')} ---\n{text}")
        except OSError:
            pass
        try:
            QMessageBox.critical(
                None, "BeeWings — ошибка",
                "Произошла ошибка, но приложение продолжит работу.\n\n"
                f"{exc_type.__name__}: {exc}\n\nЖурнал: {ERROR_LOG}")
        except Exception:
            pass
    sys.excepthook = hook


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
        try:
            proj = CropProject.open_folder(Path(folder))
        except Exception as exc:
            QMessageBox.critical(
                self.home_win, "Не удалось открыть проект",
                f"Папку не удалось открыть как проект:\n{folder}\n\n"
                f"{type(exc).__name__}: {exc}")
            return
        items = add_recent(load_recents(self.store), str(proj.root),
                           proj.progress(), datetime.now().isoformat(timespec="seconds"))
        save_recents(items, self.store)
        old = self.project_win
        self.project_win = ProjectWindow(proj, on_home=self._back_home)
        self.home_win.hide()
        self.project_win.show()
        if old is not None:
            # Tear the previous project window down cleanly: stop its worker
            # threads first so their (now stale) signals can't reach deleted
            # widgets, then schedule deletion.
            old.dispose()

    def _back_home(self) -> None:
        if self.project_win is not None:
            proj = self.project_win.project
            items = add_recent(load_recents(self.store), str(proj.root),
                               proj.progress(), datetime.now().isoformat(timespec="seconds"))
            save_recents(items, self.store)
        self._show_home()


def main() -> int:
    _install_excepthook()
    app = QApplication.instance() or QApplication(sys.argv)
    apply_style(app)
    # Keep a strong reference: a discarded controller can be garbage-collected,
    # taking its windows with it (the app window "disappears").
    controller = AppController()
    app._beewings_controller = controller  # extra anchor for the app's lifetime
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
