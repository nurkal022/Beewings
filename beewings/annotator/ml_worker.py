"""Background thread for ML landmark detection (keeps the UI responsive)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class MLDetectWorker(QThread):
    """Load the checkpoint (if needed), validate its schema, and predict — all
    off the GUI thread. Emits done(pred, conf, model) or failed(message)."""

    done = pyqtSignal(object, object, object)   # pred dict, conf dict, model
    failed = pyqtSignal(str)

    def __init__(self, checkpoint: Path, bgr: np.ndarray, tta: bool,
                 expected_n: int, allow_alpatov8: bool, model=None, parent=None):
        super().__init__(parent)
        self.checkpoint = Path(checkpoint)
        self.bgr = bgr
        self.tta = tta
        self.expected_n = expected_n
        self.allow_alpatov8 = allow_alpatov8
        self.model = model

    def run(self) -> None:
        try:
            model = self.model
            if model is None:
                # Schema check before committing to a full load.
                import torch
                peek = torch.load(self.checkpoint, map_location="cpu",
                                  weights_only=False)
                ckpt_n = peek["config"]["n_points"]
                if ckpt_n != self.expected_n and not (
                    self.allow_alpatov8 and ckpt_n == 12 and self.expected_n == 8
                ):
                    self.failed.emit(
                        f"Несовпадение схемы: checkpoint {self.checkpoint.name} "
                        f"обучен на {ckpt_n} точек, а профиль ожидает "
                        f"{self.expected_n}. Это разные методики.")
                    return
                from ..ml.inference import load as load_ml
                # CPU on purpose: this runs on a background QThread. Using the
                # GPU (MPS/Metal) here would contend with the main thread's
                # Metal rendering of the canvas and crash natively on macOS.
                # The model is small and inference is backgrounded, so CPU is
                # fast enough and keeps the UI responsive.
                model = load_ml(self.checkpoint, device="cpu")
            from ..ml.inference import predict as ml_predict
            pred, conf = ml_predict(model, self.bgr, tta=self.tta,
                                    return_confidence=True)
            self.done.emit(pred, conf, model)
        except Exception as ex:  # surface to the UI thread
            self.failed.emit(f"{type(ex).__name__}: {ex}")
