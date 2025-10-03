# threads.py -----------------------------------------------------------------
import sys
import traceback

from PyQt6.QtCore import (QObject, QThread, pyqtSignal, pyqtSlot,
                          Qt, QTimer)
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel,
                             QProgressBar)
from PyQt6.QtGui import QMovie


class _Worker(QObject):
    finished = pyqtSignal(object)             # return value (or None)
    failed   = pyqtSignal(str)       # emits traceback string

    def __init__(self, fn, args, kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    @pyqtSlot()
    def run(self):
        print("Worker.run()")
        try:
            res = self._fn(*self._args, **self._kwargs)
            self.finished.emit(res)
        except Exception:             # catch ANY error
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            self.failed.emit(tb)


class BusyDialog(QDialog):
    """Non-modal 'busy…' window with optional animated GIF."""
    def __init__(self, working_text, gif_path=None, parent=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("Please wait")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        lay = QVBoxLayout(self)

        # -- Animated GIF (optional) -------------------------------------
        self.movie = None
        if gif_path:
            self.movie = QMovie(gif_path)
            gif_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
            gif_label.setMovie(self.movie)
            self.movie.start()
            lay.addWidget(gif_label)

        # -- Status text -------------------------------------------------
        self.label = QLabel(working_text,
                            alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.label)

        # -- Indeterminate progress bar ----------------------------------
        bar = QProgressBar()
        bar.setRange(0, 0)                    # pulsing
        lay.addWidget(bar, stretch=1)

        self.resize(220, 140)

    def show_done(self, done_text):
        self.label.setText(done_text)
        if self.movie:
            self.movie.stop()


def run_in_thread(fn,
                  args=(),
                  kwargs=None,
                  working_text="Working…",
                  done_text="Done!",
                  loading_gif_path=None,
                  parent=None):
    """
    Execute *fn* in a background QThread while showing a BusyDialog.
    Returns the QThread (auto-cleaned on finish).
    """
    if kwargs is None:
        kwargs = {}

    dlg = BusyDialog(working_text, loading_gif_path, parent)
    dlg.show()

    thread = QThread(parent)
    worker = _Worker(fn, args, kwargs)
    worker.moveToThread(thread)
    thread.worker = worker #  strong reference to prevent garbage collection on the worker

    # ---------- wiring ----------
    def _on_finished(_):
        dlg.show_done(done_text)
        QTimer.singleShot(1000, dlg.close)
        thread.quit()

    worker.failed.connect(lambda tb: dlg.show_done("Failed — see console"))
    worker.failed.connect(worker.deleteLater)
    worker.failed.connect(thread.quit)

    thread.started.connect(worker.run)
    worker.finished.connect(_on_finished)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    print("run_in_thread: thread started, returning it.")
    return thread, dlg, worker
