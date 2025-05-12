# threads.py -----------------------------------------------------------------
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt6.QtWidgets import QProgressDialog, QLabel, QVBoxLayout
from PyQt6.QtGui import QMovie

class _Worker(QObject):
    finished = pyqtSignal(object)            # return value (or None)

    def __init__(self, fn, args, kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    @pyqtSlot()
    def run(self):
        res = None
        try:
            res = self._fn(*self._args, **self._kwargs)
        finally:
            self.finished.emit(res)          # always emit

def run_in_thread(fn,
                  args=(),
                  kwargs=None,
                  working_text="Working…",
                  done_text="Done!",
                  gif_path=None,             # <- NEW
                  parent=None):
    """
    Execute *fn* in a background QThread and show a non-modal dialog
    with an animated GIF while it runs.

    Parameters
    ----------
    fn : Callable
    args : tuple
    kwargs : dict | None
    working_text : str   text displayed while running
    done_text    : str   text displayed for ~1 s after finishing
    gif_path     : str | None  path to an animated GIF
    parent       : QWidget | None  owner of the dialog

    Returns
    -------
    QThread  (auto-deleted after finish)
    """
    if kwargs is None:
        kwargs = {}

    # ---------- UI ----------
    dlg = QProgressDialog(parent)
    dlg.setWindowTitle("Please wait")
    dlg.setLabelText(working_text)
    dlg.setMinimum(0)
    dlg.setMaximum(0)                        # indeterminate bar
    dlg.setWindowModality(Qt.WindowModality.NonModal)  # <- non-modal
    dlg.setCancelButton(None)

    # Optional: place GIF above the text
    if gif_path:
        gif_label = QLabel()
        gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        movie = QMovie(gif_path)
        gif_label.setMovie(movie)
        movie.start()

        # Insert the GIF at the top of the existing layout
        lay: QVBoxLayout = dlg.layout()
        lay.insertWidget(0, gif_label)

    dlg.show()

    # ---------- Worker + Thread ----------
    thread = QThread(parent)
    worker = _Worker(fn, args, kwargs)
    worker.moveToThread(thread)

    def _on_finished(_result):
        dlg.setLabelText(done_text)
        if gif_path:
            movie.stop()
        QTimer.singleShot(1000, dlg.close)
        thread.quit()

    thread.started.connect(worker.run)
    worker.finished.connect(_on_finished)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread
