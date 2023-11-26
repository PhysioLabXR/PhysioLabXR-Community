from PyQt6.QtCore import QThread, QObject


class QWorker(QObject):
    timer = None

    def _exit(self):
        self.timer.stop()
        current_thread = QThread.currentThread()
        current_thread.quit()
        current_thread.wait()
