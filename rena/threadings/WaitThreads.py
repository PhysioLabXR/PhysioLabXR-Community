import multiprocessing

from PyQt6.QtCore import QThread, pyqtSignal


class ProcessWithQueue(multiprocessing.Process):
    def __init__(self, target, args=(), kwargs=()):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.result_queue = multiprocessing.Queue()

    def run(self):
        # Call the target function and put the result in the queue
        result = self._target(*self._args, **self._kwargs)
        self.result_queue.put(result)

class WaitProcessThread(QThread):
    process_finished = pyqtSignal(object)

    def __init__(self, process: ProcessWithQueue):
        super().__init__()
        self.process = process

    def run(self):
        # Wait for the process to finish
        self.process.join()
        self.process_finished.emit(self.process.result_queue.get())