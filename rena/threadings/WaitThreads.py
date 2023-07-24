import multiprocessing

import zmq
from PyQt6.QtCore import QThread, pyqtSignal, QObject


class ProcessWithQueue(multiprocessing.Process):
    def __init__(self, target, args=(), kwargs=()):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.result_queue = multiprocessing.Queue()

    def run(self):
        # Call the target function
        result = self._target(*self._args, **self._kwargs)

        # Put the result in the queue
        self.result_queue.put(result)

class WaitForProcessWorker(QObject):
    process_finished = pyqtSignal(object)

    def __init__(self, process):
        super().__init__()
        self.process = process

    def run(self):
        self.process.join()
        # Wait for the process to finish
        result = self.process.result_queue.get()

        # Emit the signal with the result
        self.process_finished.emit(result)
