import multiprocessing
import typing

import zmq
from PyQt6 import QtCore
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QTimer


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


def start_wait_process(target: typing.Callable, args=(), finish_call_back: typing.Callable=None):
    _task_process = ProcessWithQueue(target=target, args=args)
    _task_process.start()
    wait_process_thread = QThread()
    wait_process_worker = WaitForProcessWorker(_task_process)
    wait_process_worker.moveToThread(wait_process_thread)
    if finish_call_back is not None:
        wait_process_worker.process_finished.connect(finish_call_back)
    wait_process_thread.start()
    return wait_process_worker


class WaitForResponseWorker(QObject):
    result_available = pyqtSignal()
    run_tick = pyqtSignal()

    def __init__(self, socket: zmq.Socket, poll_interval):
        super().__init__()
        self.socket = socket
        self.poll_interval = poll_interval
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.timer = None
        self.run_tick.connect(self.run)

    @QtCore.pyqtSlot()
    def run(self):
        # print("Strat polling")
        socks = dict(self.poller.poll(timeout=self.poll_interval))
        # print("Polling result: ", socks)
        if self.socket in socks and socks[self.socket] == zmq.POLLIN:
            self.timer.stop()
            current_thread = QThread.currentThread()
            current_thread.quit()
            current_thread.wait()
            self.result_available.emit()

    def exit(self):
        self.timer.stop()
        current_thread = QThread.currentThread()
        current_thread.quit()
        current_thread.wait()
        print("WaitForResponseWorker thread exited")

def start_wait_for_response(socket: zmq.Socket, poll_interval: int=100):
    wait_for_response_worker = WaitForResponseWorker(socket, poll_interval)
    wait_response_thread = QThread()
    wait_for_response_worker.moveToThread(wait_response_thread)

    poll_timer = QTimer()
    poll_timer.setInterval(poll_interval)
    poll_timer.timeout.connect(wait_for_response_worker.run_tick)
    wait_for_response_worker.timer = poll_timer

    wait_response_thread.start()
    poll_timer.start()
    return wait_for_response_worker, wait_response_thread