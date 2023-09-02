import pyqtgraph as pg
from PyQt6.QtCore import (QObject, pyqtSignal)

from physiolabxr.sub_process.TCPInterface import RenaTCPInterface, RenaTCPAddDSPWorkerRequestObject


class DSPWorker(QObject):
    signal_data = pyqtSignal(object)
    tick_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.processor_dict = None
        # self.init_dsp_client_server('John')

    @QtCore.pyqtSlot()
    def processing(self):
        pass

    def start_stream(self):
        pass

    def stop_stream(self):
        pass


class DSPServerWorker(DSPWorker):

    def __init__(self, RenaTCPInterface: RenaTCPInterface):
        super(DSPServerWorker, self).__init__()
        self.rena_tcp_interface = RenaTCPInterface
        self.identity = self.rena_tcp_interface.identity  # identity must be server
        self.is_streaming = True
        # self.processor_run()

    def run_processor_dict(self, data):
        self.processor_dict = self.processor_dict
        return data

    def process_rena_tcp_object(self, rena_tcp_object):
        if rena_tcp_object.processor_dict is not None:
            self.processor_dict = rena_tcp_object.processor_dict

    def processing(self):
        # receive data
        rena_tcp_object = self.rena_tcp_interface.recv_obj()
        # check if processor dict should be update

        self.process_rena_tcp_object(rena_tcp_object)
        # run processor dict
        data = self.run_processor_dict(rena_tcp_object.data)

        # processing.........
        self.rena_tcp_interface.send_array(array=data)

    @QtCore.pyqtSlot()
    def run(self):
        while self.is_streaming:
            data = self.rena_tcp_interface.recv_obj()
            print('John')
            # for i in range(0,100000):
            #     i = i
            #
            # print('John')

            # print('John')
            # self.processing()

        print('Server Stopped')


class RenaDSPUnit(QObject):
    def __init__(self, rena_request_object: RenaTCPAddDSPWorkerRequestObject):
        super().__init__()
        # create worker
        self.worker_thread = pg.QtCore.QThread(self)
        self.rena_tcp_interface = RenaTCPInterface(stream_name=rena_request_object.stream_name,
                                                   port_id=rena_request_object.port_id,
                                                   identity=rena_request_object.identity)
        self.worker = DSPServerWorker(RenaTCPInterface=self.rena_tcp_interface)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()
