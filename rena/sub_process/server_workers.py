import threading
import time

import pyqtgraph as pg
from PyQt5.QtCore import (QObject, pyqtSignal)

from rena.sub_process.TCPInterface import *
from rena.utils.realtime_DSP import RealtimeButterBandpass
from rena.config import *

class DSPWorker(threading.Thread):
    # signal_data = pyqtSignal(object)
    # tick_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.processor_dict = None
        # self.init_dsp_client_server('John')

    # @pg.QtCore.pyqtSlot()
    def processing(self):
        pass

    def start_stream(self):
        pass

    def stop_stream(self):
        pass


class DSPServerWorker(DSPWorker):

    def __init__(self, RenaTCPInterface: RenaTCPInterface, processor_dict=None):
        super(DSPServerWorker, self).__init__()
        self.rena_tcp_interface = RenaTCPInterface
        self.identity = self.rena_tcp_interface.identity  # identity must be server
        self.running = True
        self.processor_dict = processor_dict
        self.processor_mutex = threading.Lock()
        # self.filter1 = RealtimeButterBandpass(fs=5000, channel_num=200)
        # self.filter2 = RealtimeButterBandpass(fs=5000, channel_num=100)
        # self.filter3 = RealtimeButterBandpass(fs=5000, channel_num=100)
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

    # @pg.QtCore.pyqtSlot()
    def run(self):
        while self.running:
            # lock cretical region
            self.processor_mutex.acquire()
            try:
                data = self.rena_tcp_interface.recv_array()
                current_time = time.time()
                # if len(data) != 0:
                # data = self.filter1.process_buffer(data)
                # data = self.filter1.process_buffer(data)
                # data = self.filter1.process_buffer(data)
                # data = self.filter1.process_buffer(data)
                # self.filter1.reset_tap()
                # # data = self.filter1.process_buffer(data)
                # # data = self.filter1.process_buffer(data)
                # # data = self.filter1.process_buffer(data)
                # # data = self.filter1.process_buffer(data)
                # # data = self.filter1.process_buffer(data)
                # data = self.filter1.process_buffer(data)

                print(time.time() - current_time)
                print(data.shape[-1])
                # time.sleep(0.5)
                # print('data received')
                send = self.rena_tcp_interface.send_array(data)

            finally:
                # release mutex
                self.processor_mutex.release()

        print('Worker Stopped')


# class RenaDSPUnit(QObject):
#     def __init__(self, rena_request_object: RenaTCPRequestObject):
#         super().__init__()
#         # create worker
#         self.worker_thread = pg.QtCore.QThread(self)
#         self.rena_tcp_interface = RenaTCPInterface(stream_name=rena_request_object.stream_name,
#                                                    port_id=rena_request_object.port_id,
#                                                    identity=rena_request_object.identity)
#         self.worker = DSPServerWorker(RenaTCPInterface=self.rena_tcp_interface)
#         self.worker.moveToThread(self.worker_thread)
#         self.worker_thread.start()


class RenaDSPServerMasterWorker(threading.Thread):

    def __init__(self, RenaTCPInterface:RenaTCPInterface):
        super().__init__()
        self._rena_tcp_interface = RenaTCPInterface
        self.dsp_server_workers = {} # DSPServerWorker use stream name as the key
        self.running = True

    def add_dsp_worker(self, request_object: RenaTCPAddDSPWorkerRequestObject):
        rena_tcp_server_interface = RenaTCPInterface(stream_name=request_object.stream_name,
                                                     port_id=request_object.port_id,
                                                     identity=request_object.identity)

        self.dsp_server_workers[rena_tcp_server_interface.stream_name] = DSPServerWorker(
            RenaTCPInterface=rena_tcp_server_interface, processor_dict=request_object.processor_dict)

        self.dsp_server_workers[rena_tcp_server_interface.stream_name].start()
        # reply request done
        self._rena_tcp_interface.send_obj(RenaTCPRequestDoneObject(request_done=True))


    def update_dsp_worker(self, request_object: RenaTCPUpdateDSPWorkerRequestObject):
        # lock the worker mutex
        self.dsp_server_workers[request_object.stream_name].processor_mutex.lock()
        try:
            # update processor_dict for that worker
            self.dsp_server_workers[request_object.stream_name].processor_dict = request_object.processor_dict
        finally:
            self.dsp_server_workers[request_object.stream_name].processor_mutex.release()
        self._rena_tcp_interface.send_obj(RenaTCPRequestDoneObject(request_done=True))

    def remove_dsp_worker(self, request_object: RenaTCPRemoveWorkerRequestObject):
        self.dsp_server_workers[request_object.stream_name].processor_mutex.lock()
        try:
            # run() exit the while loop after setting running to false TODO: deadlock condition!!!
            self.dsp_server_workers[request_object.stream_name].running = False
            self.dsp_server_workers[request_object.stream_name].join()
            self.dsp_server_workers.pop(request_object.stream_name)
        finally:
            self.dsp_server_workers[request_object.stream_name].processor_mutex.release()
        self._rena_tcp_interface.send_obj(RenaTCPRequestDoneObject(request_done=True))

    def exit_server(self, request_object: RenaTCPExitServerRequestObject):
        self.running = False
        self._rena_tcp_interface.send_obj(RenaTCPRequestDoneObject(request_done=True))






    def run(self):
        while self.running:
            request_object = self._rena_tcp_interface.recv_obj()
            # classify the request type
            # 1. add worker
            # 2. remove a worker # mutex applied
            # 3. add filter or change group format # mutex applied
            ## processor_dict format: {
            # group1: {
            # channel: [1,2,3,5,7,........]
            # filters: [filter objects]
            # }
            # }
            if request_object.request_type is rena_server_add_dsp_worker_request:
                # object type:
                self.add_dsp_worker(request_object=request_object)

            elif request_object.request_type is rena_server_update_worker_request:
                self.update_dsp_worker(request_object=request_object)

            elif request_object.request_type is rena_server_remove_worker_request:
                self.remove_dsp_worker(request_object=request_object)

            elif request_object.request_type is rena_server_exit_request:
                # end all the works first?
                self.exit_server(request_object=request_object)
                # exit server in the next loop
            else:
                print('Wrong request type, please check with client')

        print('dsp_server.running==False, Exit Server!')









