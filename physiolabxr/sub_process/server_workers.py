import threading
import time

from physiolabxr.sub_process.TCPInterface import RenaTCPInterface
from physiolabxr.utils.realtime_DSP import RealtimeButterBandpass
from physiolabxr.configs.config import *

class DSPWorker(threading.Thread):
    # signal_data = pyqtSignal(object)
    # tick_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.processor_dict = None
        # self.init_dsp_client_server('John')

    # @QtCore.pyqtSlot()
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
        self.filter1 = RealtimeButterBandpass(fs=5000, channel_num=200)

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

    # @QtCore.pyqtSlot()
    def run(self):
        while self.is_streaming:
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

        print('Server Stopped')


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




    def run(self):
        while self.running:
            request_object = self._rena_tcp_interface.recv_obj()

            # classify the request type
            # 1. add worker
            # 2. remove a worker # mutex applied
            # 3. add filter or change group format # mutex applied
            if request_object.request_type is rena_server_add_dsp_worker_request:
                pass
            elif request_object.request_type is rena_server_update_worker_request:
                pass
            elif request_object.request_type is rena_server_remove_worker_request:
                pass
            elif request_object.request_type is rena_server_exit_request:
                pass
            else:
                print('Wrong request type, please check with client')









