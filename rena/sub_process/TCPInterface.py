from rena.config import *
from rena.sub_process.pyzmq_utils import *


class RenaTCPObject:
    def __init__(self, data, processor_dict=None, exit_process=False):
        self.data = data
        self.processor_dict = processor_dict
        self.exit_process = exit_process


class RenaTCPRequestObject:
    def __init__(self, request_type):
        self.request_type = request_type


class RenaTCPAddDSPWorkerRequestObject(RenaTCPRequestObject):
    def __init__(self, stream_name, port_id, identity, processor_dict):
        super().__init__(request_type=rena_server_add_dsp_worker_request)
        self.stream_name = stream_name
        self.port_id = port_id
        self.identity = identity
        self.processor_dict = processor_dict


class RenaTCPUpdateDSPWorkerRequestObject(RenaTCPRequestObject):
    def __init__(self, stream_name, group_format, processor_dict):
        super().__init__(request_type=rena_server_update_worker_request)
        self.stream_name = stream_name
        self.group_format = group_format
        self.processor_dict = processor_dict


class RenaTCPRemoveWorkerRequestObject(RenaTCPRequestObject):
    def __init__(self, stream_name, group_format, processor_dict):
        super().__init__(request_type=rena_server_remove_worker_request)
        self.stream_name = stream_name
        self.group_format = group_format
        self.processor_dict = processor_dict

class RenaTCPExitServerRequestObject(RenaTCPRequestObject):
    def __init__(self):
        super().__init__(request_type=rena_server_exit_request)






class RenaTCPInterface:
    def __init__(self, stream_name, port_id, identity):
        self.bind_header = 'tcp://*:'
        self.connect_header = 'tcp://localhost:'

        self.stream_name = stream_name
        self.port_id = port_id
        self.identity = identity
        # self.is_streaming = False

        if identity == 'server':
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            self.bind_socket()
            print('server connected: ', str(self.port_id))
        elif identity == 'client':
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.connect_socket()
            print('client connected: ', str(self.port_id))
        else:
            exit(1)

    def bind_socket(self):
        binder = self.bind_header + str(self.port_id)
        self.socket.bind(binder)

    def connect_socket(self):
        connection = self.connect_header + str(self.port_id)
        self.socket.connect(connection)

    def send_array(self, array, flags=0, copy=True, track=False):
        # self.socket.send(b'asdfasdf')
        sent = send_array(socket=self.socket, A=array, flags=flags, copy=copy, track=track)
        return sent

    def recv_array(self, flags=0, copy=True, track=False):
        received_array = recv_array(socket=self.socket, flags=flags, copy=copy, track=track)
        return received_array

    def send_obj(self, obj, flags=0, protocol=-1):
        """pickle an object, and zip the pickle before sending it"""

        sent = send_zipped_pickle(socket=self.socket, obj=obj, flags=0, protocol=-1)
        return sent
        # p = pickle.dumps(obj, protocol)
        # z = zlib.compress(p)
        # sent = self.socket.send(z, flags=flags)

    def recv_obj(self, flags=0, protocol=-1):
        received_obj = recv_zipped_pickle(socket=self.socket, flags=flags, protocol=-1)
        return received_obj
        # z = self.socket.recv(flags)
        # p = zlib.decompress(z)
        # return pickle.loads(p)

    def process_data(self):
        pass


class RenaTCPClient:
    def __init__(self, RENATCPInterface: RenaTCPInterface):
        self.tcp_interface = RENATCPInterface
        self.is_streaming = True

    def start_stream(self):
        self.is_streaming = True

    def stop_stream(self):
        self.is_streaming = False

    def process_data(self, data: RenaTCPObject):
        if self.is_streaming:
            try:
                send = self.tcp_interface.send_obj(data)
                print('client data sent')
                data = self.tcp_interface.recv_obj()
                print('client data received')
            except:  # unknown type exception
                print("Client Crashed")

            return data

    def process_array(self, data):
        if self.is_streaming:
            try:
                send_message = self.tcp_interface.send_array(data)
                data = self.tcp_interface.recv_array()
                # data = DSP.process_data(data)

            except:  # unknown type exception
                print("Client Crashed")

            return data

# class RenaTCPServer:
#     def __init__(self, RENATCPInterface: RenaTCPInterface):
#         self.tcp_interface = RENATCPInterface
#         self.is_streaming = True
#         self.dsp_processors = None
#
#     def start_stream(self):
#         self.is_streaming = True
#
#     def stop_stream(self):
#         self.is_streaming = False
#
#     def process_data(self):
#         if self.is_streaming:
#             # try:
#             print('server receiving data')
#             data = self.tcp_interface.recv_obj()
#
#             if data.exit_process:
#                 # print("Exit Server")
#                 return 0
#                 # sys.exit(0)
#
#             print('server data received')
#             # data = DSP.process_data(data)
#             #
#             #
#             #
#             #
#             #
#             send = self.tcp_interface.send_obj(data)
#             print('server data sent')
#         # except:  # unknown type exception
#         #     print("DSP Server Crashed")
#
#     def process_array(self, data):
#         if self.is_streaming:
#             try:
#                 data = self.tcp_interface.recv_array()
#                 # data = DSP.process_data(data)
#                 #
#                 #
#                 #
#                 #
#                 #
#                 send = self.tcp_interface.send_array(data)
#             except:  # unknown type exception
#                 print("DSP Server Crashed")
#         # return data
#
#     def update_renaserverinterface_processor(self, RENATCPObject: RenaTCPObject):
#         self.dsp_processors = RENATCPObject.processor_dict
