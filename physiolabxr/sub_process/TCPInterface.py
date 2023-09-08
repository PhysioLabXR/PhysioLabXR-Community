from physiolabxr.configs.config import *

from physiolabxr.sub_process.pyzmq_utils import *


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

    def __init__(self, stream_name, port_id, identity, pattern='request-reply', add_poller=False, disable_linger=False):
        self.bind_header = "tcp://*:%s"
        self.connect_header = 'tcp://localhost:'

        self.stream_name = stream_name
        self.port_id = port_id
        self.identity = identity

        self.pattern = pattern
        if pattern == 'request-reply' and identity == 'server': socket_type = zmq.REP
        elif pattern == 'request-reply' and identity == 'client': socket_type = zmq.REQ
        elif pattern == 'pipeline' and identity == 'client': socket_type = zmq.PULL
        elif pattern == 'pipeline' and identity == 'server': socket_type = zmq.PUSH
        elif pattern == 'router-dealer' and identity == 'client': socket_type = zmq.DEALER
        elif pattern == 'router-dealer' and identity == 'server': socket_type = zmq.ROUTER
        else: raise AttributeError('Unsupported interface pattern: {0} or identity {1}'.format(pattern, identity))

        self.context = zmq.Context()
        self.socket = self.context.socket(socket_type)
        if disable_linger:
            self.socket.setsockopt(zmq.LINGER, 0)  # avoid hanging on context termination

        self.address = None
        if identity == 'server': self.bind_socket()
        elif identity == 'client': self.connect_socket()
        else: raise AttributeError('Unsupported interface identity: {0}'.format(identity))

        # create poller object, so we can poll msg with a timeout
        if add_poller:
            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)
        else:
            self.poller = None

    def bind_socket(self):
        self.address = binder = self.bind_header % self.port_id
        self.socket.bind(binder)

    def connect_socket(self):
        self.address = connection = self.connect_header + str(self.port_id)
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

    def send_string(self, data: str, *args, **kwargs):
        self.socket.send(data.encode('utf-8'), *args, **kwargs)

    def send(self, data: bytes, *args, **kwargs):
        self.socket.send(data, *args, **kwargs)

    def recv_string(self, *args, **kwargs):
        return self.socket.recv(*args, **kwargs).decode('utf-8')

    def __del__(self):
        self.socket.close()
        self.context.term()
        print('Socket closed and context terminated')


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


def test_port_range(start_port, end_port):
    for port in range(start_port, end_port + 1):
        try:
            client = RenaTCPInterface(stream_name='test stream', port_id=port, identity='client', pattern='router-dealer')
            print(f"test_port_range: client connected to {client.address}")
            server = RenaTCPInterface(stream_name='test stream', port_id=port, identity='server', pattern='router-dealer')
            print(f"test_port_range: client connected to {server.address}")
            del client, server
            return port
        except zmq.error.ZMQError:
            pass
    return None