import time
import zmq
import numpy as np
from pyzmq_utils import *


class TCPInterface:
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
        elif identity == 'client':
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.connect_socket()
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

    def process_data(self):
        pass

class TCPProcessor:
    def __init__(self, TCPInterface: TCPInterface):
        self.tcp_interface = TCPInterface
        self.is_streaming = False

    def start_stream(self):
        self.is_streaming = True

    def stop_stream(self):
        self.is_streaming = False

    def send_data(self, data):
        if self.is_streaming:
            self.tcp_interface.send_array(data)

    # def receive_data(self, data):
    #     if self.is_streaming:
    #




# def receive_data(self):
#     pass

# def process_data(self):
#     pass

# def start_process(self):
#     self.is_streaming = True
#
# def stop_process(self):
#     self.is_streaming = False

# class TCPInterfacesServer(TCPInterface):
#
#     # create signal data to client
#
#     def __init__(self, stream_name, port_id, identity='server'):  # identity: server or client
#         super(TCPInterfacesServer, self).__init__(stream_name=stream_name, port_id=port_id, identity='server')


# class TCPInterfacesClient(TCPInterface):
#
#     # create signal data to client
#
#     def __init__(self, stream_name, port_id, identity='client'):  # identity: server or client
#         super().__init__()
#
#
#
#     def connect_socket(self):
#         connection = self.connect_header + str(self.port_id)
#         self.socket.connect(connection)
#
#     def send_data(self):
#         pass
#
#     def receive_data(self):
#         pass
#
#     def process_data(self):
#         pass
#
#     def start_process(self):
#         self.is_streaming = True
#
#     def stop_process(self):
#         self.is_streaming = False
#
# # def connect_socket(self):
# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind("tcp://*:5555")
#
# while True:
#     #  Wait for next request from client
#     message = socket.recv()
#     print("Received request: %s" % message)
#
#     #  Do some 'work'
#     time.sleep(1)
#
#     #  Send reply back to client
#     socket.send(b"World")
