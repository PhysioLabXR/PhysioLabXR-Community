import multiprocessing
import socket

import numpy as np
import zmq

from physiolabxr.configs.shared import DATA_BUFFER_PREFIX
from physiolabxr.utils.RNStream import max_dtype_len
from physiolabxr.utils.buffers import flatten


def send_string_router(message, routing_id, socket_interface):
    socket_interface.socket.send_multipart(
        [routing_id, message.encode('utf-8')])


def send_router(data, routing_id, socket_interface):
    socket_interface.socket.send_multipart(
        [routing_id, data])


def recv_string_router(socket_interface, is_block):
    if is_block:
        routing_id, message = socket_interface.socket.recv_multipart(flags=0)
        return message.decode('utf-8'), routing_id
    else:
        try:
            routing_id, message = socket_interface.socket.recv_multipart(
                flags=zmq.NOBLOCK)
            return message.decode('utf-8'), routing_id
        except zmq.error.Again:
            return None  # no message has arrived at the socket yet


def recv_string(socket_interface, is_block):
    if is_block:
        message = socket_interface.socket.recv_multipart(flags=0)[0]
        return message.decode('utf-8')
    else:
        try:
            message = socket_interface.socket.recv_multipart(
                flags=zmq.NOBLOCK)[0]
            return message.decode('utf-8')
        except zmq.error.Again:
            return None  # no message has arrived at the socket yet


def send_data_dict(data_dict: dict, socket_interface):
    keys = [k.encode('utf-8') for k in data_dict.keys()]
    data_timestamp_list = []
    for data, timestamps in data_dict.values():
        data_timestamp_list.append((data, timestamps))
    # data_and_timestamps = [item for sublist in list(data_buffer.values()) for item in sublist]
    send_packet = [DATA_BUFFER_PREFIX] + flatten(
        [(k, get_dtype_bypes(d.dtype), np.array(d.shape[0]), np.array(d.shape[1]), d.tobytes(), t.tobytes()) for k, (d, t) in zip(keys, data_timestamp_list)])
    socket_interface.socket.send_multipart(send_packet)

def get_dtype_bypes(dtype):
    dtype_str = str(dtype)
    return bytes(dtype_str + "".join(" " for x in range(max_dtype_len - len(dtype_str))), 'utf-8')


def recv_data_dict(socket_interface):
    data_dict = socket_interface.socket.recv_multipart()[1:]  # remove the routing ID
    assert data_dict[0] == DATA_BUFFER_PREFIX
    if len(data_dict) == 1:
        return {}
    else:
        data_dict = data_dict[1:]  # remove the prefix
        rtn = dict()
        for i in range(0, len(data_dict), 6):
            key = data_dict[i].decode('utf-8')
            dtype = np.dtype(data_dict[i + 1].decode('utf-8').strip(' '))
            shape = np.frombuffer(data_dict[i + 2], dtype=int)[0], np.frombuffer(data_dict[i + 3], dtype=int)[0]
            data = np.frombuffer(data_dict[i + 4], dtype=dtype).reshape(shape)
            timestamps = np.frombuffer(data_dict[i + 5], dtype=np.float64)
            rtn[key] = (data, timestamps)
        return rtn


def find_available_ports(start_port, num_ports=100):
    """
    Find a range of available ports.

    Parameters:
    - start_port: The starting port number to check.
    - num_ports: The number of consecutive available ports to find.

    Returns:
    - A list of available port numbers.
    """
    available_ports = []
    port = start_port

    while len(available_ports) < num_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Check if the port is available
            result = sock.connect_ex(('localhost', port))
            if result != 0:  # Port is available
                available_ports.append(port)
            else:
                # Reset the list if the current port is unavailable and we were accumulating
                available_ports.clear()

        port += 1

    return available_ports

def find_available_port_from_list(ports_list):
    """
    Find an available port from a list of ports.

    Parameters:
    - ports_list: A list of candidate port numbers.

    Returns:
    - A port number if available; otherwise, None.
    """
    for port in ports_list:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', port))
            if result != 0:  # Port is available
                return port
    return None

class PortFinderProcess(multiprocessing.Process):
    def __init__(self, queue, start_port, num_ports=100):
        super().__init__()
        self.queue = queue
        self.start_port = start_port
        self.num_ports = num_ports

    def run(self):
        available_ports = []
        port = self.start_port
        while len(available_ports) < self.num_ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                result = sock.connect_ex(('localhost', port))
                if result != 0:  # Port is available
                    available_ports.append(port)
                    self.queue.put(available_ports.copy())  # Send the updated list to the queue
            port += 1