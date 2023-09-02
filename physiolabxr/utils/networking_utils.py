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

