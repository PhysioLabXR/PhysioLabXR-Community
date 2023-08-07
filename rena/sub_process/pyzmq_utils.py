import time

import numpy
import zmq


import zlib, pickle

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
        # processor = DataProcessor()
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def send_zipped_pickle(socket, obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)

def recv_zipped_pickle(socket, flags=0, protocol=-1):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)

def can_connect_to_port(port, socket_type=zmq.PUB):
    context = zmq.Context()
    socket = context.socket(socket_type)
    address = "tcp://*:%s" % port
    try:
        socket_context = socket.bind(address)
        time.sleep(0.1)
        socket.unbind(socket.getsockopt_string(zmq.LAST_ENDPOINT))
        return False
    except zmq.ZMQError as e:
        if e.errno == zmq.EADDRINUSE:
            return True
        else:
            raise e


