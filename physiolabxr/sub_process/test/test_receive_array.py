# from TCPInterface import TCPInterface
import numpy as np

from physiolabxr.sub_process.TCPInterface import RenaTCPInterface

rand_array = np.random.rand(10,10)

stream_name = 'John'
port_id = 1234
identity = 'server'
server = RenaTCPInterface(stream_name, port_id, identity)

while True:
    print("Receiving")
    array=server.recv_array()
    print(array)
    server.send_array(rand_array)