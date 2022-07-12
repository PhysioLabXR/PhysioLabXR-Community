from TCPInterface import TCPInterface
import numpy as np

array = np.random.rand(2,2)

stream_name = 'John'
port_id = 1234
identity = 'client'
client = TCPInterface(stream_name, port_id, identity)
send = client.send_array(array)
print(send)