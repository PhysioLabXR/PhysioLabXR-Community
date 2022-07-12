import os

import numpy as np
from TCPInterface import RENATCPServer, RENATCPInterface

# send a processing object
from rena.utils.realtime_DSP import *


# if __name__ == '__main__':
#     print("Server Created")


def dsp_processor(stream_name, port_id=None, identity='server'):
    def exit_process():
        pass

    if port_id is None:
        port_id = os.getpid()
    tcp_interface = RENATCPInterface(stream_name, port_id, identity=identity)
    tcp_server = RENATCPServer(RENATCPInterface=tcp_interface)
    print('server started')
    # start process
    while True:
        tcp_server.process_data()
