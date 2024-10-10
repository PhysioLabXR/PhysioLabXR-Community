import os
import time

import numpy as np
from physiolabxr.sub_process.TCPInterface import RenaTCPServer, RenaTCPInterface

# send a processing object
from physiolabxr.utils.realtime_DSP import *


# if __name__ == '__main__':
#     print("Server Created")


def dsp_processor(stream_name, port_id=None, identity='server'):
    def exit_process():
        pass

    if port_id is None:
        port_id = os.getpid()
        print('Server PID: ', str(port_id))
    tcp_interface = RenaTCPInterface(stream_name, port_id, identity=identity)
    tcp_server = RenaTCPServer(RENATCPInterface=tcp_interface)
    print('Server Started')

    exit_return = None
    # start process
    while exit_return is None:
        exit_return = tcp_server.process_sample()
        # if exit_return:
        #     print('Exit Server')

    print('Exit Server')

