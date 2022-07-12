import numpy as np
from TCPInterface import RENATCPServer, RENATCPInterface

# send a processing object
from rena.utils.realtime_DSP import *


# if __name__ == '__main__':
#     print("Server Created")



def dsp_processor(stream_name, port_id, identity):

    def exit_process():
        pass

    # class RENATCPInterface:
    #     def __init__(self, stream_name, port_id, identity):
    tcp_interface = RENATCPInterface(stream_name, port_id, identity='server')
    tcp_server = RENATCPServer(RENATCPInterface=tcp_interface)


    # start process
    while True:
        tcp_server.process_data()






