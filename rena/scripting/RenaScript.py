import os
import sys
import threading
from abc import ABC, abstractmethod

import numpy as np

from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.utils.networking_utils import recv_string_router_dealer, send_string_router_dealer, send_router_dealer


class RenaScript(ABC, threading.Thread):
    """
    An abstract class for implementing scripting models.
    """

    def __init__(self, inputs, input_shapes, outputs, output_num_channels, params, port, run_frequency, time_window):
        """

        :param inputs:
        :param outputs:
        :param params:
        :param port: the port to which we bind the
        """
        super().__init__()
        print('RenaScript Thread started on process {0}'.format(os.getpid()))
        print('Waiting for routing ID from main app')
        self.command_info_interface = RenaTCPInterface(stream_name='RENA_REPLAY',
                                                       port_id=port,
                                                       identity='server',
                                                       pattern='router-dealer')
        _, self.routing_id = recv_string_router_dealer(self.command_info_interface, True)
        send_string_router_dealer(str(os.getpid()), self.routing_id, self.command_info_interface)
        sys.stdout = RedirectStdout(socket_interface=self.command_info_interface, routing_id=self.routing_id)
        self.inputs = dict()
        self.outputs = dict()
        # self.command_info_interface = command_info_interface
        # create data buffers

        print('Script init successfully')

    @abstractmethod
    def init(self):
        """
        Start will be called once when the run button is hit.
        """
        pass

    @abstractmethod
    def loop(self):
        """
        Loop is called
        """
        pass

    def run(self):
        print('Base start function is called')
        try:
            self.init()
        except Exception as e:
            print('Exception in init(): ', e)
        # start the loop here, accept interrupt command
        print('Entering loop')
        while True:
            try:
                self.loop()
            except Exception as e:
                print('Exception in loop: ', e)



    def __del__(self):
        sys.stdout = sys.__stdout__  # return control to regular stdout


class RedirectStdout(object):
    def __init__(self, socket_interface, routing_id):
        self.terminal = sys.stdout
        self.routing_id = routing_id
        self.socket_interface = socket_interface

    def write(self, message):
        self.terminal.write(message)
        send_string_router_dealer(message, self.routing_id, self.socket_interface)
