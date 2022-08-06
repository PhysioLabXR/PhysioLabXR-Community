import sys
import threading
from abc import ABC, abstractmethod

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
        # create
        self.inputs = dict()
        self.outputs = dict()
        # self.command_info_interface = command_info_interface
        # create data buffers

        # command_info_interface = RenaTCPInterface(stream_name='RENA_REPLAY',
        #                                           port_id=config.replay_port,
        #                                           identity='server',
        #                                           pattern='router-dealer')


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
        self.init()
        sys.stdout.flush()
        # start the loop here, accept interrupt command
        print('Entering loop')
        while True:
            self.loop()
            sys.stdout.flush()