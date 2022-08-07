import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from rena.config import script_fps_counter_buffer_size
from rena.shared import SCRIPT_STDOUT_MSG_PREFIX, SCRIPT_STOP_REQUEST, SCRIPT_STOP_SUCCESS, SCRIPT_INFO_REQUEST
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.utils.general import get_fps
from rena.utils.networking_utils import recv_string_router, send_string_router, send_router


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
        print('RenaScript: RenaScript Thread started on process {0}'.format(os.getpid()))
        self.stdout_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_STDOUT',
                                                        port_id=port,
                                                        identity='server',
                                                        pattern='router-dealer')
        self.command_info_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_COMMAND',
                                                              port_id=port + 1,
                                                              identity='server',
                                                              pattern='router-dealer')
        print('RenaScript: Waiting for stdout routing ID from main app')
        _, self.stdout_routing_id = recv_string_router(self.stdout_socket_interface, True)
        # send_string_router_dealer(str(os.getpid()), self.stdout_routing_id, self.stdout_socket_interface)
        print('RenaScript: Waiting for command_info routing ID from main app')
        _, self.command_info_routing_id = recv_string_router(self.command_info_socket_interface, True)
        # redirect stdout
        sys.stdout = RedirectStdout(socket_interface=self.stdout_socket_interface, routing_id=self.stdout_routing_id)

        # set up measuring realtime performance
        self.loop_durations = deque(maxlen=script_fps_counter_buffer_size)
        self.run_while_start_times = deque(maxlen=script_fps_counter_buffer_size)
        # setup inputs and outputs
        self.inputs = dict()
        self.outputs = dict()

        # create data buffers

        print('RenaScript: Script init successfully')

    @abstractmethod
    def init(self):
        """
        Start will be called once when the run button is hit.
        """
        pass

    @abstractmethod
    def loop(self):
        """
        Loop is called <run frequency> times per second
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Loop is called <run frequency> times per second
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
            start_time = time.time()
            try:
                self.loop()
            except Exception as e:
                print('Exception in loop: ', e)
            self.loop_durations.append(time.time() - start_time)
            self.run_while_start_times.append(start_time)
            # receive command from main process
            command_routing_id = recv_string_router(self.command_info_socket_interface, is_block=False)
            if command_routing_id is not None:
                command = command_routing_id[0]
                if command == SCRIPT_STOP_REQUEST:
                    break
                elif command == SCRIPT_INFO_REQUEST:
                    send_router(np.array([get_fps(self.run_while_start_times), np.mean(self.loop_durations)]), self.command_info_routing_id, self.command_info_socket_interface)
                else:
                    print('unknown command: ' + command)
        try:
            self.cleanup()
        except Exception as e:
            print('Exception in cleanup: ', e)
        send_string_router(SCRIPT_STOP_SUCCESS, self.command_info_routing_id, self.command_info_socket_interface)

    def __del__(self):
        sys.stdout = sys.__stdout__  # return control to regular stdout


class RedirectStdout(object):
    def __init__(self, socket_interface, routing_id):
        self.terminal = sys.stdout
        self.routing_id = routing_id
        self.socket_interface = socket_interface

    def write(self, message):
        self.terminal.write(message)
        send_string_router(SCRIPT_STDOUT_MSG_PREFIX + message, self.routing_id, self.socket_interface)

    def flush(self):
        pass