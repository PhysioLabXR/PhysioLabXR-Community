import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from pydoc import locate

import numpy as np

from exceptions.exceptions import RenaError, BadOutputError
from rena.config import script_fps_counter_buffer_size
from rena.shared import SCRIPT_STDOUT_MSG_PREFIX, SCRIPT_STOP_REQUEST, SCRIPT_STOP_SUCCESS, SCRIPT_INFO_REQUEST, \
    SCRIPT_PARAM_CHANGE, ParamChange
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.utils.data_utils import validate_output
from rena.utils.general import get_fps, DataBuffer
from rena.utils.lsl_utils import create_lsl_outlet
from rena.utils.networking_utils import recv_string_router, send_string_router, send_router, recv_data_dict


class RenaScript(ABC, threading.Thread):
    """
    An abstract class for implementing scripting models.
    """

    def __init__(self, inputs, buffer_sizes, outputs, output_num_channels, params, port, run_frequency, time_window, *args, **kwargs):
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
        self.info_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_INFO',
                                                      port_id=port + 1,
                                                      identity='server',
                                                      pattern='router-dealer')
        self.input_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_INPUT',
                                                       port_id=port + 2,
                                                       identity='server',
                                                       pattern='router-dealer')
        self.command_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_COMMAND',
                                                         port_id=port + 3,
                                                         identity='server',
                                                         pattern='router-dealer')
        print('RenaScript: Waiting for stdout routing ID from main app')
        _, self.stdout_routing_id = recv_string_router(self.stdout_socket_interface, True)
        # send_string_router_dealer(str(os.getpid()), self.stdout_routing_id, self.stdout_socket_interface)
        print('RenaScript: Waiting for info routing ID from main app')
        _, self.info_routing_id = recv_string_router(self.info_socket_interface, True)
        print('RenaScript: Waiting for command routing ID from main app')
        _, self.command_routing_id = recv_string_router(self.command_socket_interface, True)
        # redirect stdout
        sys.stdout = RedirectStdout(socket_interface=self.stdout_socket_interface, routing_id=self.stdout_routing_id)

        # set up measuring realtime performance
        self.loop_durations = deque(maxlen=script_fps_counter_buffer_size)
        self.run_while_start_times = deque(maxlen=script_fps_counter_buffer_size)
        # setup inputs and outputs
        self.input_names = inputs
        self.inputs = DataBuffer(data_type_buffer_sizes=buffer_sizes)
        self.run_frequency = run_frequency
        # set up the outputs
        self.output_names = outputs
        self.output_num_channels = dict([(x, n) for x, n in zip(outputs, output_num_channels)])
        self._output_default = dict([(x, None) for x in outputs])
        self.output_outlets = dict([(x, create_lsl_outlet(x, n, run_frequency)) for x, n in zip(outputs, output_num_channels)])
        self.outputs = None  # dict holding the output data

        # set up the parameters
        self.params = params

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
            print('Exception in init(): {0} {1}'.format(type(e), str(e)))
        # start the loop here, accept interrupt command
        print('Entering loop')
        while True:
            self.outputs = self._output_default  # reset the output to be default values
            data_dict = recv_data_dict(self.input_socket_interface)
            self.update_input_buffer(data_dict)
            loop_start_time = time.time()
            try:
                self.loop()
            except Exception as e:
                print('Exception in loop(): {0} {1}'.format(type(e), str(e)))
            self.loop_durations.append(time.time() - loop_start_time)
            self.run_while_start_times.append(loop_start_time)
            # receive info request from main process
            info_msg_routing_id = recv_string_router(self.info_socket_interface, is_block=False)
            if info_msg_routing_id is not None:
                request = info_msg_routing_id[0]
                if request == SCRIPT_INFO_REQUEST:
                    send_router(np.array([get_fps(self.run_while_start_times), np.mean(self.loop_durations)]),
                                self.info_routing_id, self.info_socket_interface)
                else:
                    print('unknown info request: ' + request)
            # receive command from main process
            command_msg_routing_id = recv_string_router(self.command_socket_interface, is_block=False)
            if command_msg_routing_id is not None:
                command = command_msg_routing_id[0]
                if command == SCRIPT_STOP_REQUEST:
                    break
                if command == SCRIPT_PARAM_CHANGE:
                    # receive the rest of the mssage about parameter change
                    change_info, _ = recv_string_router(self.command_socket_interface, is_block=True)
                    _, value = self.command_socket_interface.socket.recv_multipart()  # first element is routing ID
                    change_str, param_name, param_type = change_info.split('|')
                    param_type = locate(param_type)
                    change = ParamChange(change_str)
                    if change == ParamChange.ADD or change == ParamChange.CHANGE:
                        self.params[param_name] = np.frombuffer(np.array(value).tobytes(), dtype=param_type)[0]
                    else:
                        self.params.pop(param_name)
                    print('RenaScript: param changed')
                else:
                    print('unknown command: ' + command)
            # send the output if they are updated in the loop
            for stream_name, data in self.outputs.items():
                if data is not None:
                    try:
                        validate_output(data, self.output_num_channels[stream_name])
                        self.output_outlets[stream_name].push_sample(data)
                    except Exception as e:
                        if type(e) == BadOutputError:
                            print('Bad output data is given to stream {0}: {1}'.format(stream_name, str(e)))
                        else:
                            print('Unknown error occured when trying to send output data: {0}'.format(str(e)))
        # exiting the script loop
        try:
            self.cleanup()
        except Exception as e:
            print('Exception in cleanup(): {0} {1}'.format(type(e), str(e)))
        print('Sending stop success')
        send_string_router(SCRIPT_STOP_SUCCESS, self.command_routing_id, self.command_socket_interface)

    def __del__(self):
        sys.stdout = sys.__stdout__  # return control to regular stdout

    def update_input_buffer(self, data_dict):
        self.inputs.update_buffers(data_dict)
        # self.inputs = dict([(n, np.empty(0)) for n in self.input_names])
        # self.inputs_timestamps = dict([(n, np.empty(0)) for n in self.input_names])
        # for key, data_timestamps in data_dict.items():
        #     if data_timestamps:
        #         self.inputs[key] = data_timestamps[0]
        #         self.inputs_timestamps[key] = data_timestamps[1]

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
