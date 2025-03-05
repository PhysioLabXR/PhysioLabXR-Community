import json
import os
import sys
import threading
import time
import traceback
import warnings
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Dict, Union

import grpc
import numpy as np
import zmq

from physiolabxr.utils.time_utils import get_clock_time

try:
    from pylsl import StreamOutlet
    pylsl_imported = True
except:
    warnings.warn('pylsl is not installed, LSL output will not be available')
    pylsl_imported = False

from physiolabxr.exceptions.exceptions import BadOutputError, ZMQPortOccupiedError, RenaError, ScriptSetupError
from physiolabxr.presets.PresetEnums import PresetType
from physiolabxr.presets.ScriptPresets import ScriptOutput
from physiolabxr.configs.shared import SCRIPT_STOP_REQUEST, SCRIPT_STOP_SUCCESS, SCRIPT_INFO_REQUEST, \
    SCRIPT_PARAM_CHANGE
from physiolabxr.scripting.scripting_enums import ParamChange
from physiolabxr.sub_process.TCPInterface import RenaTCPInterface
from physiolabxr.utils.data_utils import validate_output
from physiolabxr.utils.buffers import get_fps, DataBuffer
from physiolabxr.utils.lsl_utils import create_lsl_outlet
from physiolabxr.utils.networking_utils import recv_string_router, send_string_router, send_router, recv_data_dict


class RenaScript(ABC, threading.Thread):
    """
    An abstract class for implementing scripting models.
    """

    def __init__(self, inputs, input_shapes, buffer_sizes, outputs: List[ScriptOutput], params: dict, run_frequency, time_window,
                 script_path, is_simulate, presets, redirect_stdout, redirect_stderr,
                 stdout_socket_interface, info_socket_interface,
                 input_socket_interface, command_socket_interface,
                 info_routing_id, *args, **kwargs):
        """

        :param inputs:
        :param outputs:
        :param params:
        :param port: the port to which we bind the
        """
        super().__init__()
        self.sim_clock = time.time()
        self.script_path = script_path
        self.redirect_stdout = redirect_stdout
        self.redirect_stderr = redirect_stderr
        self.stdout_socket_interface = stdout_socket_interface

        logging.info('RenaScript: RenaScript Thread started on process {0}'.format(os.getpid()))
        try:
            self.input_socket_interface = input_socket_interface
            self.command_socket_interface = command_socket_interface
        except zmq.error.ZMQError as e:
            raise ScriptSetupError("script failed to set up sockets {0}".format(e))

        self.info_socket_interface = info_socket_interface
        self.info_routing_id = info_routing_id
        logging.info('RenaScript: Waiting for command routing ID from main app for command socket')
        _, self.command_routing_id = recv_string_router(self.command_socket_interface, True)

        # set up measuring realtime performance
        self.loop_durations = deque(maxlen=run_frequency * 2)
        self.max_loop_duration = 0
        self.run_while_start_times = deque(maxlen=run_frequency * 2)
        # setup inputs and outputs
        self.input_names = inputs
        self.inputs = DataBuffer(stream_buffer_sizes=buffer_sizes)
        self.run_frequency = run_frequency
        # set up the outputs
        self.output_presets: Dict[str, ScriptOutput] = {o.stream_name: o for o in outputs}
        self.output_names = [o.stream_name for o in outputs]
        self.output_num_channels = {o.stream_name: o.num_channels for o in outputs}

        self._output_default = dict([(x.stream_name, None) for x in outputs])  # default output with None values
        self.outputs = None  # dict holding the output data

        self.output_outlets = {}

        try:
            self._create_output_streams()
        except RenaError as e:
            traceback.print_exc()
            logging.error('Error setting up output streams: {0}'.format(e))

        # set up the parameters
        self.params = params

        # other variables
        self.is_simulate = is_simulate
        self.input_shapes = input_shapes

        # set up the presets
        self.presets = presets

        self.rpc_server: grpc.Server = None

        logging.info('RenaScript: Script init completed')

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
        logging.info('user init function is called')
        try:
            self.init()
        except Exception as e:
            traceback.print_exc()
            self.redirect_stderr.send_buffered_messages()
        # start the loop here, accept interrupt command
        logging.info('Entering loop')
        while True:
            self.outputs = dict([(s_name, None) for s_name in self.output_outlets.keys()])  # reset the output to be default values
            data_dict = recv_data_dict(self.input_socket_interface)
            self.update_input_buffer(data_dict)
            loop_start_time = time.time()
            try:
                 self.loop()
            except Exception as e:
                # print('Exception in loop(): {0} {1}'.format(type(e), e))
                traceback.print_exc()
                # print(traceback.format_exc())
                self.redirect_stderr.send_buffered_messages()
            this_loop_outputs = time.time() - loop_start_time
            self.loop_durations.append(this_loop_outputs)
            self.max_loop_duration = max(this_loop_outputs, self.max_loop_duration)
            self.run_while_start_times.append(loop_start_time)
            # receive info request from main process
            info_msg_routing_id = recv_string_router(self.info_socket_interface, is_block=False)
            if info_msg_routing_id is not None:
                request = info_msg_routing_id[0]
                if request == SCRIPT_INFO_REQUEST:
                    send_router(np.array([get_fps(self.run_while_start_times), np.mean(self.loop_durations), self.max_loop_duration]),
                                self.info_routing_id, self.info_socket_interface)
                else:
                    logging.warning('unknown info request: ' + request)
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
                    change = ParamChange(change_str)
                    if change == ParamChange.ADD or change == ParamChange.CHANGE:
                        # self.params[param_name] = np.frombuffer(np.array(value).tobytes(), dtype=param_type)[0]
                        self.params[param_name] = json.loads(value.decode('utf-8'))
                    else:
                        self.params.pop(param_name)
                    logging.info('RenaScript: param changed')
                else:
                    logging.warning('unknown command: ' + command)
            # send the output if they are updated in the loop
            for stream_name, data in self.outputs.items():
                if stream_name not in self.output_outlets:
                    logging.error(f'RenaScript: output stream with name {stream_name} not found')
                    continue
                outlet = self.output_outlets[stream_name]
                if data is not None:
                    try:
                        _data, timestamp, is_data_chunk, is_timestamp_chunk = validate_output(data, self.output_num_channels[stream_name])
                        _data = _data.astype(self.output_presets[stream_name].data_type.get_data_type())
                        clock_time = get_clock_time()
                        _timestamp = clock_time if timestamp is None else timestamp  # timestamp is not a chunk when data is not chunk
                        if pylsl_imported and isinstance(outlet, StreamOutlet):
                            if is_data_chunk and is_timestamp_chunk:
                                for i in range(len(_data)):
                                    outlet.push_sample(_data[i].tolist(), timestamp=timestamp[i])  # 0.0 is default value, using it will use the local clock
                            elif is_data_chunk and not is_timestamp_chunk:
                                outlet.push_chunk(_data.tolist(), timestamp=_timestamp)  # timestamp is a number or None if not provided by the user
                            else:
                                # timestamp will never be a chunk in this case when data is not chunk
                                outlet.push_sample(_data.tolist(), timestamp=_timestamp)  # 0.0 is default value, using it will use the local clock
                        else:  # this is a zmq socket
                            if is_data_chunk and is_timestamp_chunk:
                                for i in range(len(_data)):
                                    outlet.send_multipart([bytes(stream_name, "utf-8"), np.array(timestamp[i]), np.ascontiguousarray(_data[i])])
                            elif is_data_chunk and not is_timestamp_chunk:
                                for i in range(len(_data)):
                                    outlet.send_multipart([bytes(stream_name, "utf-8"), np.array(_timestamp), np.ascontiguousarray(_data[i])])
                            else:
                                outlet.send_multipart([bytes(stream_name, "utf-8"), np.array(_timestamp), _data])
                    except Exception as e:
                        if type(e) == BadOutputError:
                            logging.error('Bad output data is given to stream {0}: {1}'.format(stream_name, str(e)))
                        else:
                            logging.error('Unknown error occurred when trying to send output data: {0}'.format(str(e)))
                        traceback.print_exc()
        # exiting the script loop
        try:
            self.cleanup()
        except Exception as e:
            traceback.print_exc()
            self.redirect_stderr.send_buffered_messages()
        if self.rpc_server is not None: self.rpc_server.stop(None)
        logging.info('RenaScript: sending stop success to main app')
        send_string_router(SCRIPT_STOP_SUCCESS, self.command_routing_id, self.command_socket_interface)

    def __del__(self):
        self.stdout_socket_interface.socket.close()
        self.input_socket_interface.socket.close()
        self.info_socket_interface.socket.close()
        self.command_socket_interface.socket.close()
        for outlet in self.output_outlets.values():
            if pylsl_imported and isinstance(outlet, StreamOutlet):
                del outlet
            else:
                outlet.close()
        self.command_socket_interface.context.term()
        sys.stdout = sys.__stdout__  # return control to regular stdout

    def update_input_buffer(self, data_dict):
        if self.is_simulate:
            # print('Sim clock is {}, time is {}'.format(self.sim_clock, time.time()))
            data_dict = dict([(stream_name, (np.random.rand(*shape),
                                        np.linspace(self.sim_clock, time.time(), num=shape[1])
                                        )) for stream_name, shape in self.input_shapes.items()])

            self.sim_clock = time.time()
        self.inputs.update_buffers(data_dict)
        # check_buffer_timestamps_monotonic(self.inputs) TODO
        # confirm timestamsp are monotonousely increasing
        # self.inputs = dict([(n, np.empty(0)) for n in self.input_names])
        # self.inputs_timestamps = dict([(n, np.empty(0)) for n in self.input_names])
        # for key, data_timestamps in data_dict.items():
        #     if data_timestamps:
        #         self.inputs[key] = data_timestamps[0]
        #         self.inputs_timestamps[key] = data_timestamps[1]

    def get_stream_info(self, stream_name, info):
        """
        info can be
        @param stream_name:
        @param info: str, can be one of the following  NominalSamplingRate, ChannelNames, DataType
        @return:
        """
        assert info in ['NominalSamplingRate', 'ChannelNames', 'DataType'], f'Unknown info type {info}'
        if info == 'NominalSamplingRate':
            return self.presets.stream_presets[stream_name].nominal_sampling_rate
        elif info == 'ChannelNames':
            return self.presets.stream_presets[stream_name].channel_names
        elif info == 'DataType':
            return self.presets.stream_presets[stream_name].data_type

    def _create_output_streams(self):
        for stream_name, o_preset in self.output_presets.items():
            if o_preset.interface_type == PresetType.LSL:
                self.output_outlets[stream_name] = create_lsl_outlet(stream_name, o_preset.num_channels, self.run_frequency, o_preset.data_type.get_lsl_type())
            elif o_preset.interface_type == PresetType.ZMQ:
                socket = self.command_socket_interface.context.socket(zmq.PUB)
                try:
                    socket.bind("tcp://*:%s" % o_preset.port_number)
                except zmq.error.ZMQError:
                    logging.warning('Error when binding to port {0} for stream {1}'.format(o_preset.port_number, stream_name))
                    raise ZMQPortOccupiedError(o_preset.port_number)
                self.output_outlets[stream_name] = socket

    def set_output(self, stream_name: str, data: Union[np.ndarray, list, tuple], timestamp: Union[np.ndarray, list, tuple, float]=None) -> None:
        """
        Set the output data of the given stream,
        use this function as an alternative to directly setting the self.outputs["stream_name"] = data

        if you have timestamp for your data, use this function to set the timestamp. Timestamps cannot be set directly in self.outputs

        expectation for @param data

        @param stream_name: the name of the stream to set the output
        @param data: can be a numpy array, tuple or list.
            If data is a 2D list, tuple or array, the first dimension is the number of frames, the second dimension is the number of channels.
            if data is a 1D list, tuple or array, it is treated as a single frame, the number of channels must match the number of channels of the output set in the GUI
        @timestamp: can be a single number or a list/tuple/numpy array of numbers,
            if it is a list/tuple/numpy array, the length of the timestamp must match the number of frames in data
        """
        self.outputs[stream_name] = {'data': data, 'timestamp': timestamp}




