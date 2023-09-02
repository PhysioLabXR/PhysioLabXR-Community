import copy
import json
import math
import os.path
import pickle
import threading
import time
import warnings
from collections import deque, defaultdict

import numpy as np
import zmq
from pylsl import pylsl

from physiolabxr.configs import shared
from physiolabxr.configs import config
from physiolabxr.presets.PresetEnums import PresetType
from physiolabxr.sub_process.TCPInterface import RenaTCPInterface
from physiolabxr.utils.RNStream import RNStream
from physiolabxr.utils.xdf_utils import load_xdf


class ReplayServer(threading.Thread):
    def __init__(self, command_info_interface):
        super().__init__()
        self.original_stream_data = None
        self.command_info_interface = command_info_interface
        self.is_replaying = False
        self.is_paused = False

        self.virtual_clock_offset = None
        self.start_time = None
        self.end_time = None
        self.virtual_clock = None
        self.total_time = None
        self.slider_offset_time = None

        self.stream_data = None

        self.stream_names = None
        self.remaining_stream_names = None

        self.outlet_infos = []
        self.outlets = {}
        self.next_sample_index_of_stream = {}  # index of the next sample of each stream that will be sent, this list contains the same number of items as the number of streams in the replay
        self.chunk_sizes = {}  # how many samples should be published at once, this list contains the same number of items as the number of streams in the replay

        self.running = True
        self.main_program_routing_id = None
        self.replay_finished = False

        # fps counter
        self.tick_times = deque(maxlen=2 ** 16)
        self.push_data_times = deque(maxlen=2 ** 16)

        self.c = 0  # use for counting in the pause session
        self.pause_time_offset_total = 0  # use for tracking the total paused time for this replay
        self.pause_start_time = None

        self.previous_file_loc = None
        self.previous_stream_data = None

    def run(self):
        while self.running:
            if not self.is_replaying:
                print('ReplayServer: pending on start replay command')
                command = self.recv_string(is_block=True)
                if command.startswith(shared.LOAD_COMMAND):
                    file_location = command.split("!")[1]
                    if file_location != self.previous_file_loc:
                        if not os.path.exists(file_location):
                            self.send_string(shared.FAIL_INFO + 'File not found at ' + file_location)
                            self.reset_replay()
                            continue
                        try:
                            print(f'ReplayServer: loading file at {file_location}')
                            if file_location.endswith('.dats'):
                                rns_stream = RNStream(file_location)
                                # self.stream_data = rns_stream.stream_in(ignore_stream=['0', 'monitor1'])  # TODO ignore replaying image data for now
                                self.original_stream_data = rns_stream.stream_in()
                            elif file_location.endswith('.p'):
                                self.original_stream_data = pickle.load(open(file_location, 'rb'))
                                # if '0' in self.stream_data.keys(): self.stream_data.pop('0')
                                # if 'monitor1' in self.stream_data.keys(): self.stream_data.pop('monitor1')
                            elif file_location.endswith('.xdf'):
                                self.original_stream_data = load_xdf(file_location)
                            else:
                                raise ValueError('Unsupported file type')
                        except Exception as e:
                            self.send_string(shared.FAIL_INFO + f'Failed to load file {e}')
                            self.reset_replay()
                            continue

                    self.previous_file_loc = file_location
                    self.send_string(shared.LOAD_SUCCESS_INFO + str(self.total_time))
                    self.stream_data = copy.deepcopy(self.original_stream_data)
                    self.setup_stream()
                    self.send(np.array([self.start_time, self.end_time, self.total_time, self.virtual_clock_offset]))  # send the timing info
                    self.send_string('|'.join(self.original_stream_data.keys()))  # send the stream names
                    # send the number of channels, average sampling rate, and number of time points
                    stream_info = defaultdict(dict)
                    for stream_name, (data, timestamps) in self.original_stream_data.items():
                        stream_info[stream_name]['n_channels'] = data.shape[0]
                        stream_info[stream_name]['n_timepoints'] = len(timestamps)
                        stream_info[stream_name]['srate'] = stream_info[stream_name]['n_timepoints'] / (timestamps[-1] - timestamps[0])
                        stream_info[stream_name]['data_type'] = str(data.dtype)
                    self.send_string(json.dumps(stream_info))
                elif command == shared.GO_AHEAD_COMMAND:
                    # go ahead and open the streams
                    # outlets are not created in setup before receiving the go-ahead from the main process because the
                    # main process need to check if there're duplicate stream names with the streams being replayed
                    # check if the stream has been setup, becuase if we come back here from a finished replay, the stream would have been reset
                    replay_stream_info = json.loads(self.recv_string(is_block=True))
                    self.stream_data = {k: v for k, v in self.original_stream_data.items() if k in replay_stream_info.keys()}

                    # for stream_name, (interface, port) in replay_stream_info.items():
                    #     self.stream_data

                    self.setup_stream()  # set up streams again, because some streams may be disabled by user
                    try:
                        for outlet_info in self.outlet_infos:
                            if replay_stream_info[outlet_info.name()]['preset_type'] == PresetType.ZMQ.value:
                                port = replay_stream_info[outlet_info.name()]['port_number']
                                socket = self.command_info_interface.context.socket(zmq.PUB)
                                socket.bind("tcp://*:%s" % port)
                                self.outlets[outlet_info.name()] = socket
                            else:
                                self.outlets[outlet_info.name()] = pylsl.StreamOutlet(outlet_info)
                    except zmq.error.ZMQError as e:
                        self.send_string(shared.FAIL_INFO + f'Failed to open stream: {e}')
                        self.reset_replay()
                        continue
                    self.send_string(shared.SUCCESS_INFO)
                    self.send(np.array([self.start_time, self.end_time, self.total_time, self.virtual_clock_offset]))  # send the timing info
                    print(f"replay started for streams: {list(self.stream_data)}")
                    self.is_replaying = True  # this is the only entry point of the replay loop
                elif command == shared.PERFORMANCE_REQUEST_COMMAND:
                    self.send(self.get_average_loop_time())
                elif command == shared.TERMINATE_COMMAND:
                    self.running = False
                    break
            else:
                while True:  # the replay loop
                    if len(self.remaining_stream_names) == 0:
                        self.replay_finished = True
                        break
                    if not self.is_paused:
                        self.tick_times.append(time.time())
                        # print("Replay FPS {0}".format(self.get_fps()), end='\r')
                        # streams get removed from the list if there are no samples left to play
                        self.replay()
                    else:
                        pause_time_offset = pylsl.local_clock() - self.pause_start_time
                        self.pause_time_offset_total = self.pause_time_offset_copy + pause_time_offset
                        self.update_virtual_clock()  # time since replay start + first stream timestamps
                    # process commands
                    command = self.recv_string(is_block=False)
                    if command == shared.VIRTUAL_CLOCK_REQUEST:
                        self.send(self.virtual_clock)
                    elif command == shared.PLAY_PAUSE_COMMAND:  # handle play_pause command
                        print("command received from replay server: ", command)
                        if not self.is_paused:
                            self.pause_start_time = pylsl.local_clock()
                            self.pause_time_offset_copy = copy.copy(self.pause_time_offset_total)
                        else:
                            print(f"resumed: pause time is ticking: {self.pause_time_offset_total}")
                        self.is_paused = not self.is_paused
                        self.send_string(shared.PLAY_PAUSE_SUCCESS_INFO)
                    elif type(command) is str and shared.SLIDER_MOVED_COMMAND in command:
                        # process slider moved command
                        times = self.recv(is_block=True)
                        set_to_time, slider_offset_time = np.frombuffer(times, dtype=float)
                        self.slider_offset_time += slider_offset_time
                        self.set_to_time(set_to_time)
                        self.send_string(shared.SLIDER_MOVED_SUCCESS_INFO)
                    elif command == shared.STOP_COMMAND:  # process stop command
                        self.is_replaying = False
                        self.is_paused = False  # reset is_paused in case is_paused had been set to True
                        self.send_string(shared.STOP_SUCCESS_INFO)
                        break
                    elif command == shared.TERMINATE_COMMAND:
                        self.running = False
                        break

                print('Replay Server: exited replay loop')
                if self.replay_finished:  # the case of a finished replay
                    self.replay_finished = False
                    command = self.recv_string(is_block=True)
                    if command == shared.VIRTUAL_CLOCK_REQUEST:
                        self.send(np.array(-1.))
                    else:
                        raise Exception('Unexpected command ' + command)
                self.reset_replay()

        self.send_string(shared.TERMINATE_SUCCESS_COMMAND)
        del self.command_info_interface  # close the socket and terminate the context
        print("Replay terminated")
        # return here

    def reset_replay(self):
        self.next_sample_index_of_stream = {}
        self.chunk_sizes = {}  # chunk sizes are initialized to 1 in setup stream
        self.virtual_clock_offset = None
        self.start_time = None
        self.end_time = None
        self.virtual_clock = None
        self.total_time = None
        self.slider_offset_time = None
        self.stream_data = None
        self.remaining_stream_names = None

        self.is_paused = False
        self.is_replaying = False

        self.outlet_infos = []
        # close all outlets if there's any

        outlet_names = list(self.outlets)
        for stream_name in outlet_names:
            if isinstance(self.outlets[stream_name], pylsl.StreamOutlet):
                del self.outlets[stream_name]
                print('Replay Server: Reset replay: removed outlet ' + stream_name)
            else:
                self.outlets[stream_name].close()
                print('Replay Server: Reset replay: closed socket ' + stream_name)
        self.outlets = {}
        self.stream_names = None

        print("Replay Server: Reset replay: removed all outlets")

    def replay(self):
        this_stream_name = None
        this_stream_next_timestamp = None

        # determine which stream to send next
        for i, stream_name in enumerate(self.remaining_stream_names):  # iterate over the remaining data streams in the replay file
            stream = self.stream_data[stream_name]
            # when a chunk can be sent depends on its last sample's timestamp
            next_sample_index_of_stream = self.next_sample_index_of_stream[stream_name] + self.chunk_sizes[stream_name] - 1  # at the first call to replay next_sample_of_stream is all zeros, and chunk_sizes is all ones
            try:
                next_timestamp_of_stream = stream[1][next_sample_index_of_stream]
            except Exception as e:
                raise (e)
            if this_stream_next_timestamp is None or next_timestamp_of_stream <= this_stream_next_timestamp:  # find the stream with the smallest timestamp for their next chunk of data
                this_stream_name = stream_name
                this_stream_next_timestamp = next_timestamp_of_stream
        del stream_name

        # retrieve the data and timestamps to be sent
        this_stream_data = self.stream_data[this_stream_name]
        this_chunk_size = self.chunk_sizes[this_stream_name]

        this_next_sample_start_index = self.next_sample_index_of_stream[this_stream_name]
        this_next_sample_end_index = this_next_sample_start_index + this_chunk_size

        this_chunk_timestamps = this_stream_data[1][this_next_sample_start_index: this_next_sample_end_index]
        this_chunk_data = (this_stream_data[0][..., this_next_sample_start_index: this_next_sample_end_index]).transpose()

        # prepare the data (if necessary)
        # if isinstance(this_chunk_data, np.ndarray):
            # load_xdf loads numbers into numpy arrays (strings will be put into lists). however, LSL doesn't seem to
            # handle them properly as providing data in numpy arrays leads to falsified data being sent. therefore the data
            # are converted to lists
            # this_chunk_data = this_chunk_data.tolist()
        self.next_sample_index_of_stream[this_stream_name] += this_chunk_size  # index of the next sample yet to be sent of this stream

        stream_total_num_samples = this_stream_data[0].shape[-1]
        # calculates a lower chunk_size if there are not enough samples left for a "complete" chunk, this will only happen when a stream is running out of samples
        if stream_total_num_samples < self.next_sample_index_of_stream[this_stream_name] + this_chunk_size:
            # print("CHUNK UPDATE")
            self.chunk_sizes[this_stream_name] = stream_total_num_samples - self.next_sample_index_of_stream[this_stream_name]

        # virtual clock is in sync with the replayed stream timestamps, it equals to (replay time) + (original data's first timestamp)
        self.update_virtual_clock()  # time since replay start + first stream timestamps
        sleep_duration = this_stream_next_timestamp - self.virtual_clock
        if sleep_duration > 0:
            time.sleep(sleep_duration)

        outlet = self.outlets[this_stream_name]
        # print("outlet for this replay is: ", outlet)
        push_call_start_time = time.perf_counter()
        if this_chunk_size == 1:
            # the data sample's timestamp is equal to (this sample's timestamp minus the first timestamp of the original data) + time since replay start
            timestamp = this_chunk_timestamps[0] + self.virtual_clock_offset + self.slider_offset_time
            data = this_chunk_data[0]

            if isinstance(outlet, pylsl.stream_outlet):
                outlet.push_sample(data.tolist(), timestamp)
            else:  # zmq
                outlet.send_multipart([bytes(this_stream_name, "utf-8"), np.array(timestamp), data.copy()])  # copy to make data contiguous
            # print("pushed, chunk size 1")
            # print(nextChunkValues)
        else:
            # according to the documentation push_chunk can only be invoked with exactly one (the last) time stamp
            timestamps = this_chunk_timestamps + self.virtual_clock_offset + self.slider_offset_time
            data = this_chunk_data
            if isinstance(outlet, pylsl.stream_outlet):
                outlet.push_chunk(data.tolist(), timestamps[-1])
            else:  # zmq
                for i in range(len(timestamps)):
                    outlet.send_multipart([bytes(this_stream_name, "utf-8"), np.array(timestamps[i]), data[i].copy()])  # copy to make data contiguous
            # print("pushed else")
            # chunks are not printed to the terminal because they happen hundreds of times per second and therefore
            # would make the terminal output unreadable
        self.push_data_times.append(time.perf_counter() - push_call_start_time)

        # remove this stream from the list if there are no remaining samples
        if self.next_sample_index_of_stream[this_stream_name] >= stream_total_num_samples:
            self.remaining_stream_names.remove(this_stream_name)
            # self.outlets.remove(self.outlets[nextStreamIndex])
            # self.next_sample_of_stream.remove(self.next_sample_of_stream[next_stream_index])
            # self.chunk_sizes.remove(self.chunk_sizes[next_stream_index])
            # self.stream_names.remove(self.stream_names[next_stream_index])

        # self.replay_progress_signal.emit(playback_position) # TODO add another TCP interface to communicate back
        # print("virtual clock time: ", self.virtual_clock)

    def setup_stream(self):
        self.virtual_clock = math.inf
        self.end_time = - math.inf
        self.outlets = {}
        self.outlet_infos = []
        self.slider_offset_time = 0
        self.tick_times = deque(maxlen=2 ** 16)
        self.push_data_times = deque(maxlen=2 ** 16)

        # flatten any high dim data
        video_keys = []
        for stream_name, (data, _) in self.stream_data.items():
            if len(data.shape) > 2:
                time_dim = data.shape[-1]
                self.stream_data[stream_name][0] = data.reshape((-1, time_dim))
                video_keys.append(stream_name)
        # change the name of video (high dim) data  TODO video data is ignored for now
        for k in video_keys:
            self.stream_data['video' + k] = self.stream_data.pop(k)

        # setup the streams
        self.stream_names = list(self.stream_data)

        for stream_name in self.stream_names:
            self.next_sample_index_of_stream[stream_name] = 0
            self.chunk_sizes[stream_name] = 1

        print("Creating outlets")
        print("\t[index]\t[name]")

        self.remaining_stream_names = copy.copy(self.stream_names)
        # create LSL outlets
        for streamIndex, stream_name in enumerate(self.stream_names):
            # if not self.isStreamVideo(stream_name):
            # stream_channel_count = self.stream_data[stream_name][0].shape[0]
            stream_channel_count = int(np.prod(self.stream_data[stream_name][0].shape[:-1]))
            stream_channel_format = 'double64'
            stream_source_id = 'Replay Stream - ' + stream_name
            outlet_info = pylsl.StreamInfo(stream_name, '', stream_channel_count, 0.0, stream_channel_format, stream_source_id)
            self.outlet_infos.append(outlet_info)
            print("\t" + str(streamIndex) + "\t" + stream_name)

        self.virtual_clock_offset = 0
        for stream in self.stream_names:
            # find the start time
            if self.virtual_clock is None or self.stream_data[stream][1][0] < self.virtual_clock:
                # virtual clock will be set to the timestamp of the first received stream data (the earliest data)
                self.virtual_clock = self.stream_data[stream][1][0]
                self.start_time = self.virtual_clock

            # find the end time
            if self.stream_data[stream][1][-1] > self.end_time:
                self.end_time = self.stream_data[stream][1][-1]

            self.total_time = self.end_time - self.start_time

        self.virtual_clock_offset = pylsl.local_clock() - self.virtual_clock
        print("Offsetting replayed timestamps by " + str(self.virtual_clock_offset))
        print("start time and end time ", self.start_time, self.end_time)

    def update_virtual_clock(self):
        self.virtual_clock = pylsl.local_clock() - self.virtual_clock_offset + self.slider_offset_time - self.pause_time_offset_total

    def set_to_time(self, set_to_time):
        remaining_stream_names_copy = copy.deepcopy(self.remaining_stream_names)
        for i, stream_name in enumerate(remaining_stream_names_copy):  # iterate over the remaining data streams in the replay file
            timestamps = self.stream_data[stream_name][1]
            future_timestamps = np.argwhere(timestamps > (self.start_time + set_to_time))
            if len(future_timestamps) == 0:
                self.remaining_stream_names.remove(stream_name)
            else:
                self.next_sample_index_of_stream[stream_name] = np.argwhere(timestamps > (self.start_time + set_to_time))[0][0]

    def is_stream_video(self, stream):
        if stream.isdigit():
            return True
        if ("monitor" in stream) or ("video" in stream):
            return True
        return False

    def send_string(self, string):
        self.command_info_interface.socket.send_multipart(
            [self.main_program_routing_id, string.encode('utf-8')])

    def send(self, data):
        self.command_info_interface.socket.send_multipart(
            [self.main_program_routing_id, data])

    def recv(self, is_block) -> bytes:
        if is_block:
            self.main_program_routing_id, command = self.command_info_interface.socket.recv_multipart(flags=0)
            return command
        else:
            try:
                self.main_program_routing_id, command = self.command_info_interface.socket.recv_multipart(
                    flags=zmq.NOBLOCK)
                return command
            except zmq.error.Again:
                return None  # no message has arrived at the socket yet

    def recv_string(self, is_block) -> str:
        if is_block:
            self.main_program_routing_id, command = self.command_info_interface.socket.recv_multipart(flags=0)
            return command.decode('utf-8')
        else:
            try:
                self.main_program_routing_id, command = self.command_info_interface.socket.recv_multipart(flags=zmq.NOBLOCK)
                return command.decode('utf-8')
            except zmq.error.Again:
                return None  # no message has arrived at the socket yet

    def recv_any(self, is_block):
        if is_block:
            self.main_program_routing_id, command = self.command_info_interface.socket.recv_multipart(flags=0)
            try:
                return command.decode('utf-8')
            except UnicodeDecodeError:
                return command
        else:
            try:
                self.main_program_routing_id, command = self.command_info_interface.socket.recv_multipart(
                    flags=zmq.NOBLOCK)
                try:
                    return command.decode('utf-8')
                except UnicodeDecodeError:
                    return command
            except zmq.error.Again:
                return None  # no message has arrived at the socket yet

    def get_fps(self):
        try:
            return len(self.tick_times) / (self.tick_times[-1] - self.tick_times[0])
        except ZeroDivisionError:
            return 0

    def get_average_loop_time(self):
        try:
            return np.mean(self.push_data_times)
        except ZeroDivisionError:
            return 0


def start_replay_server(replay_port):
    print("Replay Server Started")
    # TODO connect to a different port if this port is already in use
    try:
        command_info_interface = RenaTCPInterface(stream_name='RENA_REPLAY',
                                                  port_id=replay_port,
                                                  identity='server',
                                                  pattern='router-dealer')
    except zmq.error.ZMQError as e:
        warnings.warn("ReplayServer: encounter error setting up ZMQ interface: " + str(e))
        warnings.warn("Replay Server exiting...No replay will be available for this session")
        return
    replay_server_thread = ReplayServer(command_info_interface)
    replay_server_thread.start()


if __name__ == '__main__':
    start_replay_server()
