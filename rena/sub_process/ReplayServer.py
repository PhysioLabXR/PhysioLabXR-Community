import math
import pickle
import threading
import time
from collections import deque

import numpy as np
import zmq
from pylsl import pylsl

from rena import config, shared
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.utils.data_utils import RNStream


class ReplayServer(threading.Thread):
    def __init__(self, command_info_interface):
        super().__init__()
        self.command_info_interface = command_info_interface
        self.is_replaying = False

        self.virtual_clock_offset = None
        self.start_time = None
        self.end_time = None
        self.virtual_clock = None
        self.total_time = None

        self.stream_data = None

        self.stream_names = None
        self.selected_stream_indices = None

        self.outlets = []
        self.next_sample_of_stream = []  # index of the next sample of each stream that will be send
        self.chunk_sizes = []  # how many samples should be published at once

        self.running = True
        self.main_program_routing_id = None

        # fps counter
        self.tick_times = deque(maxlen=50)

    def run(self):
        while self.running:
            if not self.is_replaying:
                print('ReplayClient: pending on start replay command')
                command = self.recv_string(is_block=True)
                if command.startswith(shared.START_COMMAND):
                    file_loc = command.split("!")[1]
                    try:
                        if file_loc.endswith('.dats'):
                            rns_stream = RNStream(file_loc)
                            self.stream_data = rns_stream.stream_in(
                                ignore_stream=['0', 'monitor1'])  # TODO ignore replaying image data for now
                        elif file_loc.endswith('.p'):
                            self.stream_data = pickle.load(open(file_loc, 'rb'))
                        else:
                            self.send_string(shared.FAIL_INFO + 'Unsupported file type')
                            return
                    except FileNotFoundError:
                        self.send_string(shared.FAIL_INFO + 'File not found at ' + file_loc)
                        return
                    self.is_replaying = True
                    self.setup_stream()
                    self.send_string(shared.START_SUCCESS_INFO + str(self.total_time))
                    self.send(np.array([self.start_time, self.end_time, self.total_time, self.virtual_clock_offset]))
                    self.send_string('|'.join(self.stream_data.keys()))
                elif command == shared.TERMINATE_COMMAND:
                    self.running = False
            else:
                while len(self.selected_stream_indices) > 0:
                    self.tick_times.append(time.time())
                    print("Replay FPS {0}".format(self.get_fps()), end='\r')
                    # streams get removed from the list if there are no samples left to play
                    self.replay()

                    command = self.recv_string(is_block=False)
                    if command == shared.VIRTUAL_CLOCK_REQUEST:
                        self.send(self.virtual_clock)
                    elif command == shared.STOP_COMMAND:
                        # process stop command
                        self.reset_replay()
                        self.is_replaying = False
                        self.send_string(shared.STOP_SUCCESS_INFO)
                        break
                    elif command == shared.TERMINATE_COMMAND:
                        self.running = False
                        break
                print('replay finished')
                if self.is_replaying:  # the case of a finished replay
                    self.is_replaying = False
                    command = self.recv_string(is_block=True)
                    if command == shared.VIRTUAL_CLOCK_REQUEST:
                        self.send(np.array(-1.))
                    else: raise Exception('Unexpected command ' + command)
        self.send_string(shared.TERMINATE_SUCCESS_COMMAND)
        print("Replay terminated")
        # return here

    def reset_replay(self):
        self.tick_times = deque(maxlen=50)
        self.outlets = []
        self.next_sample_of_stream = []
        self.chunk_sizes = []
        self.virtual_clock_offset = None
        self.start_time = None
        self.end_time = None
        self.virtual_clock = None
        self.total_time = None
        self.stream_data = None
        self.stream_names = None
        self.selected_stream_indices = None

    def replay(self):
        nextStreamIndex = None
        nextBlockingTimestamp = None

        # determine which stream to send next
        for i, stream_name in enumerate(self.stream_names):
            stream = self.stream_data[stream_name]
            # when a chunk can be send depends on it's last sample's timestamp
            blockingElementIdx = self.next_sample_of_stream[i] + self.chunk_sizes[i] - 1
            try:
                blockingTimestamp = stream[1][blockingElementIdx]
            except Exception as e:
                raise (e)
            if nextBlockingTimestamp is None or blockingTimestamp <= nextBlockingTimestamp:
                nextStreamIndex = i
                nextBlockingTimestamp = blockingTimestamp

        # retrieve the data and timestamps to be send
        nextStream = self.stream_data[self.stream_names[nextStreamIndex]]
        # print("chunk sizes: ", self.chunk_sizes)
        chunkSize = self.chunk_sizes[nextStreamIndex]

        nextChunkRangeStart = self.next_sample_of_stream[nextStreamIndex]
        nextChunkRangeEnd = nextChunkRangeStart + chunkSize

        nextChunkTimestamps = nextStream[1][nextChunkRangeStart: nextChunkRangeEnd]
        nextChunkValues = (nextStream[0][..., nextChunkRangeStart: nextChunkRangeEnd]).transpose()

        # prepare the data (if necessary)
        if isinstance(nextChunkValues, np.ndarray):
            # load_xdf loads numbers into numpy arrays (strings will be put into lists). however, LSL doesn't seem to
            # handle them properly as providing data in numpy arrays leads to falsified data being sent. therefore the data
            # are converted to lists
            nextChunkValues = nextChunkValues.tolist()
        self.next_sample_of_stream[nextStreamIndex] += chunkSize

        stream_length = nextStream[0].shape[-1]
        # calculates a lower chunk_size if there are not enough samples left for a "complete" chunk
        if stream_length < self.next_sample_of_stream[nextStreamIndex] + chunkSize:
            # print("CHUNK UPDATE")
            self.chunk_sizes[nextStreamIndex] = stream_length - self.next_sample_of_stream[nextStreamIndex]

        self.virtual_clock = pylsl.local_clock() - self.virtual_clock_offset
        # TODO: fix this
        sleepDuration = nextBlockingTimestamp - self.virtual_clock
        if sleepDuration > 0:
            time.sleep(sleepDuration)

        outlet = self.outlets[nextStreamIndex]
        # print("outlet for this replay is: ", outlet)
        nextStreamName = self.stream_names[nextStreamIndex]
        if chunkSize == 1:
            # print(str(nextChunkTimestamps[0] + virtualTimeOffset) + "\t" + nextStreamName + "\t" + str(nextChunkValues[0]))
            outlet.push_sample(nextChunkValues[0], nextChunkTimestamps[0] + self.virtual_clock_offset)
            # print("pushed, chunk size 1")
            # print(nextChunkValues)
        else:
            # according to the documentation push_chunk can only be invoked with exactly one (the last) time stamp
            outlet.push_chunk(nextChunkValues, nextChunkTimestamps[-1] + self.virtual_clock_offset)
            # print("pushed else")
            # chunks are not printed to the terminal because they happen hundreds of times per second and therefore
            # would make the terminal output unreadable

        # remove this stream from the list if there are no remaining samples
        if self.next_sample_of_stream[nextStreamIndex] >= stream_length:
            self.selected_stream_indices.remove(self.selected_stream_indices[nextStreamIndex])
            self.outlets.remove(self.outlets[nextStreamIndex])
            self.next_sample_of_stream.remove(self.next_sample_of_stream[nextStreamIndex])
            self.chunk_sizes.remove(self.chunk_sizes[nextStreamIndex])
            self.stream_names.remove(self.stream_names[nextStreamIndex])

        # self.replay_progress_signal.emit(playback_position) # TODO add another TCP interface to communicate back
        # print("virtual clock time: ", self.virtual_clock)

    def setup_stream(self):
        self.virtual_clock = math.inf
        self.end_time = - math.inf

        # flatten any high dim data
        video_keys = []
        for stream_name, (data, _) in self.stream_data.items():
            if len(data.shape) > 2:
                time_dim = data.shape[-1]
                self.stream_data[stream_name][0] = data.reshape((-1, time_dim))
                video_keys.append(stream_name)
        # change the name of video (high dim) data
        for k in video_keys:
            self.stream_data['video' + k] = self.stream_data.pop(k)

        # setup the streams
        self.stream_names = list(self.stream_data)

        for i in range(0, len(self.stream_names)):
            self.outlets.append(None)
            self.next_sample_of_stream.append(0)
            self.chunk_sizes.append(1)

        print("Creating outlets")
        print("\t[index]\t[name]")

        self.selected_stream_indices = list(range(0, len(self.stream_names)))
        # create LSL outlets
        for streamIndex, stream_name in enumerate(self.stream_names):
            # if not self.isStreamVideo(stream_name):
            # stream_channel_count = self.stream_data[stream_name][0].shape[0]
            stream_channel_count = int(np.prod(self.stream_data[stream_name][0].shape[:-1]))
            stream_channel_format = 'double64'
            stream_source_id = 'Replay Stream - ' + stream_name
            outlet_info = pylsl.StreamInfo(stream_name, '', stream_channel_count, 0.0, stream_channel_format,
                                           stream_source_id)

            self.outlets[streamIndex] = pylsl.StreamOutlet(outlet_info)
            print("\t" + str(streamIndex) + "\t" + stream_name)

        self.virtual_clock_offset = 0
        for stream in self.stream_names:
            # find the start time
            if self.virtual_clock is None or self.stream_data[stream][1][0] < self.virtual_clock:
                # virtual clock will be set to the timestamp of the first received stream data
                self.virtual_clock = self.stream_data[stream][1][0]
                self.start_time = self.virtual_clock

            # find the end time
            if self.stream_data[stream][1][-1] > self.end_time:
                self.end_time = self.stream_data[stream][1][-1]

            self.total_time = self.end_time - self.start_time

        self.virtual_clock_offset = pylsl.local_clock() - self.virtual_clock
        print("Offsetting replayed timestamps by " + str(self.virtual_clock_offset))
        print("start time and end time ", self.start_time, self.end_time)

    def isStreamVideo(self, stream):
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

    def recv_string(self, is_block):
        if is_block:
            self.main_program_routing_id, command = self.command_info_interface.socket.recv_multipart(flags=0)
            return command.decode('utf-8')
        else:
            try:
                self.main_program_routing_id, command = self.command_info_interface.socket.recv_multipart(
                    flags=zmq.NOBLOCK)
                return command.decode('utf-8')
            except zmq.error.Again:
                return None  # no message has arrived at the socket yet

    def get_fps(self):
        try:
            return len(self.tick_times) / (self.tick_times[-1] - self.tick_times[0])
        except ZeroDivisionError:
            return 0

def start_replay_server():
    print("Replay Client Started")
    command_info_interface = RenaTCPInterface(stream_name='RENA_REPLAY',
                                              port_id=config.replay_port,
                                              identity='server',
                                              pattern='router-dealer')

    replay_client_thread = ReplayServer(command_info_interface)
    replay_client_thread.start()


if __name__ == '__main__':
    start_replay_server()
