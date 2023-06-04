import time
import numpy as np
import pyaudio
from pylsl import local_clock
import soundfile as sf
from pylsl import StreamInlet, LostError, resolve_byprop
import pylsl

from exceptions.exceptions import LSLStreamNotFoundError, ChannelMismatchError
from rena import config
from rena.config import stream_availability_wait_time
from stream_shared import lsl_continuous_resolver


class DeviceInterface:

    def __init__(self, stream_name, sampling_rate=100):
        self.stream_name = stream_name
        self.sampling_rate = sampling_rate

    def start_sensor(self):
        pass

    def process_frames(self):
        pass
        # return np.array(frames), timestamps

    def stop_sensor(self):
        pass

    def get_sampling_rate(self):
        return self.sampling_rate
