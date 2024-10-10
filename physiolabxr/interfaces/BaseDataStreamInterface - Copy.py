import time

import numpy as np
from physiolabxr.interfaces.DataStreamInterface import DataStreamInterface


class BaseDataStreamInterface(DataStreamInterface):
    """
    define variable here
    """
    nominal_sampling_rate = 1000  # change this to the nominal sampling rate of your stream

    def start_stream(self):
        print("start stream")

    def is_stream_available(self):
        return True

    def process_frames(self):
        """
        the example here generate returns one frame with 10 channels
        """
        return np.random.randn(10, 1), time.monotonic()

    def stop_stream(self):
        print("stop stream")
