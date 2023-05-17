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


class RenaAudioInputInterface:

    def __init__(self, input_device_index=0, frames_per_buffer=128, format=pyaudio.paInt16, channels=1, rate=4410):
        self.input_device_index = input_device_index
        self.frames_per_buffer = frames_per_buffer
        self.format = format
        self.channels = channels
        self.rate = rate

        self.frame_duration = 1 / rate

        self.audio = None
        self.stream = None

    def start_sensor(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      frames_per_buffer=self.frames_per_buffer,
                                      input=True,
                                      input_device_index=self.input_device_index)
        # self.stream.start_stream()

    def process_frames(self):

        # read all data from the buffer
        # try:
        frames = self.stream.read(self.stream.get_read_available())
        # except IOError as e:
        #     if e[1] != pyaudio.paInputOverflowed:
        #         raise
        #     data = b'\x00' * self.frames_per_buffer
        #     print("Buffer Error")

        current_time = time.time()

        samples = len(frames) // (self.channels * self.audio.get_sample_size(self.format))
        timestamps = np.array([current_time - (samples - i) * self.frame_duration for i in range(samples)])
        timestamps = timestamps - timestamps[-1] + local_clock() if len(frames) > 0 else np.array([])

        # byte frames to numpy
        frames = np.frombuffer(frames, dtype=np.int16)

        # frames to channel frames
        frames = np.array_split(frames, self.channels)

        return np.array(frames), timestamps

    def stop_sensor(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()


if __name__ == '__main__':
    # LSL example
    info = pylsl.StreamInfo("MyStream", "MyData", 1, 44100, pylsl.cf_int16, "myuniqueid")
    outlet = pylsl.StreamOutlet(info)

    audio_interface = RenaAudioInputInterface(input_device_index=1)
    audio_interface.start_sensor()
    while 1:
        data, timestamps = audio_interface.process_frames()
        if len(timestamps)>0:
            for index, sample in enumerate(data.T):
                outlet.push_sample(sample, timestamp=timestamps[index])
