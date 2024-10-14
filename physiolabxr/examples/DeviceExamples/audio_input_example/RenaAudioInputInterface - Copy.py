import time
import numpy as np
import pyaudio
from pylsl import local_clock

from physiolabxr.configs.config import stream_availability_wait_time


class RenaAudioInputInterface:

    def __init__(self, input_device_index=0, frames_per_buffer=1024, format=pyaudio.paInt16, channels=1, rate=44100):

        self.input_device_index = input_device_index
        self.frames_per_buffer = frames_per_buffer
        self.format = format
        self.channels = channels
        self.rate = rate

        self.frame_duration = 1/rate

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
        data = self.stream.read(self.stream.get_read_available())

        # get current timestamp with LSL
        current_time = time.time()
        # timestamps array
        samples = len(data) // (self.channels * self.audio.get_sample_size(self.format))
        # timestamps
        timestamps = np.array([current_time - (samples - i) * self.frame_duration for i in range(samples)])
        timestamps = timestamps - timestamps[-1] + local_clock() if len(timestamps) > 0 else np.array([])



        data = np.frombuffer(data, dtype=np.int16)
        data = np.array_split(data, self.channels)

        return np.array(data), timestamps


        # try:
        #     a = time.time()

        # split_channels = sf.blocks(audio_data, blocksize=self.channels * .get_sample_size(sample_format),
        #                            dtype=sample_format, channels=channels)


        # self = stream_availab


        # data = self.stream.read(1024)

        # if len(data)!=0:
        #     print(data)
        # try:
        #     current_time = time.time()
        #
        #
        # except IOError as e:
        #     print("Error")
            # if e[1] != pyaudio.paInputOverflowed:
            #     raise
            # data = b'\x00' * self.frames_per_buffer
            # print("Buffer Error")


        # a = self.stream.get_read_available()
        # data = self.stream.read(a)
        #
        # if len(data) != 0:
        #     print(len(data))

        # print(data)
        # print(self.stream.get_read_available())
        # data = self.stream.read()
        # print(len(data))
        # frames = []

        # while self.stream.is_active():
        #     try:
        #         if self.stream.get_read_available() >= input_frames_per_buffer:
        #             data = input_stream.read(input_frames_per_buffer)

        # while True:
        #     a = time.time()
        #     data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
        #     print(time.time()-a)
        #     if len(data) == 0:
        #         break
        # data = stream.read(chunk, exception_on_overflow=False)
        #
        # if len(data) == 0:
        #     break
        # try:
        #     a = time.time()
        #     data = self.stream.read(num_frames=self.frames_per_buffer, exception_on_overflow=False)
        #     frames.append(data)
        #     print(time.time()-a)
        # except IOError as e:
        #     if e[1] != pyaudio.paInputOverflowed:
        #         raise
        #     data = '\x00' * self.frames_per_buffer
        #     print("Buffer Error")
        #
        # if len(data) == 0:
        #     return data

    def stop_sensor(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()


if __name__ == '__main__':
    audio_interface = RenaAudioInputInterface()
    audio_interface.start_sensor()
    while 1:
        frame_data, timestamps = audio_interface.process_frames()
