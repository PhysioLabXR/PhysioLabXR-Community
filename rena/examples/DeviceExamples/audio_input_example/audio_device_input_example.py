import time

import pyaudio
import wave
import numpy as np



# set parameters for recording
chunk = 1024  # number of samples per frame
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2  # stereo
fs = 44100  # sampling rate
seconds = 5  # duration of recording

filename = "output.wav"

# initialize PyAudio
p = pyaudio.PyAudio()

# open stream for recording
stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True,
                input_device_index=0)

frames = []  # list to store audio frames

# record audio for the specified duration
for i in range(0, int(fs / chunk * seconds)):
    print(stream.get_read_available())
    data = stream.read(chunk)
    frames.append(data)
    # print(time.time())

# stop and close the stream
stream.stop_stream()
stream.close()

# terminate PyAudio
p.terminate()

# save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

