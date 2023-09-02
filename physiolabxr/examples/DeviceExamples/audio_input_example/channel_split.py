import pyaudio
import wave
import numpy as np

# Set parameters for recording
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
seconds = 5
input_device_index = 2
filename = "output.wav"


p = pyaudio.PyAudio()

# Open the stream
stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True,
                input_device_index=input_device_index)

frames = []

stream.is_active()

# Record audio for the specified number of seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()

# Terminate the PyAudio object
p.terminate()

# Convert the audio data to a NumPy array
audio_data = b"".join(frames)
audio_np = np.frombuffer(audio_data, dtype=np.int16)

# Split the audio data into separate channels
channel_1 = audio_np[::channels]
channel_2 = audio_np[1::channels]

print("John")