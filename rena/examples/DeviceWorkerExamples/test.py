import pyaudio
import time

chunk = 1024
sample_format = pyaudio.paInt16
channels = 2
fs = 44100

p = pyaudio.PyAudio()

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []
timestamps = []

while True:
    data = stream.read(chunk, exception_on_overflow=False)

    if len(data) == 0:
        break

    frames.append(data)

    # Calculate timestamp for each sample
    current_time = time.time()
    samples = len(data) // (channels * p.get_sample_size(sample_format))
    sample_rate = fs
    frame_duration = 1.0 / sample_rate
    frame_timestamps = [current_time - (samples - i) * frame_duration for i in range(samples)]
    timestamps.extend(frame_timestamps)

# Process the collected frames and timestamps...

stream.stop_stream()
stream.close()
p.terminate()