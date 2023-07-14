import pyaudio


def get_audio_devices():
    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:

            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
            print(audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels'))





if __name__ == '__main__':
    get_audio_devices()
