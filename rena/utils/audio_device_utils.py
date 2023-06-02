import pyaudio

def get_audio_device_info_dict():
    print('audio devices')
    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    audio_devices_info_dict = {}

    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            audio_devices_info_dict[i] = {
                'stream_name':audio.get_device_info_by_host_api_device_index(0, i).get('name'),
                'audio_device_index':i,
                'channel_num':audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')
            }
            # audio_devices[i] = create_audio_device_stream_preset(
            #     stream_name=audio.get_device_info_by_host_api_device_index(0, i).get('name'),
            #     audio_device_index=i,
            #     channel_num=audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')
            # )

    return audio_devices_info_dict
#


# def create_audio_device_stream_preset(stream_name, audio_device_index,  channel_num):
#     audio_device_preset = AudioDevicePreset(_audio_device_index= audio_device_index)
#
#     channel_indices = list(range(channel_num))
#     channel_names = ["channel" + str(channel_indices[i]) for i in channel_indices]
#     #
#     #
#     audio_device_preset = StreamPreset(
#         stream_name=stream_name,
#         channel_names=channel_names,
#         num_channels=channel_num,
#         group_info=create_default_group_info(channel_num=channel_num),
#         data_type='int16',
#         device_info={},
#         preset_type=PresetType.AUDIODEVICE,
#         device_preset=audio_device_preset
#     )
#
#     return audio_device_preset
