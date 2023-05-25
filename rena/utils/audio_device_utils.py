import pyaudio


class AudioDevice:

    def __init__(self, device_index, stream_name, channel_num):
        self.stream_name = stream_name
        self.device_index = device_index
        self.channel_num = channel_num




def get_audio_devices_dict():
    print('audio devices')
    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    audio_devices = {}

    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            # audio_devices[i] = audio.get_device_info_by_host_api_device_index(0, i).get('name')
            audio_devices[i] = AudioDevice(device_index=i,
                                           stream_name=audio.get_device_info_by_host_api_device_index(0, i).get('name'),
                                           channel_num = audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels'))
            # print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
            # print(audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels'))
    return audio_devices


if __name__ == '__main__':
    a = get_audio_devices_dict()
    print(a)

# def get_working_camera_id():
#     """
#     deprecated, not in use. Use the more optimized version as in general.get_working_camera_ports()
#     :return:
#     """
#     # checks the first 10 indexes.
#     index = 0
#     arr = []
#     i = 10
#     while i > 0:
#         cap = cv2.VideoCapture(index)
#         if cap.read()[0]:
#             arr.append(index)
#             cap.release()
#         index += 1
#         i -= 1
#     return arr


# def get_working_camera_ports():
#     """
#     Test the ports and returns a tuple with the available ports and the ones that are working.
#     """
#     non_working_ports = []
#     dev_port = 0
#     working_ports = []
#     available_ports = []
#     while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing.
#         camera = cv2.VideoCapture(dev_port)
#         if not camera.isOpened():
#             non_working_ports.append(dev_port)
#             print("Video device port %s is not working." %dev_port)
#         else:
#             is_reading, img = camera.read()
#             w = camera.get(3)
#             h = camera.get(4)
#             if is_reading:
#                 print("Video device port %s is working and reads images (%s x %s)" %(dev_port,h,w))
#                 working_ports.append(dev_port)
#             else:
#                 print("Video device port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
#                 available_ports.append(dev_port)
#         dev_port +=1
#         camera.release()
#     return available_ports, working_ports, non_working_ports