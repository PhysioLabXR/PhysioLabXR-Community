import cv2
import numpy

from physiolabxr.presets.PlotConfig import ImageFormat, ChannelFormat
from physiolabxr.presets.PresetEnums import VideoDeviceChannelOrder


def process_image(image, rgb_channel_order: VideoDeviceChannelOrder=None, scale: float=1.0):
    if rgb_channel_order is not None:
        if rgb_channel_order == VideoDeviceChannelOrder.BGR:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif rgb_channel_order == VideoDeviceChannelOrder.RGB:
            pass

    new_width = max(1, int(image.shape[1] * scale))
    new_height = max(1, int(image.shape[0] * scale))

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return image


def rotate_image(image, rotation_clockwise_degree: int=0):
    if rotation_clockwise_degree==0:
        return image
    elif rotation_clockwise_degree==90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_clockwise_degree==180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_clockwise_degree==270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


