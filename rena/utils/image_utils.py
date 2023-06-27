import cv2

from rena.presets.Presets import VideoDeviceChannelOrder


def process_image(image, channel_order: VideoDeviceChannelOrder, scale: float):
    if channel_order == VideoDeviceChannelOrder.BGR:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif channel_order == VideoDeviceChannelOrder.RGB:
        pass

    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return image