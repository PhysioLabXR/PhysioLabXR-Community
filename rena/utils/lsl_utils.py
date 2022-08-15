from pylsl import *


def create_lsl_outlet(stream_name: str, n_channels, srate):
    info = StreamInfo(stream_name, stream_name, n_channels, srate, 'float32', 'someuuid1234')
    # next make an outlet
    outlet = StreamOutlet(info)
    return outlet
