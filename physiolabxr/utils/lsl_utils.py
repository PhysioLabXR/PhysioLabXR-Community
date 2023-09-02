from pylsl import StreamInfo, StreamOutlet
from physiolabxr.utils.stream_shared import lsl_continuous_resolver


def create_lsl_outlet(stream_name: str, n_channels, srate, data_type):
    info = StreamInfo(stream_name, stream_name, n_channels, srate, data_type, 'someuuid1234')
    # next make an outlet
    outlet = StreamOutlet(info)
    return outlet


def get_available_lsl_streams(wait_time=.1):
    available_streams = [x.name() for x in lsl_continuous_resolver.results()] + [x.type() for x in lsl_continuous_resolver.results()]
    return [x for x in available_streams]