from pylsl import StreamInfo, StreamOutlet, resolve_streams, cf_double64
from stream_shared import lsl_continuous_resolver


def create_lsl_outlet(stream_name: str, n_channels, srate):
    info = StreamInfo(stream_name, stream_name, n_channels, srate, cf_double64, 'someuuid1234')
    # next make an outlet
    outlet = StreamOutlet(info)
    return outlet


def get_available_lsl_streams(wait_time=.1):
    available_streams = [x.name() for x in lsl_continuous_resolver.results()] + [x.type() for x in lsl_continuous_resolver.results()]
    return [x for x in available_streams]