"""
This file contains the global shared variables relating to the streams

"""
import warnings

from physiolabxr.configs.config import stream_availability_wait_time


class DummyResolver:
    def results(self):
        return []

try:
    from pylsl import ContinuousResolver
    lsl_continuous_resolver = ContinuousResolver(forget_after=stream_availability_wait_time)  # TODO add forget_after to settings, a stream will be marked unavailable after 2 seconds
except:
    lsl_continuous_resolver = DummyResolver()
    warnings.warn("pylsl is not installed, will use a dummy resolver. LSL interface will not work.")