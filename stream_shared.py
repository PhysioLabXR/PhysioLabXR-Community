"""
This file contains the global shared variables relating to the streams

"""
from pylsl import ContinuousResolver

class DummyResolver:
    def results(self):
        return []

lsl_continuous_resolver = ContinuousResolver(forget_after=2)  # TODO add forget_after to settings, a stream will be marked unavailable after 2 seconds
# lsl_continuous_resolver = DummyResolver()
