"""
This file contains the global shared variables relating to the streams

"""
from pylsl import ContinuousResolver

lsl_continuous_resolver = ContinuousResolver(forget_after=2)
