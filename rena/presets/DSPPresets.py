
# class (Enum):
#     TIMESERIES = 0
#     IMAGE = 1
#     BARCHART = 2
#     SPECTROGRAM = 3
from dataclasses import dataclass, asdict

from rena.presets.preset_class_helpers import SubPreset
from rena.utils.realtime_DSP import DataProcessor, RealtimeButterBandpass


# @dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
# class DSPPreset(metaclass=SubPreset):
#     pass
#
# @dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
# class RealtimeButterBandpassPreset(DSPPreset):
#     pass
#
# @dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
# class RealtimeNotch(DSPPreset):
#     pass



if __name__ == '__main__':
    # test = DSPPreset()
    print("TEST")