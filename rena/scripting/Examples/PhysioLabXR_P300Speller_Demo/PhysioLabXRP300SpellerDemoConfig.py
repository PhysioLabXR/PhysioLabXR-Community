from enum import Enum

# OpenBCI Stream Name
OpenBCIStreamName = 'OpenBCI_Cython_8_LSL'

# Sampling Rate
eeg_sampling_rate = 250

channel_num = 8

# eeg channel index for the experiment. The Cython8 board has 8 eeg channels
eeg_channel_index = [0, 1, 2, 3, 4, 5, 6, 7]

# eeg channels from cython 8 board
eeg_channel_names = [
    "Fz",
    "Cz",
    "Pz",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1"
]

# channel types are 8 eeg channels
channel_types = ['eeg'] * 8

# event marker channel index

class IndexClass(int, Enum):
    pass
class EventMarkerChannelInfo(IndexClass):
    StateEnterExitMarker = 0,
    FlashBlockStartEndMarker = 1,
    FlashingMarker = 2,
    FlashingItemIndexMarker = 3, # the 0 - 5 is row, 7 - 11 is column
    FlashingTargetMarker = 4,
    StateInterruptMarker = 5,

class ExperimentStateMarker(IndexClass):
    StartState = 1,
    TrainIntroductionState = 2,
    TrainState = 3,
    TestIntroductionState = 4,
    TestState = 5,
    EndState = 6,



