from enum import Enum

import numpy as np

# OpenBCI Stream Name
EEG_STREAM_NAME = 'OpenBCICyton8Channels'
EVENT_MARKER_CHANNEL_NAME = 'PhysioLabXRP300SpellerDemoEventMarker'
PREDICTION_PROBABILITY_CHANNEL_NAME = "PhysioLabXRP300SpellerDemoPredictionProbability"

# Sampling Rate
EEG_SAMPLING_RATE = 250

# eeg channel number
EEG_CHANNEL_NUM = 8

# epoch configuration
EEG_EPOCH_T_MIN = -0.2
EEG_EPOCH_T_MAX = 1.0


# eeg channels from cython 8 board
EEG_CHANNEL_NAMES = [
    "Fz",
    "Cz",
    "Pz",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1"
]

# # prediction lsl outlet configuration
# PREDICTION_STREAM_NAME = 'PhysioLabXRP300SpellerDemoPrediction'
# PREDICTION_STREAM_TYPE = 'Prediction'
# PREDICTION_STREAM_CHANNEL_NUM = 36


class IndexClass(int, Enum):
    pass

class EventMarkerChannelInfo(IndexClass):
    StateEnterExitMarker = 0,
    FlashBlockStartEndMarker = 1,
    FlashingMarker = 2,
    FlashingItemIndexMarker = 3,  # the 0 - 5 is row, 7 - 11 is column
    FlashingTargetMarker = 4,
    StateInterruptMarker = 5,

class ExperimentStateMarker(IndexClass):
    StartState = 1,
    TrainIntroductionState = 2,
    TrainState = 3,
    TestIntroductionState = 4,
    TestState = 5,
    EndState = 6,

    # # this is not included in the unity paradigm
    # IDLEState = 7


ROW_FLASH_MARKER_LIST = [0, 1, 2, 3, 4, 5]
COL_FLASH_MARKER_LIST = [7, 8, 9, 10, 11]
Target_Flash_Marker = 1
Non_Target_Flash_Marker = 0

Board = [['A', 'B', 'C', 'D', 'E', 'F'],
         ['G', 'H', 'I', 'J', 'K', 'L'],
         ['M', 'N', 'O', 'P', 'Q', 'R'],
         ['S', 'T', 'U', 'V', 'W', 'X'],
         ['Y', 'Z', '0', '1', '2', '3'],
         ['4', '5', '6', '7', '8', '9']]
