import numpy as np

from physiolabxr.scripting.RenaScript import RenaScript

from physiolabxr.utils.buffers import DataBuffer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics
from enum import Enum

# configurations
eeg_stream_name = 'UnicornHybridBlackBluetooth'
eeg_channel_names = [
    "EEG1", "EEG2", "EEG3", "EEG4", "EEG5", "EEG6", "EEG7", "EEG8",
    "Accelerometer X", "Accelerometer Y", "Accelerometer Z",
    "Gyroscope X", "Gyroscope Y", "Gyroscope Z",
    "Battery Level",
    "Counter",
    "Validation Indicator",
    "Timestamp",
    "Marker"
]

eeg_channel_index = [0, 1, 2, 3, 4, 5, 6, 7]

event_marker_stream_name = 'CortexTypeP300SpellerEventMarkerLSL'


class IndexClass(int, Enum):
    pass


class EventMarkerChannelInfo(IndexClass):
    FlashingBlockMarker = 0
    FlashingMarker = 1
    FlashingRowOrColumnMarker = 2
    FlashingRowOrColumnIndexMarker = 3
    FlashingTargetMarker = 4


class CortexType(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        self.model = LogisticRegression()

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):

        # get all the flashing events



        print('Loop function is called')






    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    # RPC Calls

    def process_train_trail_epochs(self):



        self.inputs.clear_buffer_data()
        pass

    def train_epochs(self):
        self.inputs.clear_buffer_data()
        pass

    def predict(self):
        self.inputs.clear_buffer_data()
        pass
