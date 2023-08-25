import numpy as np
import brainflow

from rena.scripting.Examples.PhysioLabXR_P300Speller_Demo.PhysioLabXRP300SpellerDemoConfig import *
from rena.scripting.RenaScript import RenaScript


class PhysioLabXRGameP300SpellerOpenBCICython8Script(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        # test network
        self.EXPERIMENT_STATE = None



    # Start will be called once when the run button is hit.
    def init(self):
        print('Init function is called')
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        print('Loop function is called')
        if EEG_STREAM_NAME not in self.inputs.keys() or EVNET_MARKER_CHANNEL_NAME not in self.inputs.keys():
            # if no event marker or no eeg stream, we do not do anything
            print('No EEG stream or no event marker stream, return')
            return

        print(self.inputs)

        # if EventMarkerChannelInfo.StateInterruptMarker in self.inputs.get_data(Event_Marker_Stream_Name):
        #     # if the state is interrupted, we do not do anything
        #     return



    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
