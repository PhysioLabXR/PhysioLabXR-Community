import numpy as np
import brainflow
from sklearn.linear_model import LogisticRegression

from rena.scripting.Examples.PhysioLabXR_P300Speller_Demo.PhysioLabXRP300SpellerDemoConfig import *
from rena.scripting.RenaScript import RenaScript
from rena.utils.buffers import DataBuffer


class PhysioLabXRGameP300SpellerDemoScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        # test network
        self.EXPERIMENT_STATE = ExperimentStateMarker.StartState
        self.IN_FLASHING_BLOCK = False
        self.model = LogisticRegression()
        self.data_buffer = DataBuffer()

        self.train_state_x = []
        self.train_state_y = []

        self.test_state_x = []
        self.test_state_y = []

    # Start will be called once when the run button is hit.
    def init(self):
        print('Init function is called')
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        # print('Loop function is called')
        if EEG_STREAM_NAME not in self.inputs.keys() or EVENT_MARKER_CHANNEL_NAME not in self.inputs.keys():
            # if no event marker or no eeg stream, we do not do anything
            print('No EEG stream or no event marker stream, return')
            # state is None, and Flashing is False. We interrupt the experiment
            self.EXPERIMENT_STATE = None
            self.IN_FLASHING_BLOCK = False
            return

        event_marker_data = self.inputs.get_data(EVENT_MARKER_CHANNEL_NAME)
        event_marker_timestamps = self.inputs.get_timestamps(EVENT_MARKER_CHANNEL_NAME)

        # in this example, we only care about the Train, Test, Interrupt, and Block Start/Block end markers
        # process event markers
        # try:
        for event_index, event_marker_timestamp in enumerate(event_marker_timestamps):
            event_marker = event_marker_data[:, event_index]

            StateEnterExitMarker = event_marker[EventMarkerChannelInfo.StateEnterExitMarker]
            FlashBlockStartEndMarker = event_marker[EventMarkerChannelInfo.FlashBlockStartEndMarker]
            FlashingMarker = event_marker[EventMarkerChannelInfo.FlashingMarker]
            FlashingItemIndexMarker = event_marker[EventMarkerChannelInfo.FlashingItemIndexMarker]
            FlashingTargetMarker = event_marker[EventMarkerChannelInfo.FlashingTargetMarker]
            StateInterruptMarker = event_marker[EventMarkerChannelInfo.StateInterruptMarker]

            if StateInterruptMarker:
                # state is None, and Flashing is False. We interrupt the experiment
                self.EXPERIMENT_STATE = None
                self.IN_FLASHING_BLOCK = False

            elif StateEnterExitMarker:
                self.switch_state(StateEnterExitMarker)

            elif FlashBlockStartEndMarker:
                print('Block Start/End Marker: ', FlashBlockStartEndMarker)

                if FlashBlockStartEndMarker == 1:  # flash start
                    self.IN_FLASHING_BLOCK = True
                    print('Start Flashing Block')
                    self.inputs.clear_up_to(event_marker_timestamp)
                    # self.data_buffer.update_buffers(self.inputs.buffer)
                if FlashBlockStartEndMarker == -1:  # flash end
                    self.IN_FLASHING_BLOCK = False
                    print('End Flashing Block')
                    if self.EXPERIMENT_STATE == ExperimentStateMarker.TrainState:
                        # train callback
                        self.train_callback()
                        pass
                    elif self.EXPERIMENT_STATE == ExperimentStateMarker.TestState:
                        # test callback
                        self.test_callback()
                        pass
            elif FlashingMarker:  # flashing
                print("Flashing")
                print('Flashing Marker: ', FlashingMarker)
                print('Flashing Target Marker: ', FlashingTargetMarker)
                print('Flashing Item Index Marker: ', FlashingItemIndexMarker)
            else:
                pass
        # except Exception as e:
        #     print(e)
        #     return

        # if flashing
        if self.IN_FLASHING_BLOCK:
            # the event marker in the buffer only contains the event marker in the current flashing block
            self.data_buffer.update_buffers(self.inputs.buffer)
            print('In Flashing Block, save data to buffer')

        self.inputs.clear_buffer_data()


    def switch_state(self, new_state):
        if new_state == ExperimentStateMarker.StartState:
            print('Start State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.StartState

        elif new_state == ExperimentStateMarker.TrainIntroductionState:
            print('Train Introduction State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.TrainIntroductionState

        elif new_state == ExperimentStateMarker.TrainState:
            print('Train State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.TrainState

        elif new_state == ExperimentStateMarker.TestIntroductionState:
            print('Test Introduction State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.TestIntroductionState

        elif new_state == ExperimentStateMarker.TestState:
            print('Test State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.TestState

        elif new_state == ExperimentStateMarker.EndState:
            print('End State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.EndState

        else:
            print('Exit Current State: ', new_state)
            self.EXPERIMENT_STATE = None

    def train_callback(self):
        # train callback

        pass

    def test_callback(self):
        # test callback

        pass

    # cleanup is called when the stop button is hit
    def cleanup(self):
        self.model = None
        print('Cleanup function is called')
