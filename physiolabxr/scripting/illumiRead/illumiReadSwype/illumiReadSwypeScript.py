import time
from collections import deque
import cv2
import numpy as np
import time
import os
import pickle
import sys
import matplotlib.pyplot as plt
from physiolabxr.scripting.RenaScript import RenaScript
from pylsl import StreamInfo, StreamOutlet, cf_float32
import torch
from physiolabxr.scripting.illumiRead.illumiReadSwype import illumiReadSwypeConfig
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeConfig import EventMarkerLSLStreamInfo, \
    GazeDataLSLStreamInfo


class AOIAugmentationScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

        self.currentExperimentState: illumiReadSwypeConfig.ExperimentState = \
            illumiReadSwypeConfig.ExperimentState.InitState

        self.currentBlock: illumiReadSwypeConfig.ExperimentBlock = \
            illumiReadSwypeConfig.ExperimentBlock.InitBlock

        self.device = torch.device('cpu')

        self.current_image_name = None

        self.process_gaze_data_time_buffer = deque(maxlen=1000)

    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):

        if (EventMarkerLSLStreamInfo.StreamName not in self.inputs.keys()) or (
                GazeDataLSLStreamInfo.StreamName not in self.inputs.keys()):  # or GazeDataLSLOutlet.StreamName not in self.inputs.keys():
            return
        # print("process event marker call start")
        self.process_event_markers()

    def cleanup(self):
        print('Cleanup function is called')

    def process_event_markers(self):
        event_markers = self.inputs[EventMarkerLSLStreamInfo.StreamName][0]
        self.inputs.clear_stream_buffer_data(EventMarkerLSLStreamInfo.StreamName)

        # state shift
        for event_marker in event_markers.T:
            # print(f"working on event marker {event_marker}")
            block_marker = event_marker[illumiReadSwypeConfig.EventMarkerLSLStreamInfo.BlockChannelIndex]
            state_marker = event_marker[illumiReadSwypeConfig.EventMarkerLSLStreamInfo.ExperimentStateChannelIndex]
            user_inputs_marker = event_marker[illumiReadSwypeConfig.EventMarkerLSLStreamInfo.UserInputsChannelIndex]

            # ignore the block_marker <0 and state_marker <0 those means exit the current state
            if block_marker:
                if block_marker > 0:
                    self.enter_block(block_marker)
                    print(self.currentBlock)
                else:
                    self.exit_block(block_marker)
                    print("Exit Current Block")

            if state_marker:
                if state_marker > 0:
                    self.enter_state(state_marker)
                    print(self.currentExperimentState)
                else:
                    self.exit_state(state_marker)
                    print("Exit Current State")

            # if state_marker and state_marker > 0:  # evoke state change
            #     self.enter_state(state_marker)
            #     print(self.currentExperimentState)
            #     if state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardDewellTimeState or \
            #             state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardClickState or \
            #             state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState or \
            #             state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchState:
            #
            #         print("state change")

            print("process event marker call end")

    def no_aoi_augmentation_state_init_callback(self):
        pass

    def enter_block(self, block_marker):

        if block_marker == illumiReadSwypeConfig.ExperimentBlock.StartBlock.value:
            self.currentBlock = illumiReadSwypeConfig.ExperimentBlock.StartBlock
        elif block_marker == illumiReadSwypeConfig.ExperimentBlock.IntroductionBlock.value:
            self.currentBlock = illumiReadSwypeConfig.ExperimentBlock.IntroductionBlock
        elif block_marker == illumiReadSwypeConfig.ExperimentBlock.PracticeBlock.value:
            self.currentBlock = illumiReadSwypeConfig.ExperimentBlock.PracticeBlock
        elif block_marker == illumiReadSwypeConfig.ExperimentBlock.TrainBlock.value:
            self.currentBlock = illumiReadSwypeConfig.ExperimentBlock.TrainBlock
        elif block_marker == illumiReadSwypeConfig.ExperimentBlock.TestBlock.value:
            self.currentBlock = illumiReadSwypeConfig.ExperimentBlock.TestBlock
        elif block_marker == illumiReadSwypeConfig.ExperimentBlock.EndBlock.value:
            self.currentBlock = illumiReadSwypeConfig.ExperimentBlock.EndBlock
        else:
            print("Invalid block marker")

    def exit_block(self, block_marker):
        if block_marker == -illumiReadSwypeConfig.ExperimentBlock.StartBlock.value:
            self.currentBlock = None
        elif block_marker == -illumiReadSwypeConfig.ExperimentBlock.IntroductionBlock.value:
            self.currentBlock = None
        elif block_marker == -illumiReadSwypeConfig.ExperimentBlock.PracticeBlock.value:
            self.currentBlock = None
        elif block_marker == -illumiReadSwypeConfig.ExperimentBlock.TrainBlock.value:
            self.currentBlock = None
        elif block_marker == -illumiReadSwypeConfig.ExperimentBlock.TestBlock.value:
            self.currentBlock = None
        elif block_marker == -illumiReadSwypeConfig.ExperimentBlock.EndBlock.value:
            self.currentBlock = None
        else:
            print("Invalid block marker")

    def enter_state(self, state_marker):

        if state_marker == illumiReadSwypeConfig.ExperimentState.InitState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.InitState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.CalibrationState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.CalibrationState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.StartState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.StartState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.IntroductionInstructionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.IntroductionInstructionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardDewellTimeIntroductionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardDewellTimeIntroductionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardDewellTimeState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardDewellTimeState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardClickIntroductionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardClickIntroductionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardClickState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardClickState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeIntroductionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeIntroductionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchInstructionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchInstructionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.FeedbackState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.FeedbackState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.EndState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.EndState

    def exit_state(self, state_marker):
        if state_marker == -illumiReadSwypeConfig.ExperimentState.InitState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.CalibrationState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.StartState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.IntroductionInstructionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardDewellTimeIntroductionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardDewellTimeState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardClickIntroductionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardClickState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeIntroductionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchInstructionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.FeedbackState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.EndState.value:
            self.currentExperimentState = None

