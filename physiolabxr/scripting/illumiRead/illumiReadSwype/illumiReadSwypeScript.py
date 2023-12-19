import time
from collections import deque
import cv2
import numpy as np
import time
import os
import pickle
import sys
import matplotlib.pyplot as plt
from itertools import groupby
from physiolabxr.scripting.RenaScript import RenaScript
from pylsl import StreamInfo, StreamOutlet, cf_float32
import torch
from physiolabxr.scripting.illumiRead.illumiReadSwype import illumiReadSwypeConfig
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeConfig import EventMarkerLSLStreamInfo, \
    GazeDataLSLStreamInfo, UserInputLSLStreamInfo
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeUtils import illumiReadSwypeUserInput, \
    word_candidate_list_to_lvt
from physiolabxr.scripting.illumiRead.utils.VarjoEyeTrackingUtils.VarjoGazeUtils import VarjoGazeData
from physiolabxr.scripting.illumiRead.utils.gaze_utils.general import GazeFilterFixationDetectionIVT, GazeType
from physiolabxr.scripting.illumiRead.utils.language_utils.neuspell_utils import SpellCorrector
from physiolabxr.utils.buffers import DataBuffer


class IllumiReadSwypeScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

        self.currentExperimentState: illumiReadSwypeConfig.ExperimentState = \
            illumiReadSwypeConfig.ExperimentState.InitState

        self.currentBlock: illumiReadSwypeConfig.ExperimentBlock = \
            illumiReadSwypeConfig.ExperimentBlock.InitBlock

        self.illumiReadSwyping = False

        self.data_buffer = DataBuffer()
        self.gaze_data_sequence = list()
        self.user_input_sequence = list()

        self.device = torch.device('cpu')

        self.current_image_name = None

        self.process_gaze_data_time_buffer = deque(maxlen=1000)

        self.ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)

        # spelling correction
        self.spell_corrector = SpellCorrector()
        self.spell_corrector.correct_string("WHAT")

        # create stream outlets
        illumireadswype_keyboard_suggestion_strip_lsl_stream_info = StreamInfo(
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.StreamName,
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.StreamType,
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.ChannelNum,
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.NominalSamplingRate,
            channel_format=cf_float32)

        self.illumireadswype_keyboard_suggestion_strip_lsl_outlet = StreamOutlet(illumireadswype_keyboard_suggestion_strip_lsl_stream_info)  # shape: (1024, 1)


    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):

        if (EventMarkerLSLStreamInfo.StreamName not in self.inputs.keys()) or (
                GazeDataLSLStreamInfo.StreamName not in self.inputs.keys()) or (
                UserInputLSLStreamInfo.StreamName not in self.inputs.keys()):  # or GazeDataLSLOutlet.StreamName not in self.inputs.keys():
            return
        # print("process event marker call start")
        self.process_event_markers()

        # self.process_gaze_data()

        # gaze callback

        self.state_callbacks()

        # if self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardDewellTimeState or \
        #         self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardClickState or \
        #         self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState or \
        #         self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchState:
        # self.process_gaze_data()

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
                    # clear the gaze data buffer before decoding the next state
                    self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)

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

            # print("process event marker call end")

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

    def state_callbacks(self):
        if self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardClickState:
            self.keyboard_click_state_callback()
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardDewellTimeState:
            self.keyboard_dewelltime_state_callback()
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState:
            self.keyboard_illumireadswype_state_callback()
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchState:
            self.keyboard_freeswitch_state_callback()

    def keyboard_click_state_callback(self):
        # print("keyboard click state")

        pass

    def keyboard_dewelltime_state_callback(self):
        # print("keyboard dewell time state")

        pass

    def keyboard_illumireadswype_state_callback(self):
        # print("keyboard illumiread swype state")

        # get the user input data
        for user_input_data_t in self.inputs[UserInputLSLStreamInfo.StreamName][0].T:
            user_input_button_2 = user_input_data_t[
                illumiReadSwypeConfig.UserInputLSLStreamInfo.UserInputButton2ChannelIndex]  # swyping invoker

            if not self.illumiReadSwyping and user_input_button_2:
                # start swyping
                self.illumiReadSwyping = True
                print("start swyping")
                # start

            if self.illumiReadSwyping and not user_input_button_2:
                # end swyping

                print("start decoding swype path")

                # merge fixations

                grouped_list = [list(group) for key, group in
                                groupby(self.gaze_data_sequence, key=lambda x: x.gaze_type)]

                # TODO: merge fixations that are too close to each other

                # TODO: discard fixations that are too short < 90ms

                # now, we map the fixations to the keyboard keys
                # user_input_data = self.data_buffer[UserInputLSLStreamInfo.StreamName][0]
                # user_input_timestamp = self.data_buffer[UserInputLSLStreamInfo.StreamName][1]

                fixation_character_sequence = []

                for group in grouped_list:
                    if group[0].gaze_type == GazeType.FIXATION:

                        fixation_start_time = group[0].timestamp
                        fixation_end_time = group[-1].timestamp

                        # find the user input data that is closest to the fixation
                        user_input_during_fixation_data, user_input_during_fixation_timestamps = self.data_buffer.get_stream_in_time_range(
                            UserInputLSLStreamInfo.StreamName, fixation_start_time, fixation_end_time)

                        user_input_sequence = []
                        for user_input, timestamp in zip(user_input_during_fixation_data.T,
                                                         user_input_during_fixation_timestamps):
                            user_input_sequence.append(illumiReadSwypeUserInput(user_input, timestamp))

                        # check if the fixation consists of only one user input
                        user_input_dict = {}
                        for user_input in user_input_sequence:
                            if user_input.key_hit_index in user_input_dict:
                                user_input_dict[user_input.key_hit_index] += 1
                            else:
                                user_input_dict[user_input.key_hit_index] = 1

                        # find the most frequent user input
                        if len(user_input_dict) > 0:
                            most_frequent_user_input_index = max(user_input_dict, key=user_input_dict.get)
                            print("most frequent user input index: ", most_frequent_user_input_index)
                            user_input = illumiReadSwypeConfig.KeyIDIndexDict[most_frequent_user_input_index]
                            print("most frequent user input: ", user_input)
                            fixation_character_sequence.append(user_input)

                        else:
                            print("no user input detected")

                        # update the character sequence

                        # TODO: decoding method

                        # fixation_start_user_input_index = np.searchsorted(user_input_timestamp, [fixation_start_time], side='right')
                        # fixation_end_user_input_index = np.searchsorted(user_input_timestamp, [fixation_end_time], side='left')

                        pass

                # reset
                self.illumiReadSwyping = False
                self.gaze_data_sequence = []
                self.data_buffer = DataBuffer()
                self.ivt_filter.reset_data_processor()

                # send the character sequence to the keyboard
                print(fixation_character_sequence)
                # use the spell correction algorithm to correct the character sequence

                # word_candidate_list = ["hello", "world", "how", "are"]
                #
                # lvt, overflow_flat = word_candidate_list_to_lvt(word_candidate_list)
                # self.illumireadswype_keyboard_suggestion_strip_lsl_outlet.push_sample(lvt)

                # ['W', 'H', 'W', 'R', 'S', 'Y', 'E']

                # remove the None character
                fixation_character_sequence = [x for x in fixation_character_sequence if x is not None]



                fixation_character_string = "".join(fixation_character_sequence).lower()
                if len(fixation_character_string) > 0:

                    word_candidate_list = self.spell_corrector.correct_string(fixation_character_string, 4) # the output is a list of list
                    word_candidate_list = np.array(word_candidate_list).flatten().tolist()
                    print(word_candidate_list)

                    # send the top n words to the feedback state
                    lvt, overflow_flat = word_candidate_list_to_lvt(word_candidate_list)

                    self.illumireadswype_keyboard_suggestion_strip_lsl_outlet.push_sample(lvt)

        if self.illumiReadSwyping:
            # print("update_buffer")
            # swyping, save the gaze data and user input data
            # self.data_buffer.update_buffer(

            for gaze_data_t, timestamp in (
                    zip(self.inputs[GazeDataLSLStreamInfo.StreamName][0].T,
                        self.inputs[GazeDataLSLStreamInfo.StreamName][1])):
                gaze_data = VarjoGazeData()
                gaze_data.construct_gaze_data_varjo(gaze_data_t, timestamp)
                gaze_data = self.ivt_filter.process_sample(gaze_data)
                self.gaze_data_sequence.append(gaze_data)

            # for user_input_t, timestamp in (
            #         zip(self.inputs[UserInputLSLStreamInfo.StreamName][0].T,
            #             self.inputs[UserInputLSLStreamInfo.StreamName][1])):
            #     gaze_hit_keyboard_background = user_input_data_t[
            #         illumiReadSwypeConfig.UserInputLSLStreamInfo.GazeHitKeyboardBackgroundChannelIndex]
            #     keyboard_background_hit_point_local = [
            #         user_input_data_t[
            #             illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyboardBackgroundHitPointLocalXChannelIndex],
            #         user_input_data_t[
            #             illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyboardBackgroundHitPointLocalYChannelIndex],
            #         user_input_data_t[
            #             illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyboardBackgroundHitPointLocalZChannelIndex]
            #     ]
            #
            #     gaze_hit_key = user_input_data_t[illumiReadSwypeConfig.UserInputLSLStreamInfo.GazeHitKeyChannelIndex]
            #     key_hit_point_local = [
            #         user_input_data_t[illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyHitPointLocalXChannelIndex],
            #         user_input_data_t[illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyHitPointLocalYChannelIndex],
            #         user_input_data_t[illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyHitPointLocalZChannelIndex]
            #     ]
            #
            #     key_hit_index = user_input_data_t[illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyHitIndexChannelIndex]
            #
            #     user_input_button_1 = user_input_data_t[
            #         illumiReadSwypeConfig.UserInputLSLStreamInfo.UserInputButton1ChannelIndex]  # swyping invoker
            #     user_input_button_2 = user_input_data_t[
            #         illumiReadSwypeConfig.UserInputLSLStreamInfo.UserInputButton2ChannelIndex]
            #
            #     user_input = illumiReadSwypeUserInput(
            #         gaze_hit_keyboard_background,
            #         keyboard_background_hit_point_local,
            #         gaze_hit_key,
            #         key_hit_point_local,
            #         key_hit_index,
            #         user_input_button_1,
            #         user_input_button_2,
            #         timestamp
            #     )
            #
            #     self.user_input_sequence.append(user_input)
            #

            self.data_buffer.update_buffers(self.inputs.buffer)

        # clear the processed user input data and gaze data
        self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)
        self.inputs.clear_stream_buffer_data(UserInputLSLStreamInfo.StreamName)

    def keyboard_freeswitch_state_callback(self):
        print("keyboard free switch state")

        pass

    # def process_gaze_data(self):
    #
    #     for gaze_data_t in self.inputs[GazeDataLSLStreamInfo.StreamName][0].T:
    #
    #         gaze_data = VarjoGazeData()
    #         gaze_data.construct_gaze_data_varjo(gaze_data_t)
    #         gaze_data = self.ivt_filter.process_sample(gaze_data)
    #         print(gaze_data.get_gaze_type())
    #
    #
    #     self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)
