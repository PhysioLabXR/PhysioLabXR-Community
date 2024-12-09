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
from pylsl import StreamInfo, StreamOutlet,StreamInlet, resolve_stream ,cf_float32, LostError
import torch
from physiolabxr.scripting.illumiRead.illumiReadSwype import illumiReadSwypeConfig
from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.gaze2word import Gaze2Word
from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.Tap2Char import Tap2Char
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeConfig import EventMarkerLSLStreamInfo, \
    GazeDataLSLStreamInfo, UserInputLSLStreamInfo
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeUtils import illumiReadSwypeUserInput,\
    word_candidate_list_to_lvt
from physiolabxr.scripting.illumiRead.utils.VarjoEyeTrackingUtils.VarjoGazeUtils import VarjoGazeData
from physiolabxr.scripting.illumiRead.utils.gaze_utils.general import GazeFilterFixationDetectionIVT, GazeType
from physiolabxr.scripting.illumiRead.utils.language_utils.neuspell_utils import SpellCorrector
from physiolabxr.utils.buffers import DataBuffer

import csv
import pandas as pd
from nltk import RegexpTokenizer
from physiolabxr.rpc.decorator import rpc, async_rpc


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
        # self.spell_corrector = SpellCorrector()
        # self.spell_corrector.correct_string("WHAT")

        # create stream outlets
        illumireadswype_keyboard_suggestion_strip_lsl_stream_info = StreamInfo(
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.StreamName,
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.StreamType,
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.ChannelNum,
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.NominalSamplingRate,
            channel_format=cf_float32)

        self.illumireadswype_keyboard_suggestion_strip_lsl_outlet = StreamOutlet(illumireadswype_keyboard_suggestion_strip_lsl_stream_info)  # shape: (1024, 1)

        # create gaze2word object
        gaze_data_path = r'C:\Users\Season\Documents\PhysioLab\physiolabxr\scripting\illumiRead\illumiReadSwype\gaze2word\GazeData.csv'

        # load from pickle if exists
        if os.path.exists('g2w.pkl'):
            with open('g2w.pkl', 'rb') as f:
                self.g2w = pickle.load(f)
        else:
            print("Instantiating g2w...")
            self.g2w = Gaze2Word(gaze_data_path)
            with open('g2w.pkl', 'wb') as f:
                pickle.dump(self.g2w, f)
            print("Finished instantiating g2w")
        
        # trim the vocab for g2w
        file_path = self.params['trial_sentences']
        df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
        
        sentences = df.iloc[:, 0].tolist() + df.iloc[:, 1].tolist()
        sentences = [s for s in sentences if isinstance(s, str)]
        tokenizer = RegexpTokenizer(r'\w+')
        words = [tokenizer.tokenize(s) for s in sentences]
        words = [word for sublist in words for word in sublist]  # flatten the list
        self.g2w.trim_vocab(words)
        
        # t2c definition
        if os.path.exists('t2c.pkl'):
            with open('t2c.pkl', 'rb') as f:
                self.t2c = pickle.load(f)
        else:
            print("Instantiating t2c...")
            self.t2c = Tap2Char(gaze_data_path)
            with open('t2c.pkl', 'wb') as f:
                pickle.dump(self.t2c, f)
            print("Finished instantiating t2c")

        # the current context for the inputfield
        self.context = ""
        self.nextChar = ""
    
    #  ----------------- RPC START-----------------------------------------------------------------
    # get the rpc call of word context from unity
    @async_rpc
    def ContextRPC(self, input0: str) -> str:
        
        self.context = input0.lower().rstrip()
        
        return f"Sucess:{self.context}"
    
    # the rpc for Tap2Char prediction
    # input: float of x and y position of the tap
    @async_rpc
    def Tap2CharRPC(self, input0: float, input1:float) -> str:
        # merge the x and y input to a ndarray
        tap_position = np.array([input0, input1])
        result = self.t2c.predict(tap_position, self.context)
        
        highest_prob_char = str(result[0][0])
        
        return highest_prob_char
    
    @async_rpc
    def SwypePredictRPC(self) -> str:
        highest_prob_word = self._swype_predict()
        return highest_prob_word
        
        
    
    # ----------------- RPC END--------------------------------------------------------------------

    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        
        if (EventMarkerLSLStreamInfo.StreamName not in self.inputs.keys()) or (
                GazeDataLSLStreamInfo.StreamName not in self.inputs.keys()) or (
                UserInputLSLStreamInfo.StreamName not in self.inputs.keys()) :  # or GazeDataLSLOutlet.StreamName not in self.inputs.keys():
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

    # NEW
    def _swype_predict(self):
        grouped_list = [
            list(group) for key, group in
            groupby(self.gaze_data_sequence, key=lambda x: x.gaze_type)
        ]
        fixation_character_sequence = []
        # gaze_trace = []
        fixation_trace = []

        for group in grouped_list:
            # if the gaze is a fixation
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
                for user_input in user_input_sequence:
                    # gaze trace under fixation mode
                    if not np.array_equal(user_input.keyboard_background_hit_point_local, [1,1,1]):
                        fixation_trace.append(user_input.keyboard_background_hit_point_local[:2])

                # TODO: decoding method

                # fixation_start_user_input_index = np.searchsorted(user_input_timestamp, [fixation_start_time], side='right')
                # fixation_end_user_input_index = np.searchsorted(user_input_timestamp, [fixation_end_time], side='left')

                pass

        # reset
        self.illumiReadSwyping = False
        self.gaze_data_sequence = []
        self.data_buffer = DataBuffer()
        self.ivt_filter.reset_data_processor()
        
        # the fixation trace length should be greater than 1
        print(fixation_trace)
        if len(fixation_trace) >= 1:
            
            # use the trimmed vocab to predict g2w
            fixation_trace = np.array(fixation_trace)
            
            # predict the candidate words
            cadidate_list = self.g2w.predict(4,fixation_trace,run_dbscan=True,prefix = self.context, verbose=True, filter_by_starting_letter=0.35, use_trimmed_vocab=True, njobs=16)
            word_candidate_list = [item[0] for item in cadidate_list]
            
            word_candidate_list = np.array(word_candidate_list).flatten().tolist()
            print(word_candidate_list)

            # send the top n words to the feedback state
            lvt, overflow_flat = word_candidate_list_to_lvt(word_candidate_list)

            return lvt

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
                # gaze_trace = []
                fixation_trace = []

                for group in grouped_list:
                    # if the gaze is a fixation
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
                        for user_input in user_input_sequence:
                            # gaze trace under fixation mode
                            if not np.array_equal(user_input.keyboard_background_hit_point_local, [1,1,1]):
                                fixation_trace.append(user_input.keyboard_background_hit_point_local[:2])

                        # TODO: decoding method

                        # fixation_start_user_input_index = np.searchsorted(user_input_timestamp, [fixation_start_time], side='right')
                        # fixation_end_user_input_index = np.searchsorted(user_input_timestamp, [fixation_end_time], side='left')

                        pass

                # reset
                self.illumiReadSwyping = False
                self.gaze_data_sequence = []
                self.data_buffer = DataBuffer()
                self.ivt_filter.reset_data_processor()
                
                # the fixation trace length should be greater than 1
                print(fixation_trace)
                if len(fixation_trace) >= 1:
                    
                    # use the trimmed vocab to predict g2w
                    fixation_trace = np.array(fixation_trace)
                    
                    # predict the candidate words
                    cadidate_list = self.g2w.predict(4,fixation_trace,run_dbscan=True,prefix = self.context, verbose=True, filter_by_starting_letter=0.35, use_trimmed_vocab=True, njobs=16)
                    word_candidate_list = [item[0] for item in cadidate_list]
                    
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

            self.data_buffer.update_buffers(self.inputs.buffer)

        # clear the processed user input data and gaze data
        self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)
        self.inputs.clear_stream_buffer_data(UserInputLSLStreamInfo.StreamName)

    def keyboard_freeswitch_state_callback(self):
        print("keyboard free switch state")

        pass
