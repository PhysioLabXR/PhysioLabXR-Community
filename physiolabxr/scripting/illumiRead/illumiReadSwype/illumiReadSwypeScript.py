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


# import sys, traceback
#
# def boom_on_exported_memoryview(unraisable):
#     exc = unraisable.exc_value
#     if isinstance(exc, BufferError) and 'memoryview has' in str(exc):
#         print("\n=== Unraisable BufferError intercepted ===")
#         print("object repr :", repr(unraisable.object))
#         print("traceback   :", exc)          # still prints the BufferError text
#         # Crash right here – the next three lines turn it into a normal traceback
#         raise exc.with_traceback(unraisable.exc_traceback or None)
#     # For anything else, fall back to the stock hook
#     sys.__unraisablehook__(unraisable)
#
# sys.unraisablehook = boom_on_exported_memoryview
# # mv_trace.py  – import **very early** (sitecustomize is perfect)
#
# # mv_leak_watch.py  ——  import this ONCE, as early as you can
# import sys, tracemalloc, textwrap
#
# tracemalloc.start(25)          # keep 25 frames per allocation
#
# def boom_on_memoryview_leak(unraisable):
#     exc = unraisable.exc_value
#     if isinstance(exc, BufferError) and "memoryview has" in str(exc):
#         mv = unraisable.object            # the view the GC is trying to kill
#         print("\n====== GC hit leaked memoryview ======")
#
#         # 1) Where was this view allocated?
#         tb = tracemalloc.get_object_traceback(mv)
#         if tb:
#             print("Allocated at:")
#             for line in tb.format():
#                 # tb.format() already gives "  File '...', line X, in func"
#                 print("  ", line)
#         else:
#             print("*No tracemalloc info (view predates tracemalloc.start())*")
#
#         # 2) What Python stack *still exists* at GC time?
#         print("\nTriggering traceback from inside GC:")
#         raise exc.with_traceback(unraisable.exc_traceback)
#
#     # anything else → let default hook handle it
#     sys.__unraisablehook__(unraisable)
#
# sys.unraisablehook = boom_on_memoryview_leak


from physiolabxr.scripting.RenaScript import RenaScript
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream, cf_float32, LostError
from physiolabxr.scripting.illumiRead.illumiReadSwype import illumiReadSwypeConfig
from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.gaze2word import *
from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.Tap2Char import Tap2Char
from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.ngram import NGramModel
from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.ngram_interpolation import InterpolatedNGramModel
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeConfig import EventMarkerLSLStreamInfo, \
    GazeDataLSLStreamInfo, UserInputLSLStreamInfo
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeUtils import illumiReadSwypeUserInput, \
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
        self.HandSwyping = False

        self.data_buffer = DataBuffer()
        self.gaze_data_sequence = list()
        self.fixation_trace = list()
        self.user_input_sequence = list()

        self.current_image_name = None

        self.process_gaze_data_time_buffer = deque(maxlen=1000)

        self.ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)


        # create stream outlets
        illumireadswype_keyboard_suggestion_strip_lsl_stream_info = StreamInfo(
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.StreamName,
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.StreamType,
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.ChannelNum,
            illumiReadSwypeConfig.illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo.NominalSamplingRate,
            channel_format=cf_float32)

        self.illumireadswype_keyboard_suggestion_strip_lsl_outlet = StreamOutlet(
            illumireadswype_keyboard_suggestion_strip_lsl_stream_info)  # shape: (1024, 1)

        # Get the current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        gaze_data_path = os.path.join(current_dir,'gaze2word', 'GazeData.csv')


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
        self.file_directory = os.path.join(current_dir, 'StudySentences')

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

        # load the NGramModel
        if os.path.exists('ngram_model_interpolate.pkl'):
            self.ngram_model = pickle.load(open('ngram_model_interpolate.pkl', 'rb'))
        else:
            self.ngram_model = InterpolatedNGramModel(top_unigrams=25000)
            pickle.dump(self.ngram_model, open("ngram_model_interpolate.pkl", "wb"))

        # the current context for the inputfield
        self.context = ""
        self.nextChar = ""

    #  ----------------- RPC START-----------------------------------------------------------------
    # get the rpc call of word context from unity
    @async_rpc
    def ContextRPC(self, input0: str) -> str:
        start_time = time.perf_counter()

        self.context = input0.lower().rstrip("?")
        # predict the candidate words
        completions = self.ngram_model.predict_word_completion(self.context, k=5, ignore_punctuation=True)
        word_candidates = ".".join([item[0] for item in completions])
        # print(self.context)
        # print(f"Time spent predicting: {time.perf_counter() - start_time:.8f}s")
        return word_candidates

    # the rpc for Tap2Char prediction
    # input: float of x and y position of the tap
    @async_rpc
    def Tap2CharRPC(self, input0: float, input1: float) -> str:
        # merge the x and y input to a ndarray
        tap_position = np.array([input0, input1])
        result = self.t2c.predict(tap_position, self.context)

        highest_prob_char = str(result[0][0])

        return highest_prob_char

    @async_rpc
    def ExcelLoaderRPC(self, input0:str)-> str:

        user_study,session, practice = input0.split(".")
        user_study = int(user_study)
        session = int(session)
        practice = int(practice)

        if(practice ==1):
            file_path = os.path.join(self.file_directory, 'PracticeSentences.xlsx')
        else:
            if(user_study == 3):
                subfolder = "user_study4_session_sentences"
                file_name = f"session{session}-short-longe-sentences.xlsx"
                file_path = os.path.join(self.file_directory, subfolder, file_name)
            elif(user_study == 2):
                subfolder = "user_study3_session_sentences"
                file_name = f"session{session}_study_3.xlsx"
                file_path = os.path.join(self.file_directory, subfolder, file_name)
            else:
                # if(session == 1):
                #     file_path = os.path.join(self.file_directory, 'session1-short-longe-sentences.xlsx')
                # elif(session == 2):
                #     file_path = os.path.join(self.file_directory, 'session2-short-longe-sentences.xlsx')
                # elif(session == 3):
                #     file_path = os.path.join(self.file_directory, 'session3-short-longe-sentences.xlsx')
                # elif(session == 4):
                #     file_path = os.path.join(self.file_directory, 'session4-short-longe-sentences.xlsx')
                # elif(session == 5):
                #     file_path = os.path.join(self.file_directory, 'session5-short-longe-sentences.xlsx')
                file_name = f"session{session}-short-longe-sentences.xlsx"
                file_path = os.path.join(self.file_directory, file_name)

        if(practice ==1):
            df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
            sentences = df.iloc[:, 0].tolist() + df.iloc[:, 1].tolist()
        else:
            # if not user study 3, then load sentences for column 0 and 1
            if(user_study ==2):
                df = pd.read_excel(file_path, sheet_name='Sheet', header=None)
                sentences = df.iloc[:, 0].tolist()
            elif(user_study ==3):
                df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
                sentences = df.iloc[:, 0].tolist()
            else:
                df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
                sentences = df.iloc[:, 0].tolist() + df.iloc[:, 1].tolist()

        sentences = [s for s in sentences if isinstance(s, str)]
        tokenizer = RegexpTokenizer(r'\w+')
        words = [tokenizer.tokenize(s) for s in sentences]
        words = [word for sublist in words for word in sublist]  # flatten the list
        self.g2w.trim_vocab(words)

        return "Excel Loaded"

    @async_rpc
    def SwypePredictRPC(self)-> str:
        highest_prob_words = self._swype_predict()
        return '.'.join(highest_prob_words)

    # ----------------- RPC END--------------------------------------------------------------------

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


        # gaze callback
        self.state_callbacks()


    def cleanup(self):
        print('Cleanup function is called')

    def process_event_markers(self):
        event_markers = self.inputs[EventMarkerLSLStreamInfo.StreamName][0]
        self.inputs.clear_stream_buffer_data(EventMarkerLSLStreamInfo.StreamName)

        # state shift
        for event_marker in event_markers.T:
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
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardHandTapIntroductionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardHandTapIntroductionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardHandTapState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardHandTapState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardClickIntroductionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardClickIntroductionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.GazePinchState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.GazePinchState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeIntroductionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeIntroductionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchInstructionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchInstructionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardHandSwypeIntroductionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardHandSwypeIntroductionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardHandSwypeState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardHandSwypeState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardPartialSwypeIntroductionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardPartialSwypeIntroductionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardPartialSwypeState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardPartialSwypeState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardGlanceWriterIntroductionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardGlanceWriterIntroductionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardGlanceWriterState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardGlanceWriterState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardDoubleCrossingIntroductionState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardDoubleCrossingIntroductionState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardDoubleCrossingState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardDoubleCrossingState
        # elif state_marker == illumiReadSwypeConfig.ExperimentState.FeedbackState.value:
        #     self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.FeedbackState
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
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardHandTapIntroductionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardHandTapState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardClickIntroductionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.GazePinchState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeIntroductionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchInstructionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchState.value:
            self.currentExperimentState = None
        # elif state_marker == -illumiReadSwypeConfig.ExperimentState.FeedbackState.value:
        #     self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardHandSwypeIntroductionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardHandSwypeState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardPartialSwypeIntroductionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardPartialSwypeState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardGlanceWriterIntroductionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardGlanceWriterState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardDoubleCrossingIntroductionState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardDoubleCrossingState.value:
            self.currentExperimentState = None
        # ------------------------------------------------------------------------------------------------------------------
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.EndState.value:
            self.currentExperimentState = None

    def state_callbacks(self):
        if self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.GazePinchState:
            self.gaze_pinch_state_callback()
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardHandTapState:
            self.keyboard_dewelltime_state_callback()
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState:
            self.keyboard_illumireadswype_state_callback()
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardFreeSwitchState:
            self.keyboard_freeswitch_state_callback()
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardHandSwypeState:
            self.keyboard_handswype_state_callback()
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardPartialSwypeState:
            self.keyboard_partialswype_state_callback()

        # the current 2 new eye swipe technique are now using partial swipe call back
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardGlanceWriterState:
            self.keyboard_glance_writer_state_callback()
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardDoubleCrossingState:
            self.keyboard_glance_writer_state_callback()

    def gaze_pinch_state_callback(self):
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

                self.fixation_trace = []

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
                            if not np.array_equal(user_input.keyboard_background_hit_point_local, [1, 1, 1]):
                                self.fixation_trace.append(user_input.keyboard_background_hit_point_local[:2])

                        pass

                # reset
                self.illumiReadSwyping = False
                self.gaze_data_sequence = []
                self.data_buffer = DataBuffer()
                self.ivt_filter.reset_data_processor()

                # the fixation trace length should be greater than 1
                # print(f'Fixation Trace Length: {len(self.fixation_trace)}')
                if len(self.fixation_trace) >= 1:
                    # use the trimmed vocab to predict g2w
                    temp_fixation_trace = np.array(self.fixation_trace)

                    # predict the candidate words
                    cadidate_list = self.g2w.predict(4, temp_fixation_trace, run_dbscan=True, prefix=self.context,
                                                     verbose=True, filter_by_starting_letter=0.35,
                                                     use_trimmed_vocab=True, njobs=16)
                    word_candidate_list = [item[0] for item in cadidate_list]

                    word_candidate_list = np.array(word_candidate_list).flatten().tolist()
                    print(word_candidate_list)

                    # send the top n words to the feedback state
                    lvt, overflow_flat = word_candidate_list_to_lvt(word_candidate_list)

                    self.illumireadswype_keyboard_suggestion_strip_lsl_outlet.push_sample(lvt)

        if self.illumiReadSwyping:

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

    # hand swype will use the same streaming channel as the illumiReadSwype
    def keyboard_handswype_state_callback(self):

        for user_input_data_t in self.inputs[UserInputLSLStreamInfo.StreamName][0].T:
            user_input_button_2 = user_input_data_t[
                illumiReadSwypeConfig.UserInputLSLStreamInfo.UserInputButton2ChannelIndex]  # swyping invoker

            if not self.illumiReadSwyping and user_input_button_2:
                # start swyping
                self.illumiReadSwyping = True
                print("start swyping")

            if self.illumiReadSwyping and not user_input_button_2:
                # end swyping

                print("start decoding swype path")
                if UserInputLSLStreamInfo.StreamName in self.inputs and len(self.inputs[UserInputLSLStreamInfo.StreamName][0]) > 0:
                    grouped_list = [list(group) for key, group in
                                    groupby(self.gaze_data_sequence, key=lambda x: x.gaze_type)]
                    self.fixation_trace = []
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
                                if not np.array_equal(user_input.keyboard_background_hit_point_local, [1, 1, 1]):
                                    self.fixation_trace.append(user_input.keyboard_background_hit_point_local[:2])
                else:
                    print(f"UserInput is empty, will not decode.")
                # reset
                self.illumiReadSwyping = False
                self.gaze_data_sequence = []
                self.data_buffer = DataBuffer()
                self.ivt_filter.reset_data_processor()

                if len(self.fixation_trace) >= 1:
                    # use the trimmed vocab to predict g2w
                    temp_fixation_trace = np.array(self.fixation_trace)

                    # predict the candidate words
                    cadidate_list = self.g2w.predict(4, temp_fixation_trace, run_dbscan=True, prefix=self.context,
                                                     verbose=False, filter_by_starting_letter=0.35,
                                                     use_trimmed_vocab=True, njobs=16)
                    word_candidate_list = [item[0] for item in cadidate_list]

                    word_candidate_list = np.array(word_candidate_list).flatten().tolist()
                    print(word_candidate_list)

                    # send the top n words to the feedback state
                    lvt, overflow_flat = word_candidate_list_to_lvt(word_candidate_list)

                    self.illumireadswype_keyboard_suggestion_strip_lsl_outlet.push_sample(lvt)

        if self.illumiReadSwyping:

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

    def keyboard_partialswype_state_callback(self):

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
                grouped_list = [list(group) for key, group in
                                groupby(self.gaze_data_sequence, key=lambda x: x.gaze_type)]

                end_fixation_trace = []

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
                            if not np.array_equal(user_input.keyboard_background_hit_point_local, [1, 1, 1]):
                                end_fixation_trace.append(user_input.keyboard_background_hit_point_local[:2])

                        pass

                # reset and start decoding
                print("start decoding swype path")

                self.illumiReadSwyping = False
                self.gaze_data_sequence = []
                self.data_buffer = DataBuffer()
                self.ivt_filter.reset_data_processor()

                if len(end_fixation_trace) >= 1:
                    # use the trimmed vocab to predict g2w
                    temp_fixation_trace = np.array(end_fixation_trace)

                    # predict the candidate words
                    cadidate_list = self.g2w.predict(4, temp_fixation_trace, run_dbscan=True, prefix=self.context,
                                                     verbose=True, filter_by_starting_letter=0.35,
                                                     use_trimmed_vocab=True, njobs=16)
                    word_candidate_list = [item[0] for item in cadidate_list]

                    word_candidate_list = np.array(word_candidate_list).flatten().tolist()
                    print(word_candidate_list)

                    # send the top n words to the feedback state
                    lvt, overflow_flat = word_candidate_list_to_lvt(word_candidate_list)

                    self.illumireadswype_keyboard_suggestion_strip_lsl_outlet.push_sample(lvt)


        if self.illumiReadSwyping:

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

    def keyboard_glance_writer_state_callback(self):

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
                end_fixation_trace = []
                # end swyping
                # the sweyepe way: detecting fixation and only use fixation for decoding ------------------------
                # grouped_list = [list(group) for key, group in
                #                 groupby(self.gaze_data_sequence, key=lambda x: x.gaze_type)]
                # for group in grouped_list:
                #     # if the gaze is a fixation
                #     if group[0].gaze_type == GazeType.FIXATION:
                #
                #         fixation_start_time = group[0].timestamp
                #         fixation_end_time = group[-1].timestamp
                #
                #         # find the user input data that is closest to the fixation
                #         user_input_during_fixation_data, user_input_during_fixation_timestamps = self.data_buffer.get_stream_in_time_range(
                #             UserInputLSLStreamInfo.StreamName, fixation_start_time, fixation_end_time)
                #
                #         user_input_sequence = []
                #         for user_input, timestamp in zip(user_input_during_fixation_data.T,
                #                                          user_input_during_fixation_timestamps):
                #             user_input_sequence.append(illumiReadSwypeUserInput(user_input, timestamp))
                #
                #         # check if the fixation consists of only one user input
                #         for user_input in user_input_sequence:
                #             # gaze trace under fixation mode
                #             if not np.array_equal(user_input.keyboard_background_hit_point_local, [1, 1, 1]):
                #                 end_fixation_trace.append(user_input.keyboard_background_hit_point_local[:2])
                # ------------------------------------------------------------------------------------------------
                if UserInputLSLStreamInfo.StreamName in self.inputs.buffer and len(self.inputs[UserInputLSLStreamInfo.StreamName][0]) > 0:
                    if len(self.gaze_data_sequence) > 0:
                        user_input_sequence, user_input_sequence_ts = self.data_buffer.get_stream_in_time_range(
                            UserInputLSLStreamInfo.StreamName,
                            self.gaze_data_sequence[0].get_timestamp(),
                            self.gaze_data_sequence[-1].get_timestamp())
                        for user_input, ts in zip(user_input_sequence.T, user_input_sequence_ts):
                            user_input_struct = illumiReadSwypeUserInput(user_input, ts)
                            if not np.array_equal(user_input_struct.keyboard_background_hit_point_local, [1, 1, 1]):
                                end_fixation_trace.append(user_input_struct.keyboard_background_hit_point_local[:2])

                    # for gaze_point in self.gaze_data_sequence:
                    #     # find the user input data that is closest to the fixation
                    #     ts = gaze_point.get_timestamp()
                    #     user_input_seq_idx = find_closes_time_index(user_input_sequence[1], ts, return_diff=False)

                    # reset and start decoding
                    print("start decoding swype path")

                    self.illumiReadSwyping = False
                    self.data_buffer = DataBuffer()
                    self.ivt_filter.reset_data_processor()

                    if len(end_fixation_trace) >= 1:
                        # use the trimmed vocab to predict g2w
                        temp_fixation_trace = np.array(end_fixation_trace)

                        # predict the candidate words
                        # cadidate_list = self.g2w.predict(4, temp_fixation_trace, run_dbscan=True, prefix=self.context,
                        #                                  verbose=True, filter_by_starting_letter=0.35,
                        #                                  use_trimmed_vocab=True, njobs=16)

                        print("Predicting glance writer")

                        candidate_list = self.g2w.predict_glancewriter(4,temp_fixation_trace,prefix = self.context)

                        word_candidate_list = [item[0] for item in candidate_list]

                        word_candidate_list = np.array(word_candidate_list).flatten().tolist()
                        print(word_candidate_list)

                        # send the top n words to the feedback state
                        lvt, overflow_flat = word_candidate_list_to_lvt(word_candidate_list)

                        self.illumireadswype_keyboard_suggestion_strip_lsl_outlet.push_sample(lvt)
                else:
                    print("UserInput is empty, will not decode.")
                self.gaze_data_sequence = []

        if self.illumiReadSwyping:

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

    def _swype_predict(self):
        grouped_list = [
            list(group) for key, group in
            groupby(self.gaze_data_sequence, key=lambda x: x.gaze_type)
        ]

        self.fixation_trace = []
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
                    if not np.array_equal(user_input.keyboard_background_hit_point_local, [1, 1, 1]):
                        self.fixation_trace.append(user_input.keyboard_background_hit_point_local[:2])
                pass
        # reset
        # self.illumiReadSwyping = False
        # self.gaze_data_sequence = []
        # self.data_buffer = DataBuffer()
        # self.ivt_filter.reset_data_processor()


        if len(self.fixation_trace) >= 1:
            # use the trimmed vocab to predict g2w
            temp_fixation_trace = np.array(self.fixation_trace)

            # predict the candidate words
            cadidate_list = self.g2w.predict(4, temp_fixation_trace, run_dbscan=True, prefix=self.context, verbose=True,
                                             filter_by_starting_letter=0.35, use_trimmed_vocab=True, njobs=16)
            word_candidate_list = [item[0] for item in cadidate_list]

            word_candidate_list = np.array(word_candidate_list).flatten().tolist()
            return word_candidate_list
        else:
            return ["None"]

def find_closes_time_index(target_timestamps, source_timestamp, return_diff=False):
    index_in_target = np.argmin(np.abs(target_timestamps - source_timestamp))
    if return_diff:
        return index_in_target, target_timestamps[index_in_target] - source_timestamp
    return index_in_target