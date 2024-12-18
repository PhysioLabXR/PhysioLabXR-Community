import time
from collections import deque, OrderedDict
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
from physiolabxr.scripting.illumiRead.illumiReadSwype.eeg_prediction import full_eeg_pipeline
import csv
import pandas as pd
from nltk import RegexpTokenizer
from physiolabxr.rpc.decorator import rpc, async_rpc
import re
from physiolabxr.scripting.illumiRead.illumiReadSwype.train_sweyepe_model import train_model
from physiolabxr.scripting.illumiRead.illumiReadSwype.train_eeg_preprocessing import preprocess_eeg,save_processed_eeg

# Keyboard proximity map for horizontally adjacent keys
keyboard_proximity = {
    'q': 'qw', 'w': 'we', 'e': 'er', 'r': 'rt', 't': 'ty',
    'y': 'yu', 'u': 'ui', 'i': 'io', 'o': 'op', 'p': 'o',
    'a': 'as', 's': 'ad', 'd': 'sf', 'f': 'dg', 'g': 'fh',
    'h': 'gj', 'j': 'hk', 'k': 'jl', 'l': 'k',
    'z': 'zx', 'x': 'zc', 'c': 'xv', 'v': 'cb', 'b': 'vn',
    'n': 'bm', 'm': 'n'
}

# Helper function to check if two letters are close based on the proximity map
def matches_with_proximity(letter, possible_letters):
    """Check if a letter or any of its close neighbors are in possible_letters."""
    neighbors = keyboard_proximity.get(letter, '') + letter
    return any(l in possible_letters for l in neighbors)

# Helper function to check if two consecutive letters in target text can be parsed together
def parse_two_with_proximity(current_letter, next_letter, possible_letters):
    """Check if current and next letter match the fixation with proximity."""
    if current_letter in possible_letters and next_letter in possible_letters:
        if next_letter in keyboard_proximity.get(current_letter, ''):
            return True
    return False
def parse_tuple(val):
    """Convert a string tuple like '(-0.5332503 0.01100001)' to a Python tuple."""
    return tuple(map(float, val.strip('()').split()))


def parse_letter_locations(gaze_data_path):
    """Parse key ground truth locations from the CSV."""
    letters = []
    key_ground_truth_local = []

    with open(gaze_data_path, 'r') as file:
        header = file.readline()  # Skip header row
        for line in file:
            if not line.strip() or line.startswith('Key'):  # Skip invalid lines
                continue
            if re.match(r'^[a-zA-Z]', line):  # Match valid letter rows
                parts = line.strip().split(',')
                letters.append(parts[0])  # Letter
                key_ground_truth_local.append(parse_tuple(parts[3]))  # KeyGroundTruthLocal column

    # Group by letter and calculate mean location
    df = pd.DataFrame({'Letter': letters, 'KeyGroundTruthLocal': key_ground_truth_local})
    grouped = df.groupby('Letter')['KeyGroundTruthLocal']
    letter_locations = OrderedDict()
    for letter, group in grouped:
        ground_truth_array = np.array(list(group))
        letter_locations[letter] = np.mean(ground_truth_array, axis=0)
    return letter_locations


class IllumiReadSwypeScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        self.all_context = []

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
        self.collected_fixations = []  # Store final fixations for evaluation

        self.process_gaze_data_time_buffer = deque(maxlen=1000)

        self.ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)

        self.eeg_fixations = []  # Store fixation points and associated timestamps in collection state
        # self.gaze_data_sequence = []  # Buffer for gaze data in real-time processing
        self.letter_locations = None  # Letter positions for mapping
        self.fixation_summary = []  # Final fixation summary with possible letters

        # Load letter locations for mapping
        gaze_data_path = r'C:\Users\6173-group\Documents\PhysioLabXR\physiolabxr\scripting\illumiRead\illumiReadSwype\gaze2word\GazeData.csv'
        self.letter_locations = parse_letter_locations(gaze_data_path)
        self.evaluation_done = False  # Ensure the evaluation runs only once
        self.trace_eeg = {"channels": [], "timestamps": []}
        # spelling correction
        # self.spell_corrector = SpellCorrector()f
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
        gaze_data_path = r'C:\Users\6173-group\Documents\PhysioLabXR\physiolabxr\scripting\illumiRead\illumiReadSwype\gaze2word\GazeData.csv'

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
        if state_marker == illumiReadSwypeConfig.ExperimentState.EegState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.EegState
            # Initialize state EEG storage
            self.stateEeg = {"channels": [], "timestamps": []}  # Save EEG data during the state
            print("Entered EEG State")

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
        elif state_marker == illumiReadSwypeConfig.ExperimentState.EegState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.EegState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.IntroductionEegState.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.IntroductionEegState
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState_noeeg.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState_noeeg
        elif state_marker == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeIntroductionState_noeeg.value:
            self.currentExperimentState = illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeIntroductionState_noeeg




    def exit_state(self, state_marker):
        """Exit state logic extended for EEG state."""
        if state_marker == -illumiReadSwypeConfig.ExperimentState.EegState.value:
            # Step 1: Mark fixation data
            fixation_results = self.mark_fixation_data()
            print(fixation_results)

            # Step 2: Process and mark EEG data
            marked_eeg_results = self.process_eeg_with_fixation_results(fixation_results)
            print(marked_eeg_results)
            self.save_marked_eeg_to_csv(marked_eeg_results)
            time.sleep(3)

            weight,bias = train_model(r'C:\Users\6173-group\Documents\PhysioLabXR\physiolabxr\scripting\illumiRead\illumiReadSwype\m'
                                      r''
                                      r'arked_eeg_results.csv')


            # Reset EEG state storage
            self.stateEeg = None

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
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.EegState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.IntroductionEegState.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState_noeeg.value:
            self.currentExperimentState = None
        elif state_marker == -illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeIntroductionState_noeeg.value:
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
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.EegState:
            self.keyboard_eeg_state_callback()
        elif self.currentExperimentState == illumiReadSwypeConfig.ExperimentState.KeyboardIllumiReadSwypeState_noeeg:
            self.keyboard_illumireadswype_state_noeeg_callback()

    def keyboard_click_state_callback(self):
        # print("keyboard click state")

        pass

    def keyboard_dewelltime_state_callback(self):
        # print("keyboard dewell time state")
        pass

    def keyboard_illumireadswype_state_callback(self):
        """Real-time processing of IllumiRead Swype state with separate fixation and letter mapping logic."""
        # Process user input data
        for user_input_data_t in self.inputs[UserInputLSLStreamInfo.StreamName][0].T:
            user_input_button_2 = user_input_data_t[
                illumiReadSwypeConfig.UserInputLSLStreamInfo.UserInputButton2ChannelIndex
            ]  # Swyping invoker

            if not self.illumiReadSwyping and user_input_button_2:
                # Start swyping
                self.illumiReadSwyping = True
                print("Start swyping")

            if self.illumiReadSwyping and not user_input_button_2:
                # End swyping
                print("End swyping and decode swype path")

                # Group gaze data into fixations
                grouped_list = [
                    list(group) for key, group in groupby(self.gaze_data_sequence, key=lambda x: x.gaze_type)
                ]

                # fixation_summary = []  # Store fixation start, end, and possible letters
                fixation_trace = []  # Store keyboard hit points for gaze2word
                fixation_timestamps = []  # Store timestamps for each fixation point
                for group in grouped_list:
                    if group[0].gaze_type == GazeType.FIXATION:
                        fixation_start_time = group[0].timestamp
                        fixation_end_time = group[-1].timestamp

                        # # Collect gaze fixation points
                        # fixation_points = [
                        #     (gaze_point.timestamp, gaze_point.get_combined_eye_gaze_direction()[:2])
                        #     for gaze_point in group
                        # ]
                        #
                        # # Generate fixation centroid (gaze points only)
                        # fixation_array = np.array([[point[1][0], point[1][1]] for point in fixation_points])
                        # if len(fixation_array) == 0:
                        #     continue  # Skip if no valid fixation points
                        # centroid = np.mean(fixation_array, axis=0)

                        # Add keyboard hit points during this fixation for letter mapping
                        user_input_during_fixation_data, user_input_during_fixation_timestamps = self.data_buffer.get_stream_in_time_range(
                            UserInputLSLStreamInfo.StreamName, fixation_start_time, fixation_end_time
                        )
                        keyboard_hits_with_timestamps = [
                            (input_data.keyboard_background_hit_point_local[:2], timestamp)
                            for user_input, timestamp in zip(
                                user_input_during_fixation_data.T, user_input_during_fixation_timestamps
                            )
                            if not np.array_equal(
                                (input_data := illumiReadSwypeUserInput(user_input,
                                                                        timestamp)).keyboard_background_hit_point_local,
                                [1, 1, 1]
                            )
                        ]

                        # Save keyboard hit points for fixation trace
                        # fixation_trace.extend(keyboard_hit_points)
                        fixation_trace.extend([hit[0] for hit in keyboard_hits_with_timestamps])
                        fixation_timestamps.extend([hit[1] for hit in keyboard_hits_with_timestamps])


                        # Match keyboard hit points to letters
                        # possible_letters = [
                        #     letter for hit_point in keyboard_hit_points
                        #     for letter, location in self.letter_locations.items()
                        #     if np.linalg.norm(location - hit_point) <= 0.2  # Proximity radius
                        # ]

                        # Save fixation summary
                        # self.fixation_summary.append({
                        #     "fixation_start": fixation_start_time,
                        #     "fixation_end": fixation_end_time,
                        #     "possible_letters": list(set(possible_letters))  # Deduplicate letters
                        # })

                        # Debug: print fixation details
                        # print(
                        #     f"Fixation from {fixation_start_time} to {fixation_end_time}: Possible Letters - {list(set(possible_letters))}")

                # Reset swyping state

                # all_channels = np.concatenate(self.trace_eeg["channels"],
                #                               axis=1)  # Combine all slices along the sample axis
                # all_timestamps = np.concatenate(self.trace_eeg["timestamps"])  # Combine all timestamp slices

                all_channels = np.hstack(self.trace_eeg["channels"])  # Shape: (24, n)
                all_timestamps = np.hstack(self.trace_eeg["timestamps"])  # Shape: (n,)
                if fixation_trace:
                    print('hello')
                    # print("previous trace length: ",len(fixation_trace))
                    print(fixation_trace)
                    print(fixation_timestamps)
                    print(len(fixation_trace), len(fixation_timestamps))
                    print(fixation_timestamps[0])
                    print(fixation_timestamps[-1])
                    print('eegshapes trace')
                    print(all_channels.shape)
                    print(all_timestamps.shape)
                    print(all_timestamps[0])
                    print(all_timestamps[-1])

                    results = full_eeg_pipeline(all_channels, all_timestamps)

                    print("good_timestamps")
                    if results and results['predictions']:
                        print('true times')
                        if len(results['predictions']['true_timestamps']) >= 0:

                            print(results['predictions']['true_timestamps'])

                # progress eeg and fixation here

                    true_timestamps = np.array(results['predictions']['true_timestamps'])  # Final timestamps from EEG
                    fixation_timestamps = np.array(fixation_timestamps)  # Timestamps for the fixation trace
                    fixation_trace = np.array(fixation_trace)  # Trace points corresponding to fixation timestamps

                # Map each `true_timestamp` to the closest `fixation_timestamp`
                    closest_indices = np.abs(fixation_timestamps[:, None] - true_timestamps).argmin(axis=0)

                # Use the indices to filter the fixation trace
                    filtered_trace = fixation_trace[closest_indices]

                    print(filtered_trace)

                    if len(filtered_trace) < 20:
                        filtered_trace = fixation_trace

                    if 1 <= len(filtered_trace):
                        fixation_trace = filtered_trace


                self.trace_eeg = {"channels": [], "timestamps": []}
                self.illumiReadSwyping = False
                self.gaze_data_sequence = []
                self.data_buffer = DataBuffer()
                self.ivt_filter.reset_data_processor()

                # Process fixation trace with gaze2word if sufficient data
                if len(fixation_trace) >= 1:
                    fixation_trace = np.array(fixation_trace)
                    candidate_list = self.g2w.predict(
                        4, fixation_trace, run_dbscan=True, prefix=self.context, verbose=True,
                        filter_by_starting_letter=0.35, use_trimmed_vocab=True, njobs=16
                    )
                    word_candidate_list = [item[0] for item in candidate_list]
                    word_candidate_list = np.array(word_candidate_list).flatten().tolist()
                    print("Predicted Words:", word_candidate_list)

                    # Send top words to feedback state
                    lvt, overflow_flat = word_candidate_list_to_lvt(word_candidate_list)
                    self.illumireadswype_keyboard_suggestion_strip_lsl_outlet.push_sample(lvt)
        if self.illumiReadSwyping:
            for gaze_data_t, timestamp in zip(
                    self.inputs[GazeDataLSLStreamInfo.StreamName][0].T,
                    self.inputs[GazeDataLSLStreamInfo.StreamName][1]
            ):
                gaze_data = VarjoGazeData()
                gaze_data.construct_gaze_data_varjo(gaze_data_t, timestamp)
                gaze_data = self.ivt_filter.process_sample(gaze_data)
                self.gaze_data_sequence.append(gaze_data)

            # Update the data buffer for user input
            self.data_buffer.update_buffers(self.inputs.buffer)
        if self.inputs['DSI24']:

            dsi_channels = self.inputs["DSI24"][0]
            dsi_timestamps = self.inputs["DSI24"][1]
            self.trace_eeg["channels"].append(dsi_channels)  # Add all current EEG samples
            self.trace_eeg["timestamps"].append(dsi_timestamps)  # Add corresponding EEG time
        #     dsi_channels = self.inputs["DSI24"][0]
        #     dsi_timestamps = self.inputs["DSI24"][1]
        #     if self.stateEeg:
        #         self.stateEeg["channels"].append(dsi_channels)
        #         self.stateEeg["timestamps"].append(dsi_timestamps)

        # Clear processed data streams
        self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)
        self.inputs.clear_stream_buffer_data(UserInputLSLStreamInfo.StreamName)
        self.inputs.clear_stream_buffer_data("DSI24")

    def keyboard_freeswitch_state_callback(self):
        print("keyboard free switch state")

        pass

    def keyboard_eeg_state_callback(self):
        """Real-time processing of IllumiRead Swype state with separate fixation and letter mapping logic."""
        # Process user input data
        for user_input_data_t in self.inputs[UserInputLSLStreamInfo.StreamName][0].T:
            user_input_button_2 = user_input_data_t[
                illumiReadSwypeConfig.UserInputLSLStreamInfo.UserInputButton2ChannelIndex
            ]  # Swyping invoker

            if not self.illumiReadSwyping and user_input_button_2:
                # Start swyping
                self.illumiReadSwyping = True
                print("Start swyping")

            if self.illumiReadSwyping and not user_input_button_2:
                # End swyping
                print("End swyping and decode swype path")

                # Group gaze data into fixations
                grouped_list = [
                    list(group) for key, group in groupby(self.gaze_data_sequence, key=lambda x: x.gaze_type)
                ]

                # fixation_summary = []  # Store fixation start, end, and possible letters
                fixation_trace = []  # Store keyboard hit points for gaze2word
                # fixation_timestamps = []  # Store timestamps for each fixation point
                for group in grouped_list:
                    if group[0].gaze_type == GazeType.FIXATION:
                        fixation_start_time = group[0].timestamp
                        fixation_end_time = group[-1].timestamp

                        # Collect gaze fixation points
                        fixation_points = [
                            (gaze_point.timestamp, gaze_point.get_combined_eye_gaze_direction()[:2])
                            for gaze_point in group
                        ]

                        # Generate fixation centroid (gaze points only)
                        fixation_array = np.array([[point[1][0], point[1][1]] for point in fixation_points])
                        if len(fixation_array) == 0:
                            continue  # Skip if no valid fixation points
                        centroid = np.mean(fixation_array, axis=0)

                        # Add keyboard hit points during this fixation for letter mapping
                        user_input_during_fixation_data, user_input_during_fixation_timestamps = self.data_buffer.get_stream_in_time_range(
                            UserInputLSLStreamInfo.StreamName, fixation_start_time, fixation_end_time
                        )
                        keyboard_hit_points = [
                            input_data.keyboard_background_hit_point_local[:2]
                            for user_input, timestamp in zip(
                                user_input_during_fixation_data.T, user_input_during_fixation_timestamps
                            )
                            if not np.array_equal(
                                (input_data := illumiReadSwypeUserInput(user_input,
                                                                        timestamp)).keyboard_background_hit_point_local,
                                [1, 1, 1]
                            )
                        ]

                        # Save keyboard hit points for fixation trace
                        fixation_trace.extend(keyboard_hit_points)

                        # Match keyboard hit points to letters
                        possible_letters = [
                            letter for hit_point in keyboard_hit_points
                            for letter, location in self.letter_locations.items()
                            if np.linalg.norm(location - hit_point) <= 0.2  # Proximity radius
                        ]

                        # Save fixation summary
                        self.fixation_summary.append({
                            "fixation_start": fixation_start_time,
                            "fixation_end": fixation_end_time,
                            "possible_letters": list(set(possible_letters))  # Deduplicate letters
                        })

                        # Debug: print fixation details
                        print(
                            f"Fixation from {fixation_start_time} to {fixation_end_time}: Possible Letters - {list(set(possible_letters))}")

                # Reset swyping state
                self.illumiReadSwyping = False
                self.gaze_data_sequence = []
                self.data_buffer = DataBuffer()
                self.ivt_filter.reset_data_processor()

                # Process fixation trace with gaze2word if sufficient data
                if len(fixation_trace) >= 1:
                    fixation_trace = np.array(fixation_trace)
                    candidate_list = self.g2w.predict(
                        4, fixation_trace, run_dbscan=True, prefix=self.context, verbose=True,
                        filter_by_starting_letter=0.35, use_trimmed_vocab=True, njobs=16
                    )
                    word_candidate_list = [item[0] for item in candidate_list]
                    word_candidate_list = np.array(word_candidate_list).flatten().tolist()
                    print("Predicted Words:", word_candidate_list)

                    # Send top words to feedback state
                    lvt, overflow_flat = word_candidate_list_to_lvt(word_candidate_list)
                    self.illumireadswype_keyboard_suggestion_strip_lsl_outlet.push_sample(lvt)
        if self.illumiReadSwyping:
            for gaze_data_t, timestamp in zip(
                    self.inputs[GazeDataLSLStreamInfo.StreamName][0].T,
                    self.inputs[GazeDataLSLStreamInfo.StreamName][1]
            ):
                gaze_data = VarjoGazeData()
                gaze_data.construct_gaze_data_varjo(gaze_data_t, timestamp)
                gaze_data = self.ivt_filter.process_sample(gaze_data)
                self.gaze_data_sequence.append(gaze_data)

            # Update the data buffer for user input
            self.data_buffer.update_buffers(self.inputs.buffer)
        if self.inputs["DSI24"]:
            dsi_channels = self.inputs["DSI24"][0]
            dsi_timestamps = self.inputs["DSI24"][1]
            if self.stateEeg:
                self.stateEeg["channels"].append(dsi_channels)
                self.stateEeg["timestamps"].append(dsi_timestamps)

        # Clear processed data streams
        self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)
        self.inputs.clear_stream_buffer_data(UserInputLSLStreamInfo.StreamName)
        self.inputs.clear_stream_buffer_data("DSI24")

    def mark_fixation_data(self):
        """Mark fixation data with possible letter matches."""
        fixation_results = []  # Store fixation data with marking

        # # Target text for marking
        # action_info_path = 'ActionInfo.csv'  # Adjust path to your ActionInfo.csv
        # action_info = pd.read_csv(action_info_path)
        #
        # # Extract and flatten target text
        # sweyepe_texts = action_info.loc[action_info['conditionType'] == 'Sweyepe', 'targetText'].drop_duplicates()
        sweyepe_texts = ['Hello World']
        sweyepe_texts = [list(target_text.replace(" ", "").lower()) for target_text in sweyepe_texts]
        target_text = [char for sentence in sweyepe_texts for char in sentence]

        sentence_position = 0  # Track current position in the sentence

        # Process each fixation
        for fixation in self.fixation_summary:
            fixation_start = fixation['fixation_start']
            fixation_end = fixation['fixation_end']
            possible_letters = fixation['possible_letters']

            if sentence_position >= len(target_text):
                break

            current_letter = target_text[sentence_position]
            next_letter = target_text[sentence_position + 1] if sentence_position + 1 < len(target_text) else None

            if current_letter in possible_letters:
                # Mark fixation as 2 (correct)
                fixation_results.append({
                    "fixation_start": fixation_start,
                    "fixation_end": fixation_end,
                    "possible_letters": possible_letters,
                    "current_letter": current_letter,
                    "mark": 2
                })

                # Advance through consecutive identical letters
                while sentence_position + 1 < len(target_text) and target_text[sentence_position + 1] == current_letter:
                    sentence_position += 1
                    fixation_results.append({
                        "fixation_start": fixation_start,
                        "fixation_end": fixation_end,
                        "possible_letters": possible_letters,
                        "current_letter": current_letter,
                        "mark": 2
                    })

                # Check proximity for the next letter
                if next_letter and parse_two_with_proximity(current_letter, next_letter, possible_letters):
                    fixation_results.append({
                        "fixation_start": fixation_start,
                        "fixation_end": fixation_end,
                        "possible_letters": possible_letters,
                        "current_letter": next_letter,
                        "mark": 2
                    })
                    sentence_position += 1

                sentence_position += 1

            else:
                # Mark fixation as 1 (incorrect)
                fixation_results.append({
                    "fixation_start": fixation_start,
                    "fixation_end": fixation_end,
                    "possible_letters": possible_letters,
                    "current_letter": current_letter,
                    "mark": 1
                })

                if next_letter and matches_with_proximity(next_letter, possible_letters):
                    sentence_position += 1

        return fixation_results

    def process_eeg_with_fixation_results(self, fixation_results):
        """Match and mark each EEG timestamp based on fixation results."""
        if not self.stateEeg or not fixation_results:
            print("No EEG data or fixation results to process.")
            return []

        # Concatenate collected EEG data
        dsi_channels = np.hstack(self.stateEeg["channels"])  # Shape: (24, n)
        dsi_timestamps = np.hstack(self.stateEeg["timestamps"])  # Shape: (n,)

        preprocessed_eeg = preprocess_eeg(dsi_channels, dsi_timestamps)
        print(f"Preprocessed EEG: {preprocessed_eeg}")

        # Use preprocessed EEG data and timestamps for further steps
        processed_data = preprocessed_eeg['processed_data']  # Shape: (24, n)
        processed_timestamps = preprocessed_eeg['timestamps']  # Shape: (n,)

        # Initialize the results
        filtered_dsi_results = []

        # Iterate over all EEG timestamps and mark them
        for i, timestamp in enumerate(processed_timestamps):
            mark = 0  # Default mark for non-fixation time

            # Check if the timestamp falls within any fixation
            for fixation in fixation_results:
                fixation_start = fixation['fixation_start']
                fixation_end = fixation['fixation_end']
                if fixation_start <= timestamp <= fixation_end:
                    mark = fixation['mark']  # Assign mark (1 or 2)
                    break  # No need to check further fixations

            # Add the timestamp, corresponding EEG data, and mark
            filtered_dsi_results.append({
                "timestamp": timestamp,
                "eeg_data": processed_data[:, i],
                "mark": mark
            })

        return filtered_dsi_results

    def save_marked_eeg_to_csv(self, filtered_dsi_results):
        """Save the marked EEG data to a CSV file."""
        csv_file_path = r'C:\Users\6173-group\Documents\PhysioLabXR\physiolabxr\scripting\illumiRead\illumiReadSwype\marked_eeg_results.csv'
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['Timestamp'] + [f'Channel_{i + 1}' for i in range(24)] + ['Mark'])

            # Write the data
            for res in filtered_dsi_results:
                writer.writerow([res['timestamp']] + res['eeg_data'].tolist() + [res['mark']])

        print(f"Marked EEG results saved to {csv_file_path}")

    def keyboard_illumireadswype_state_noeeg_callback(self):
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
                            if not np.array_equal(user_input.keyboard_background_hit_point_local, [1, 1, 1]):
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
                    cadidate_list = self.g2w.predict(4, fixation_trace, run_dbscan=True, prefix=self.context,
                                                     verbose=True, filter_by_starting_letter=0.35,
                                                     use_trimmed_vocab=True, njobs=16)
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