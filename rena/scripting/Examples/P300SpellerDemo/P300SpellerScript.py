import pickle
from enum import Enum

import numpy as np
import mne
from brainflow import BrainFlowInputParams, BoardShim
from matplotlib import pyplot as plt

# from rena.examples.MNE_Example.mne_raw_example import generate_mne_stim_channel, add_stim_channel_to_raw_array
from pylsl import StreamInfo, StreamOutlet

from rena.scripting.RenaScript import RenaScript
import brainflow
from datetime import datetime
# class P300DetectorMarker(Enum):
#
from rena.utils.general import DataBuffer
from rena.scripting.Examples.P300SpellerDemo.P300Speller_params import *
from rena.scripting.Examples.P300SpellerDemo.P300Speller_utils import *
from sklearn.linear_model import LogisticRegression


# START_TRAINING_MARKER = 90
# END_TRAINING_MARKER = 91
#
# START_TESTING_MARKER = 100
# END_TESTING_MARKER = 101
#
# FLASH_START_MARKER = 9
# FLASH_END_MARKER = 10
#
# NONTARGET_MARKER = 11
# TARGET_MARKER = 12
#
# ROW_FLASH_LABEL = 1
# COL_FLASH_LABEL = 2
#
# IDLE_STATE = 0
# RECORDING_STATE = 1
#
# TRAINING_STATE = 99
# TESTING_STATE = 100
#
# START_FLASHING_MARKER = 3
# END_FLASHING_MARKER = 4
#
# EEG_SAMPLING_RATE = 250.0
#
# Time_Window = 1.1  # second
#
# OpenBCIStreamName = 'OpenBCI_Cython_8_LSL'
#
# P300EventStreamName = 'P300Speller'
#
# sampling_rate = 250
# data_duration = 2
# channel_num = 8
# data_array = np.random.rand(8, data_duration * sampling_rate)
#
# channel_types = ['eeg'] * 8
# # channel_names = [
# #     "Fp1",
# #     "Fp2",
# #     "C3",
# #     "C4",
# #     "P7",
# #     "P8",
# #     "O1",
# #     "O2"
# # ]
# channel_names = [
#     "Fz",
#     "Cz",
#     "Pz",
#     "C3",
#     "C4",
#     "P3",
#     "P4",
#     "O1"
# ]
#
# event_id = {'non_target': 11, 'target': 12}
#
# montage = 'standard_1005'


# info = mne.create_info(channel_names, sampling_rate, channel_types)
# info['description'] = 'P300Speller'


class P300Speller(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        # params = BrainFlowInputParams()
        # board = BoardShim(2, params)
        # self.eeg_names = board.get_eeg_names(2)
        self.info = mne.create_info(channel_names, EEG_SAMPLING_RATE, ch_types='eeg')
        self.info['description'] = 'P300Speller'
        self.time_offset_before = -0.2
        self.time_offset_after = 1
        self.data_buffer = DataBuffer()
        self.board_state = IDLE_STATE
        self.game_state = IDLE_STATE
        self.model_name = 'P300SpellerModel'
        self.train_data_filename = ''
        self.test_data_filename = ''

        self.training_raw = []
        self.testing_raw = []

        # outlet
        self.lsl_info = StreamInfo("P300SpellerRenaScript", "P300SpellerRenaScript", 1, 10, 'float32', 'someuuid1234')
        self.p300_speller_script_lsl = StreamOutlet(self.lsl_info)

        print("P300Speller Decoding Script Setup Complete!")

        self.model = LogisticRegression()
        # with (open('C:/Users/Haowe/Desktop/RENA/data/P300SpellerData/HaowenCompleteTestDataModel/model.pickle', "rb")) as openfile:
        #     self.model = pickle.load(openfile)

    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):

        if P300EventStreamName not in self.inputs.keys() or OpenBCIStreamName not in self.inputs.keys():
            return

        print(self.inputs.get_data(P300EventStreamName)[
            p300_speller_event_marker_channel_index['P300SpellerGameStateControlMarker']])

        if InterruptExperimentMarker in self.inputs.get_data(P300EventStreamName)[
            p300_speller_event_marker_channel_index['P300SpellerGameStateControlMarker']]:
            self.game_state = IDLE_STATE
            self.board_state = IDLE_STATE

            self.inputs.clear_buffer()
            self.data_buffer.clear_buffer()

            self.training_raw = []
            self.testing_raw = []

            print("Game interrupted. Back to Idle state and aboard all the collected data")

        if self.game_state == IDLE_STATE:
            if START_TRAINING_MARKER in self.inputs.get_data(P300EventStreamName)[
                p300_speller_event_marker_channel_index['P300SpellerGameStateControlMarker']]:
                self.inputs.clear_buffer_data()  # clear buffer
                self.game_state = TRAINING_STATE
                # report final result
                print('enter training state')
            elif START_TESTING_MARKER in self.inputs.get_data(P300EventStreamName)[
                p300_speller_event_marker_channel_index['P300SpellerGameStateControlMarker']]:
                self.inputs.clear_buffer_data()  # clear buffer
                self.game_state = TESTING_STATE
                print('enter testing state')

        elif self.game_state == TRAINING_STATE:
            if END_TRAINING_MARKER in self.inputs.get_data(P300EventStreamName)[
                p300_speller_event_marker_channel_index['P300SpellerGameStateControlMarker']]:
                self.inputs.clear_buffer_data()  # clear buffer
                self.game_state = IDLE_STATE
                # report final result
                print('end training state')
            else:
                self.collect_trail(self.training_callback)
                print('collect training state data')

        elif self.game_state == TESTING_STATE:
            if END_TESTING_MARKER in self.inputs.get_data(P300EventStreamName)[
                p300_speller_event_marker_channel_index['P300SpellerGameStateControlMarker']]:
                self.inputs.clear_buffer_data()  # clear buffer
                self.game_state = IDLE_STATE
                print('end testing state')
                # report final result
            else:
                self.collect_trail(self.testing_callback)
                print('collect testing state data')

    def cleanup(self):
        print('Cleanup function is called')
        # event marker followed by horizontal and vertical index
        # ................|..................|(processed marker).................|...................|............
        # ....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|

    def collect_trail(self, callback_function):
        if self.board_state == IDLE_STATE:
            if TRAIL_START_MARKER in self.inputs.get_data(P300EventStreamName)[
                p300_speller_event_marker_channel_index['P300SpellerStartTrailStartEndMarker']]:
                # self.data_buffer = DataBuffer()
                self.inputs.clear_buffer_data()  # clear buffer
                self.board_state = RECORDING_STATE

        elif self.board_state == RECORDING_STATE:
            if TRAIL_END_MARKER in self.inputs.get_data(P300EventStreamName)[
                p300_speller_event_marker_channel_index['P300SpellerStartTrailStartEndMarker']]:
                callback_function()
                self.data_buffer.clear_buffer_data()
                self.inputs.clear_buffer_data()
                self.board_state = IDLE_STATE
            else:
                self.data_buffer.update_buffers(self.inputs.buffer)  # update the data_buffer with all inputs
                self.inputs.clear_buffer_data()  # clear the data buffer
        else:
            pass

    def training_callback(self):
        x_list = []
        y_list = []
        epoch = None
        raw = self.generate_raw_data()
        self.training_raw.append(raw)
        for raw_array in self.training_raw:
            raw_processed = p300_speller_process_raw_data(raw_array, l_freq=1, h_freq=50, notch_f=60, picks='eeg',
                                                          )
            # raw_processed.plot_psd()
            flashing_events = mne.find_events(raw_processed, stim_channel='P300SpellerTargetNonTargetMarker')
            epoch = mne.Epochs(raw_processed, flashing_events, tmin=-0.1, tmax=1, baseline=(-0.1, 0), event_id=event_id,
                               preload=True)

            x = epoch.get_data(picks='eeg')
            y = epoch.events[:, 2]
            x_list.append(x)
            y_list.append(y)

        x = np.concatenate([x for x in x_list])
        y = np.concatenate([y for y in y_list])
        x, y = rebalance_classes(x, y, by_channel=True)
        x = x.reshape(x.shape[0], -1)

        visualize_eeg_epochs(epoch, event_id, event_color)
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=test_size)
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        confusion_matrix(y_test, y_pred)

        # push the result

    def callback_to_lsl(self):
        pass


    def testing_callback(self):
        raw = self.generate_raw_data()
        self.testing_raw.append(raw)
        raw_processed = p300_speller_process_raw_data(raw, l_freq=1, h_freq=50, notch_f=60, picks='eeg')
        flashing_events = mne.find_events(raw_processed, stim_channel='P300SpellerFlashingMarker')
        epoch = mne.Epochs(raw_processed, flashing_events, tmin=-0.1, tmax=1, baseline=(-0.1, 0), event_id={'flashing_event':1},
                           preload=True)
        x = epoch.get_data(picks='eeg')
        x = x.reshape(x.shape[0], -1)
        y_pred_target_prob = self.model.predict_proba(x)[:,1]

        colum_row_marker = mne.find_events(raw_processed, stim_channel='P300SpellerFlashingRowOrColumMarker')[:, -1]
        colum_row_index = mne.find_events(raw_processed, stim_channel='P300SpellerFlashingRowColumIndexMarker')[:, -1]

        merged_list = [(colum_row_marker[i], colum_row_index[i]) for i in range(0, len(colum_row_index))]

        # mask the row colum information
        row_1 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER, 1)]]
        row_2 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER, 2)]]
        row_3 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER, 3)]]
        row_4 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER, 4)]]
        row_5 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER, 5)]]
        row_6 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER, 6)]]

        col_1 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (COL_FLASHING_MARKER, 1)]]
        col_2 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (COL_FLASHING_MARKER, 2)]]
        col_3 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (COL_FLASHING_MARKER, 3)]]
        col_4 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (COL_FLASHING_MARKER, 4)]]
        col_5 = y_pred_target_prob[[i for i, j in enumerate(merged_list) if j == (COL_FLASHING_MARKER, 5)]]


        target_row_index = np.argmax([row_1.mean(), row_2.mean(), row_3.mean(), row_4.mean(), row_5.mean(), row_6.mean()])
        target_col_index = np.argmax([col_1.mean(), col_2.mean(), col_3.mean(), col_4.mean(), col_5.mean()])

        print("TargetRow: ", target_row_index)
        print("TargetCol: ", target_col_index)
        #
        target_char_index = target_row_index*5+target_col_index
        self.p300_speller_script_lsl.push_sample([target_char_index])

        # training update


        # def data_structure(self, raw, epoch, row_col_info):

    #     return {'raw': raw, 'epochs': epoch, 'row_col_info': row_col_info}

    def generate_raw_data(self):

        flashing_markers, flashing_row_or_colum_marker, flashing_row_colum_index_marker, target_non_target_marker, flashing_ts = self.get_p300_speller_events()
        raw = mne.io.RawArray(self.data_buffer[OpenBCIStreamName][0], self.info)

        add_stim_channel(raw, self.data_buffer[OpenBCIStreamName][1], flashing_ts, flashing_markers,
                         stim_channel_name='P300SpellerFlashingMarker')
        add_stim_channel(raw, self.data_buffer[OpenBCIStreamName][1], flashing_ts, flashing_row_or_colum_marker,
                         stim_channel_name='P300SpellerFlashingRowOrColumMarker')
        add_stim_channel(raw, self.data_buffer[OpenBCIStreamName][1], flashing_ts, flashing_row_colum_index_marker,
                         stim_channel_name='P300SpellerFlashingRowColumIndexMarker')
        add_stim_channel(raw, self.data_buffer[OpenBCIStreamName][1], flashing_ts, target_non_target_marker,
                         stim_channel_name='P300SpellerTargetNonTargetMarker')

        return raw

    def get_p300_speller_events(self):
        markers = self.data_buffer[P300EventStreamName][0]
        ts = self.data_buffer[P300EventStreamName][1]

        flashing_markers_channel = markers[p300_speller_event_marker_channel_index['P300SpellerFlashingMarker']]
        flashing_markers_index = np.where(flashing_markers_channel != 0)

        flashing_markers = markers[p300_speller_event_marker_channel_index['P300SpellerFlashingMarker']][
            flashing_markers_index]
        flashing_row_or_colum_marker = \
            markers[p300_speller_event_marker_channel_index['P300SpellerFlashingRowOrColumMarker']][
                flashing_markers_index]
        flashing_row_colum_index_marker = \
            markers[p300_speller_event_marker_channel_index['P300SpellerFlashingRowColumIndexMarker']][
                flashing_markers_index]
        target_non_target_marker = markers[p300_speller_event_marker_channel_index['P300SpellerTargetNonTargetMarker']][
            flashing_markers_index]
        flashing_ts = ts[flashing_markers_index]
        return flashing_markers, flashing_row_or_colum_marker, flashing_row_colum_index_marker, target_non_target_marker, flashing_ts

    def save_data(self):
        now = datetime.now()
        # if self.game_state == TRAINING_STATE:
        #     file_name = now.strftime("%m_%d_%Y_%H_%M_%S") + '_train.pickle'
        # elif self.game_state == TESTING_STATE:
        #     file_name = now.strftime("%m_%d_%Y_%H_%M_%S") + '_test.pickle'
        # else:
        #     return
        # with open(file_name, 'wb') as handle:
        #     pickle.dump(self.data_dict_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
