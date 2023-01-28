from enum import Enum

import numpy as np
import mne
from brainflow import BrainFlowInputParams, BoardShim
from matplotlib import pyplot as plt
from mne.time_frequency import psd_welch

from rena.scripting.RenaScript import RenaScript
import brainflow

# class P300DetectorMarker(Enum):
#
NONTARGET_MARKER = 1
TARGET_MARKER = 2

START_FLASHING_MARKER = 3
END_FLASHING_MARKER = 4

EEG_SAMPLING_RATE = 250.0

Time_Window = 1.1  # second

OpenBCIStreamName = 'OpenBCI_Cyton_8'

P300EventStreamName = 'P300Speller'


# FLASH_END_MARKER = 2

# EEG_SAMPLING_RATE = 125

# PAST_TIME_WINDOW = 4
# BASELINE_TIME = 0.1
# channel_picks = (4, 5, 6, 7)
# blink_frequencies = np.array([
#      4,
#      6.6,
#      7.5,
#      8.57,
#      10,
#      12,
#      15,
#      20,
#      25
# ])

class P300Detector(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        params = BrainFlowInputParams()
        board = BoardShim(2, params)
        self.eeg_names = board.get_eeg_names(2)
        self.mne_raw_info = mne.create_info(self.eeg_names, EEG_SAMPLING_RATE, ch_types='eeg')

        self.time_offset_before = 0.2
        self.time_offset_after = 1

        self.sample_num_before_event_marker = EEG_SAMPLING_RATE * self.time_offset_before
        self.sample_num_after_event_marker = EEG_SAMPLING_RATE * self.time_offset_after

        self.processed_event_marker_timestamp_offset = None

        print("Tic Tac Toe Decoding Script Setup Complete!")

        self.target_epoch = []
        self.non_target_epoch = []

    # def get_sample_num_before(self):
    #
    # def get_sample_num_after(self):

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        self.outputs['output1'] = [self.params['老李是傻逼']]
        if P300EventStreamName and OpenBCIStreamName not in self.inputs.keys():
            return

        if TARGET_MARKER or NONTARGET_MARKER in self.inputs.get_data(P300EventStreamName):
            # we have event marker in the data buffer
            event_marker_timestamps = self.inputs.get_timestamps(P300EventStreamName)
            event_marker_data = self.inputs.get_data(P300EventStreamName)

            # find the last unprocessed event_marker_time_stamp_index
            unprocessed_event_marker_indices = \
            np.where(event_marker_timestamps > self.processed_event_marker_timestamp_offset)[0]
            if unprocessed_event_marker_indices.size > 0:
                # we have unprocessed index
                last_unprocessed_event_marker_indices = unprocessed_event_marker_indices[0]
                # unprocessed event marker label and time stamps
                unprocessed_event_marker_timestamps = event_marker_timestamps[:, last_unprocessed_event_marker_indices:]
                unprocessed_event_marker_data = event_marker_timestamps[:, last_unprocessed_event_marker_indices:]

                for event_marker_index, event_marker in enumerate(unprocessed_event_marker_data):
                    event_marker_timestamp = unprocessed_event_marker_timestamps[event_marker_index]




                    if event_marker == NONTARGET_MARKER:
                        pass
                    elif event_marker == TARGET_MARKER:
                        pass
                    else:
                        pass

    def cleanup(self):
        print('Cleanup function is called')

        # ................|..................|(processed marker).................|...................|............
        # ....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|

        # stream name:
        #
        #

        # if 'TicTacToeEvents' in self.inputs.keys():
        #     # check if there is an end flashing event
        #     if np.any(self.inputs.get_data('TicTacToeEvents') == FLASH_END_MARKER):
        #         if 'EEG' not in self.inputs.keys():
        #             print('Warning: EEG data not available at the time when end flashing marker is received, not calculating SSVEP')
        #             return
        #         print('Flashing end event received, decoding frequency power density')
        #         flashing_end_time = self.inputs.get_timestamps('TicTacToeEvents')[np.argwhere(self.inputs.get_data('TicTacToeEvents') == FLASH_END_MARKER)][0, -1]
        #         epoch_start_time = flashing_end_time - PAST_TIME_WINDOW
        #         epoch_start_index = np.argmin(np.abs(self.inputs.get_timestamps('EEG') - epoch_start_time))
        #
        #         epoch = self.inputs.get_data('EEG')[:, epoch_start_index:]
        #         epoch_timestamps = self.inputs.get_timestamps('EEG')[epoch_start_index:]
        #
        #         decoded_freq_index = self.preprocess_data(epoch, epoch_timestamps)
        #         self.outputs['DecodedFreqIndex'] = [decoded_freq_index]
        #         print('Sent Decoded Sample {}'.format(decoded_freq_index))
        #         self.inputs.clear_stream_buffer('TicTacToeEvents')
    # cleanup is called when the stop button is hit
    # def cleanup(self):
    #     print('Cleanup function is called')

    # def preprocess_data(self, epoch, epoch_timestamps):
    #     processed = mne.filter.filter_data(epoch, EEG_SAMPLING_RATE, l_freq=50, h_freq=1, n_jobs=1)
    #     # processed = mne.filter.notch_filter(processed, EEG_SAMPLING_RATE, freqs=60, n_jobs=1)
    #     raw = mne.io.RawArray(processed, self.mne_raw_info)
    #     psd, freq = psd_welch(raw, n_fft=1028, n_per_seg=256 * 3, picks='all', n_jobs=1)
    #
    #     powers = []  # power density for each frequency
    #     for flashing_f in blink_frequencies:
    #         power = psd[(channel_picks), np.argmin(np.abs(freq - flashing_f))]
    #         powers.append(power.mean())
    #     print('Decoded powers: {}'.format(powers))
    #     return np.argmax(powers)
    #
    # def plot_welch(self, raw):
    #     psd, freq = psd_welch(raw, n_fft=1028, n_per_seg=256 * 3, picks='all', n_jobs=1)
    #     psd = 10 * np.log10(psd)
    #
    #     psd = psd[channel_picks, :]  # pick the occipital channels
    #     psd_mean = psd.mean(0)  # pick the occipital
    #     plt.plot(freq, psd_mean, color='b')
    #
    #     plt.ylabel('Power Spectral Density (dB)')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.legend()
    #
    #     plt.show()

# class P300DetectorMarker(Enum):
#
#     NONTARGET_MARKER = 1
#     TARGET_MARKER = 2
#
#     START_FLASHING_MARKER = 3
#     END_FLASHING_MARKER = 4
#
#     EEG_SAMPLING_RATE = 250.0
#
#     Time_Window = 1.1  # second
#
#     OpenBCIStreamName = 'OpenBCI_Cyton_8'
#
#     P300EventStreamName = 'P300Speller'
