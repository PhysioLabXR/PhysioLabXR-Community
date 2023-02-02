from enum import Enum

import numpy as np
import mne
from brainflow import BrainFlowInputParams, BoardShim
from matplotlib import pyplot as plt
from mne.time_frequency import psd_welch

# from rena.examples.MNE_Example.mne_raw_example import generate_mne_stim_channel, add_stim_channel_to_raw_array
from rena.scripting.RenaScript import RenaScript
import brainflow

# class P300DetectorMarker(Enum):
#
from rena.utils.general import DataBuffer

FLASH_START_MARKER = 9
FLASH_END_MARKER = 10

NONTARGET_MARKER = 1
TARGET_MARKER = 2

START_FLASHING_MARKER = 3
END_FLASHING_MARKER = 4

EEG_SAMPLING_RATE = 250.0

Time_Window = 1.1  # second

OpenBCIStreamName = 'OpenBCI_Cython_8_LSL'

P300EventStreamName = 'P300Speller'

IDEAL_STATE = 0
RECORDING_STATE = 1

sampling_rate = 250
data_duration = 2
channel_num = 8
data_array = np.random.rand(8, data_duration * sampling_rate)

channel_types = ['eeg'] * 8
channel_names = [
    "Fp1",
    "Fp2",
    "C3",
    "C4",
    "P7",
    "P8",
    "O1",
    "O2"
]

event_id = {'target': 1, 'non_target': 2}

montage = 'standard_1005'


# info = mne.create_info(channel_names, sampling_rate, channel_types)
# info['description'] = 'P300Speller'


class P300Detector(RenaScript):
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
        self.raw = None
        self.time_offset_before = -0.2
        self.time_offset_after = 1

        self.data_buffer = DataBuffer()
        self.current_state = IDEAL_STATE

        print("P300Speller Decoding Script Setup Complete!")

    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        # self.outputs['output1'] = [self.params['老李是傻逼']]

        '''
        if self.cur_state == 'idle':
            if start_flshing_marker in self.inputs[EventMarkers]
                self.data_buffer = DataBuffer()
                next_state = 'recording'
                # also need to clear inputs here

        if self.cur_state == 'recording':
            if end_flashing_marker in self.inputs[EventMarkers]

                #### processing epochs for a block
                raw = mne.raw with EEG and stim
                raw = mne.filter(raw)
                target_distractor_events = mne.find_event()
                epochs = mne.Epochs(raw, target_distractor_events, start=-0.1, stop=1.0, baseline=(-0.1, 0))
                epochs.plot()
                #### end of processing epochs for a block

                next_state = 'idle'
            self.data_buffer.update_buffer(self.inputs)
            self.clear_inputs()

        cur_state = next_state
        '''

        if P300EventStreamName not in self.inputs.keys() or OpenBCIStreamName not in self.inputs.keys():
            return
        if self.current_state == IDEAL_STATE:
            if FLASH_START_MARKER in self.inputs.get_data(P300EventStreamName):
                # self.data_buffer = DataBuffer()
                self.inputs.clear_buffer_data()  # clear buffer
                self.current_state = RECORDING_STATE
                return

        if self.current_state == RECORDING_STATE:
            if FLASH_END_MARKER in self.inputs.get_data(P300EventStreamName):
                # processed = mne.filter.filter_data(self.data_buffer[OpenBCIStreamName][0],
                #                                    EEG_SAMPLING_RATE,
                #                                    l_freq=1, h_freq=50, n_jobs=1)
                # processed = mne.filter.notch_filter(processed, EEG_SAMPLING_RATE, freqs=60, n_jobs=1)
                # self.raw = mne.io.RawArray(processed, self.mne_raw_info)
                evoked = self.process_epoch_data()
                self.data_buffer.clear_buffer_data()
                self.inputs.clear_buffer_data()
                self.current_state = IDEAL_STATE

            # print('flash')
            self.data_buffer.update_buffers(self.inputs.buffer)  # update the data_buffer with all inputs
            self.inputs.clear_buffer_data()  # clear the data buffer

    def cleanup(self):
        print('Cleanup function is called')

        # ................|..................|(processed marker).................|...................|............
        # ....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|....|

    def process_epoch_data(self):
        # self.data_buffer
        self.raw = mne.io.RawArray(self.data_buffer[OpenBCIStreamName][0], self.info)
        stim_data = generate_mne_stim_channel(data_ts=self.data_buffer[OpenBCIStreamName][1],
                                              event_ts=self.data_buffer[P300EventStreamName][1],
                                              events=self.data_buffer[P300EventStreamName][0])

        add_stim_channel_to_raw_array(raw_array=self.raw, stim_data=stim_data)
        flashing_events = mne.find_events(self.raw, stim_channel='STI')
        epochs = mne.Epochs(self.raw, flashing_events, tmin=-0.1, tmax=1, baseline=(-0.1, 0), event_id=event_id,
                            preload=True)
        evoked = epochs.average(by_event_type=True)
        return evoked
        # target_epochs = epochs['target']
        # non_target_epochs = epochs['non_target']

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


def generate_mne_stim_channel(data_ts, event_ts, events, deviate=25e-2):
    stim_array = np.zeros((1, data_ts.shape[0]))

    # event_data_indices = []
    # for t_e in event_ts:
    #     min_ts = math.inf
    #     min_ts_index = None
    #     for i, t_d in enumerate(data_ts):
    #         t_diff = abs(t_e - t_d)
    #         if t_diff < min_ts:
    #             min_ts = t_diff
    #             min_ts_index = i
    #     event_data_indices.append(min_ts_index)

    event_data_indices = [np.argmin(np.abs(data_ts - t)) for t in event_ts if
                          np.min(np.abs(data_ts - t)) < deviate]

    for index, event_data_index in enumerate(event_data_indices):
        stim_array[0, event_data_index] = events[0, index]

    return stim_array


def add_stim_channel_to_raw_array(raw_array, stim_data, stim_channel_name='STI'):
    # if len(stim_data.shape)==1:
    #     stim_data = stim_data.reshape(1,stim_data.shape[0])
    info = mne.create_info([stim_channel_name], raw_array.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_data, info)
    raw_array.add_channels([stim_raw], force_update_info=True)
