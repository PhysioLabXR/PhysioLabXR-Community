import time

import cv2
import numpy as np
import scipy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn.metrics import classification_report

from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.events.utils import get_indices_when
from physiolabxr.scripting.physio.HDCA import HDCA
from physiolabxr.scripting.physio.epochs import get_event_locked_data, buffer_event_locked_data, \
    get_baselined_event_locked_data, visualize_epochs
from physiolabxr.scripting.physio.interpolation import interpolate_zeros
from physiolabxr.scripting.physio.preprocess import preprocess_samples_eeg_pupil
from physiolabxr.scripting.physio.utils import rebalance_classes


class ERPClassifier(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        self.dtn_events = (1, 2)  # 1 is distractor, 2 is target

        self.event_marker_name = 'Example-EventMarker-DTN-Block'
        self.video_output_name = 'Example-Video-Output'
        self.video_input_name = 'Example-Video'
        self.video_shape = (400, 400, 3)  # the shape of the video stream, because all inputs are flattened as they comes in, we need to reshape the video frames to be able to put shapes on them
        self.last_dtn_time = 0

        self.eeg_channels = self.get_stream_info('Example-BioSemi-64Chan', 'ChannelNames')  # List of EEG channels
        self.eeg_srate = self.get_stream_info('Example-BioSemi-64Chan', 'NominalSamplingRate')  # Sampling rate of the EEG data in Hz
        self.tmin_eeg = -0.1  # Time before event marker to include in the epoch
        self.tmax_eeg = 0.8  # Time after event marker to include in the epoch

        self.eeg_baseline_time = 0.1  # Time period since the ERP epoch start to use as baseline
        self.eeg_erp_length = int((self.tmin_eeg - self.tmax_eeg) * self.eeg_srate)  # Length of the ERP epoch in samples

        self.eeg_midlines = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'Iz']  # List of midline EEG channels
        self.eeg_picks = {self.eeg_channels.index(channel): channel for channel in self.eeg_midlines}  # Indices of the midline EEG channels, used for visualization

        self.eye_channels = self.get_stream_info('Example-Eyetracking-Pupil', 'ChannelNames')  # List of eye tracking channels
        self.eye_srate = self.get_stream_info('Example-Eyetracking-Pupil', 'NominalSamplingRate')  # Sampling rate of the eye tracking data in Hz

        self.tmin_eye = -0.5  # Time before event marker to include in the epoch
        self.tmax_eye = 3.  # Time after event marker to include in the epoch
        self.eye_baseline_time = 0.5  # Time period since the ERP epoch start to use as baseline
        self.eye_erp_length = int((self.tmin_eye - self.tmax_eye) * self.eye_srate)  # Length of the ERP epoch in samples

        self.event_locked_data_buffer = {}  # Dictionary to store event-locked data
        self.process_next_look = False
        self.block_end_time = None
        self.block_count = 0
        self.start_training_at_block = 1

        self.random_seed = 42

        self.model = None
        self.event_indicators = []
        self.prediction_indicators = []
        self.indicator_radius = int(0.5 * 0.9 * self.video_shape[0] // 30)  # 0.5 for the radius, 0.9 for the margins, 30 for the number items per block
        self.indicator_separation = int(self.video_shape[0] // 30)

    def loop(self):
        if self.process_next_look:
            self.process_next_look = False
            self.block_count += 1
            # we need to interpolate the pupil data to remove the blinks when pupil size will be zeroes
            left_pupil = self.inputs['Example-Eyetracking-Pupil'][0][self.eye_channels.index("Left Pupil Size")]
            right_pupil = self.inputs['Example-Eyetracking-Pupil'][0][self.eye_channels.index("Right Pupil Size")]
            self.inputs['Example-Eyetracking-Pupil'][0][self.eye_channels.index("Left Pupil Size")] = interpolate_zeros(left_pupil)
            self.inputs['Example-Eyetracking-Pupil'][0][self.eye_channels.index("Right Pupil Size")] = interpolate_zeros(right_pupil)

            # filter the eeg data
            # self.inputs['Example-BioSemi-64Chan'][0] = mne.filter.filter_data(self.inputs['Example-BioSemi-64Chan'][0], sfreq=self.eeg_srate, l_freq=1, h_freq=50, n_jobs=1)

            # get the event-locked EEG data
            # get the event-locked eeg and eye (pupil) data
            event_locked_data, last_event_time = get_event_locked_data(event_marker=self.inputs[self.event_marker_name],
                                                                       data={'eeg': self.inputs['Example-BioSemi-64Chan'],
                                                                             'eye': self.inputs['Example-Eyetracking-Pupil']},
                                                                       events_of_interest=self.dtn_events,
                                                                       tmin={'eeg': self.tmin_eeg, 'eye': self.tmin_eye},
                                                                       tmax={'eeg': self.tmax_eeg, 'eye': self.tmax_eye},
                                                                       srate={'eeg': self.eeg_srate, 'eye': self.eye_srate},
                                                                       return_last_event_time=True, verbose=1)

            self.inputs.clear_up_to(self.block_end_time)  # Clear the input buffer up to the last event time to avoid processing duplicate data
            self.event_locked_data_buffer = buffer_event_locked_data(event_locked_data, self.event_locked_data_buffer)  # Buffer the event-locked data for further processing

            # visualize eeg and pupil separately
            eeg_epochs = {event: data['eeg'] for event, data in self.event_locked_data_buffer.items()}
            baselined_eeg_epochs = get_baselined_event_locked_data(eeg_epochs,
                                                                   self.eeg_baseline_time,
                                                                   self.eeg_srate)  # Obtain baselined event-locked data for the chosen channel

            visualize_epochs(baselined_eeg_epochs, picks=self.eeg_picks)

            pupil_epochs = {event: data['eye'][:, 4:] for event, data in self.event_locked_data_buffer.items()}

            baselined_pupil_epochs = get_baselined_event_locked_data(pupil_epochs,
                                                                     self.eye_baseline_time,
                                                                     self.eye_srate)
            # downsample eye tracking data
            baselined_resampled_pupil_epochs = {e: scipy.signal.resample(x, x.shape[-1] // 10, axis=-1) for e, x in baselined_pupil_epochs.items()}
            visualize_epochs(baselined_resampled_pupil_epochs)

            if self.block_count >= self.start_training_at_block:
                # build classifier
                y = np.concatenate([np.ones(x['eeg'].shape[0]) * e for e, x in self.event_locked_data_buffer.items()], axis=0)
                # adjust the labels's value to from 1, 2 to 0, 1
                y = y - 1
                # count the target ratio
                print(f"target: {np.sum(y)}. distractor {np.sum(y==0)}. target ratio: {np.sum(y == 1) / len(y)}")


                x_eeg = np.concatenate([x for e, x in baselined_eeg_epochs.items()], axis=0)
                x_pupil = np.concatenate([x for e, x in baselined_resampled_pupil_epochs.items()], axis=0)

                # rebalance the dataset
                x_eeg, y_eeg = rebalance_classes(x_eeg, y, by_channel=True, random_seed=42)
                x_pupil, y_pupil = rebalance_classes(x_pupil, y, by_channel=True, random_seed=42)

                assert np.all(y_eeg == y_pupil)
                y = y_eeg

                # preprocess the data, compute pca and ica for eeg
                x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed, pca, ica = preprocess_samples_eeg_pupil(x_eeg, x_pupil, 20)

                # split the data into train and test, we only do one split as we are not doing cross validation here
                skf = StratifiedShuffleSplit(n_splits=1, random_state=self.random_seed, test_size=0.2)
                train, test = [(train, test) for train, test in skf.split(x_eeg, y)][0]
                x_eeg_train, x_eeg_pca_ica_train = x_eeg_znormed[train], x_eeg_pca_ica[train]
                x_eeg_test, x_eeg_pca_ica_test = x_eeg_znormed[test], x_eeg_pca_ica[test]
                x_pupil_train, x_pupil_test = x_pupil_znormed[train], x_pupil_znormed[test]
                y_train, y_test = y[train], y[test]

                target_names = ['distractor', 'target']

                # hdca classifier

                self.model = HDCA(target_names)
                roc_auc_combined_train, roc_auc_eeg_train, roc_auc_pupil_train = self.model.fit(x_eeg_train, x_eeg_pca_ica_train, x_pupil_train, y_train, num_folds=1, is_plots=True, exg_srate=self.eeg_srate, notes=f"Block ID {self.block_count}", verbose=0, random_seed=self.random_seed)  # give the original eeg data, no need to apply HDCA again
                y_pred_train, roc_auc_eeg_pupil_test, roc_auc_eeg_test, roc_auc_pupil_test = self.model.eval(x_eeg_train, x_eeg_pca_ica_train, x_pupil_train, y_train, notes=f"Block ID {self.block_count}")
                y_pred_test, roc_auc_eeg_pupil_test, roc_auc_eeg_test, roc_auc_pupil_test = self.model.eval(x_eeg_test, x_eeg_pca_ica_test, x_pupil_test, y_test, notes=f"Block ID {self.block_count}")

                # report the results
                print(f"Block ID {self.block_count}:\n")
                print('Train performance: \n' + classification_report(y_train, y_pred_train, target_names=target_names))
                print(f"combined ROC {roc_auc_combined_train}, ROC EEG {roc_auc_eeg_train}, ROC pupil {roc_auc_pupil_train}")

                print('Test performance: \n' + classification_report(y_test, y_pred_test, target_names=target_names))
                print(f"combined ROC {roc_auc_eeg_pupil_test}, ROC EEG {roc_auc_eeg_test}, ROC pupil {roc_auc_pupil_test}")

                # TODO add single trial decoding, after a certain number of blocks start single trial decoding
                # TODO also include the camera data, visualize the decoder results on the camera image

        if self.event_marker_name in self.inputs.keys():
            event_markers = self.inputs[self.event_marker_name][0]
            event_times = self.inputs[self.event_marker_name][1]

            block_ids = event_markers[1, :]
            if (block_end_indices := get_indices_when(block_ids, lambda x: x < 0)) is not None:  # if we received the block end event, which is a negative block ID
                self.block_end_time = event_times[block_end_indices[-1]]
                self.process_next_look = True
                # clear the event and prediction indicators at the end of each block
                self.event_indicators = []
                self.prediction_indicators = []
                time.sleep(self.tmax_eye * 1.5)  # wait for the last pupil epoch to be received

            # check if any DTN (distractor, target, or novelty) event is received
            dtns = event_markers[0, :][event_times > self.last_dtn_time]
            dtn_indices, found_dtns = get_indices_when(dtns, lambda x: np.isin(x, self.dtn_events), return_data=True)
            if dtn_indices is not None:  # if we received the block end event, which is a negative block ID
                self.last_dtn_time = event_times[dtn_indices[-1]]
                # add the dtn events to the event indicators, so they can be visualized on the video
                self.event_indicators += found_dtns[:, 0].tolist()

        if self.video_input_name in self.inputs.keys():
            video_frame = self.inputs[self.video_input_name][0][:, -1].reshape(self.video_shape).copy()  # get the video frames
            if len(self.event_indicators) > 0:
                # draw event indicator as blue (distractor) and red (target) on the bottom part of the video frame
                for i, event_indicator in enumerate(self.event_indicators):
                    cv2.circle(video_frame, (self.video_shape[0] - 10, i * self.indicator_separation), self.indicator_radius, (0, 0, 255) if event_indicator == 1 else (255, 0, 0), 1)
            # send the video frame to the output
            self.outputs[self.video_output_name] = video_frame.reshape(-1)

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')


