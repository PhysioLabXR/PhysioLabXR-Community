from enum import Enum

import mne
import numpy as np
from mne import create_info
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.physio.epochs import get_event_locked_data
from physiolabxr.utils.buffers import flatten

event_marker_stream_name = 'EventMarker_BallGame'
class GameStates(Enum):
    idle = 'idle'
    train = 'train'
    fit = 'fit'
    eval = 'eval'
    
class Events(Enum):
    train_start = 1
    left_trial = 2
    right_trial = 3
    eval_start = 6

class CSPDecoder:

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis()
        self.clf = Pipeline([('CSP', self.csp), ('LDA', self.lda)])

    def fit(self, X, y):
        X = self.csp.fit_transform(X, y)
        # fit classifier
        self.lda.fit(X, y)

    def csp_transfomr(self, X):
        return self.csp.transform(X)

    def transform(self, X):
        X = self.csp.transform(X)
        return self.lda.transform(X)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class MotorImageryBalanceBall(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        self.cur_state = 'idle'

    # Start will be called once when the run button is hit.
    def init(self):
        # self.train_data_buffer = DataBuffer()
        # self.eval_data_buffer = DataBuffer()
        self.cur_state = 'idle'
        self.transition_markers = [Events.train_start.value, -Events.train_start.value, Events.eval_start.value, -Events.eval_start.value]
        self.eeg_channels = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "P4", ]
        self.decoder_tmin = 2.
        self.decoder_tmax = 5.
        self.srate = 128
        self.decode_t_len = int((self.decoder_tmax - self.decoder_tmin) * self.srate)
        self.label_mapping = {2: 0, 3: 1}
        self.decoder = None

        # loop is called <Run Frequency> times per second
    def loop(self):

        if event_marker_stream_name not in self.inputs.keys(): # or  #EVENT_MARKER_CHANNEL_NAME not in self.inputs.keys():
            # print('Event marker stream not found')
            return

        self.process_event_markers()
        if self.cur_state == GameStates.train:
            pass
            # keep collecting data
            # print("In training")
        elif self.cur_state == GameStates.eval:
            self.decode()
            # print("In evaluation")

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    def process_event_markers(self):
        if event_marker_stream_name in self.inputs.keys() and len(np.intersect1d(self.inputs[event_marker_stream_name][0], self.transition_markers)) > 0:
            last_processed_marker_index = None
            for i, event_marker in enumerate(self.inputs[event_marker_stream_name][0].T):
                game_event_marker = event_marker[0]
                print(f'Event marker is {event_marker} at index {i}')

                # state transition logic
                if game_event_marker == Events.train_start.value:
                    self.cur_state = GameStates.train
                    print('Entering training block')
                    last_processed_marker_index = i

                elif game_event_marker == -Events.train_start.value:  # exiting train state
                    # collect the trials and train the decoding model
                    self.collect_trials_and_train()
                    self.cur_state = GameStates.idle
                    print('Exiting training block')
                    last_processed_marker_index = i

                elif event_marker == Events.eval_start.value:
                    self.cur_state = GameStates.eval
                    print('Entering evaluation block')
                    last_processed_marker_index = i

                elif event_marker == -Events.eval_start.value:
                    self.cur_state = GameStates.idle
                    print('Exiting evaluation block')
                    last_processed_marker_index = i

            # # collect event marker data
            # if self.cur_state == GameStates.train:
            #     event_type = game_state_event_marker
            #     timestamp = self.inputs[event_marker_stream_name][1][i]
            #
            #     # self.train_data_buffer.
            #     pass
            #
            # elif self.cur_state == GameStates.eval:
            #     pass

        # self.inputs.clear_stream_buffer_data(event_marker_stream_name)
            if last_processed_marker_index is not None:
                self.inputs.clear_stream_up_to_index(event_marker_stream_name, last_processed_marker_index+1)

    def collect_trials_and_train(self):
        event_locked_data, last_event_time = get_event_locked_data(event_marker=self.inputs[event_marker_stream_name],
                                                                   data=self.inputs["OpenBCICyton8Channels"],
                                                                   events_of_interest=[Events.left_trial.value, Events.right_trial.value],
                                                                   tmin=self.decoder_tmin, tmax=self.decoder_tmax, srate=self.srate, return_last_event_time=True, verbose=1)
        # TODO check the shape of the event locked data, how long is it. does it equal decode_t_len

        train_end_index = np.argwhere(self.inputs[event_marker_stream_name][0][0] == - Events.train_start.value).item()
        train_end_time = self.inputs[event_marker_stream_name][1][train_end_index]
        self.inputs.clear_up_to(train_end_time)  # Clear the input buffer up to the last event time to avoid processing duplicate data

        # build the classifier, ref https://mne.tools/dev/auto_examples/decoding/decoding_csp_eeg.html
        labels = flatten([[events] * len(data) for events, data in event_locked_data.items()])
        labels = np.array([self.label_mapping[label] for label in labels])
        epochs_data = np.concatenate(list(event_locked_data.values()), axis=0)
        info = create_info(ch_names=self.eeg_channels, sfreq=self.srate, ch_types='eeg')
        montage = mne.channels.make_standard_montage("biosemi64")
        info.set_montage(montage)

        self.decoder = CSPDecoder(n_components=4)
        self.decoder.fit(epochs_data, labels)
        # get the classification score
        y_pred = self.decoder.transform(epochs_data)
        score = self.decoder.lda.score(self.decoder.csp_transfomr(epochs_data), labels)
        print(f"Fitting completed. Classification score: {score}. Plotting CSP...")
        self.decoder.csp.plot_patterns(info, ch_type="eeg", units="Patterns (AU)", size=1.5)

    def decode(self):
        data = self.inputs["OpenBCICyton8Channels"][0][None, :, -self.decode_t_len:]
        y_pred = self.decoder.transform(data)[0]  # only one sample in batch
        # normalize y_pred from -10 to 10 to 0 to 1
        y_pred = sigmoid(y_pred)

        self.outputs["MotorImageryInference"] = y_pred


