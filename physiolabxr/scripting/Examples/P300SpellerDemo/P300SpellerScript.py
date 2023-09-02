import pickle
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet
from sklearn.linear_model import LogisticRegression
from physiolabxr.scripting.Examples.P300SpellerDemo.Cyton8ChannelsConfig import *
from physiolabxr.scripting.Examples.P300SpellerDemo.P300Speller_utils import *
from physiolabxr.scripting.RenaScript import RenaScript
# from physiolabxr.utils.general import DataBuffer
from physiolabxr.utils.buffers import DataBuffer


class P300Speller(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        # params = BrainFlowInputParams()
        # board = BoardShim(2, params)
        # self.eeg_names = board.get_eeg_names(2)

        self.info = mne.create_info(eeg_channel_names, eeg_sampling_rate, ch_types=channel_types)
        self.info['description'] = 'P300Speller'

        # during for egg time locking
        self.time_offset_before = -0.2
        self.time_offset_after = 1

        self.data_buffer = DataBuffer()
        self.board_state = IDLE_STATE  # game board state
        self.game_state = IDLE_STATE  # game state
        self.model_name = 'P300SpellerModel'
        self.train_data_filename = ''
        self.test_data_filename = ''

        self.training_raw = []
        self.testing_raw = []

        # outlet
        self.lsl_info = StreamInfo("P300SpellerRenaScript", "P300SpellerRenaScript", 2, 10, 'float32', 'someuuid1234')
        self.p300_speller_script_lsl = StreamOutlet(self.lsl_info)

        print("P300Speller Decoding Script Setup Complete!")

        self.model = LogisticRegression()
        # with (open('C:/Users/Haowe/Desktop/RENA/data/P300SpellerData/HaowenCompleteTestDataModel/model.pickle', "rb")) as openfile:
        #     self.model = pickle.load(openfile)

    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):

        #
        if P300EventStreamName not in self.inputs.keys() or OpenBCIStreamName not in self.inputs.keys():
            return

        # print(self.inputs.get_data(P300EventStreamName)[
        #     p300_speller_event_marker_channel_index['P300SpellerGameStateControlMarker']])

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
                # print('collect training state data')

        elif self.game_state == TESTING_STATE:
            if END_TESTING_MARKER in self.inputs.get_data(P300EventStreamName)[
                p300_speller_event_marker_channel_index['P300SpellerGameStateControlMarker']]:
                self.inputs.clear_buffer_data()  # clear buffer
                self.game_state = IDLE_STATE
                print('end testing state')
                # report final result
            else:
                self.collect_trail(self.testing_callback)
                # print('collect testing state data')

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
                self.data_buffer.update_buffers(self.inputs.buffer[eeg_channel_index,:])  # update the data_buffer with all inputs
                self.inputs.clear_buffer_data()  # clear the data buffer
        else:
            print("WARNING: Unknown state")

    def training_callback(self):
        x_list = []
        y_list = []
        epoch = None
        raw = self.generate_raw_data()
        self.training_raw.append(raw)
        # run the post-processing for each trail
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

        # visualization function (optional)
        visualize_eeg_epochs(epoch, event_id, event_color, eeg_channel_names)

        x = np.concatenate([x for x in x_list])  # collect all x samples
        y = np.concatenate([y for y in y_list])  # collect all y samples
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=test_size)
        x_train, y_train = rebalance_classes(x_train, y_train, by_channel=True)  # re-balance the class
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        confusion_matrix(y_test, y_pred)
        score = f1_score(y_test, y_pred, average='macro')
        print("score:", score)
        self.p300_speller_script_lsl.push_sample([TRAIN_RESPONSE_MARKER, score])
        # push the result

    def callback_to_lsl(self):
        pass

    def testing_callback(self):
        raw = self.generate_raw_data()
        self.testing_raw.append(raw)
        raw_processed = p300_speller_process_raw_data(raw, l_freq=1, h_freq=50, notch_f=60, picks='eeg')
        flashing_events = mne.find_events(raw_processed, stim_channel='P300SpellerFlashingMarker')
        epoch = mne.Epochs(raw_processed, flashing_events, tmin=-0.1, tmax=1, baseline=(-0.1, 0),
                           event_id={'flashing_event': 1},
                           preload=True)
        x = epoch.get_data(picks='eeg')
        x = x.reshape(x.shape[0], -1)
        y_pred_target_prob = self.model.predict_proba(x)[:, 1]

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

        target_row_index = np.argmax(
            [row_1.mean(), row_2.mean(), row_3.mean(), row_4.mean(), row_5.mean(), row_6.mean()])
        target_col_index = np.argmax([col_1.mean(), col_2.mean(), col_3.mean(), col_4.mean(), col_5.mean()])

        # print("TargetRow: ", target_row_index)
        # print("TargetCol: ", target_col_index)
        #
        target_char_index = target_row_index * 5 + target_col_index
        print("targetIndex:")
        print(target_char_index)
        self.p300_speller_script_lsl.push_sample([TEST_RESPONSE_MARKER, target_char_index])

    def generate_raw_data(self):
        # add stim channels to raw data
        flashing_markers, flashing_row_or_colum_marker, flashing_row_colum_index_marker, target_non_target_marker, flashing_ts = self.get_p300_speller_events()
        raw_array = mne.io.RawArray(self.data_buffer[OpenBCIStreamName][0], self.info)

        add_stim_channel(raw_array, self.data_buffer[OpenBCIStreamName][1], flashing_ts, flashing_markers,
                         stim_channel_name='P300SpellerFlashingMarker')
        add_stim_channel(raw_array, self.data_buffer[OpenBCIStreamName][1], flashing_ts, flashing_row_or_colum_marker,
                         stim_channel_name='P300SpellerFlashingRowOrColumMarker')
        add_stim_channel(raw_array, self.data_buffer[OpenBCIStreamName][1], flashing_ts, flashing_row_colum_index_marker,
                         stim_channel_name='P300SpellerFlashingRowColumIndexMarker')
        add_stim_channel(raw_array, self.data_buffer[OpenBCIStreamName][1], flashing_ts, target_non_target_marker,
                         stim_channel_name='P300SpellerTargetNonTargetMarker')

        return raw_array

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
