"""


You will find plenty of try, except Exception in the functions defined here. This is to help debugging with breakpoints
in case a function raises exception. We don't have to dig back into the function to find what's wrong
"""

import copy
import json
import pickle
import time
from collections import defaultdict

import mne
import numpy as np
import torch
from mne import Epochs, EpochsArray
from pylsl import StreamInfo, StreamOutlet, pylsl
from renaanalysis.eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim, \
    gaze_event_detection_I_DT
from renaanalysis.learning.models import EEGPupilCNN
from renaanalysis.learning.train import train_model_pupil_eeg, train_model_pupil_eeg_no_folds
from renaanalysis.params.params import conditions, dtnn_types, tmax_pupil, random_seed
from renaanalysis.utils.Event import get_events
from renaanalysis.utils.RenaDataFrame import RenaDataFrame
from renaanalysis.utils.data_utils import epochs_to_class_samples, compute_pca_ica, reject_combined, \
    binary_classification_metric, _epochs_to_samples_eeg_pupil
from renaanalysis.utils.utils import get_item_events, visualize_eeg_epochs, visualize_pupil_epochs
from renaanalysis.utils.viz_utils import visualize_block_gaze_event
from scipy.stats import stats
from sklearn.metrics import confusion_matrix

from physiolabxr.scripting.Examples.RenaProcessingParameters import locking_filters, event_names, epoch_margin
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.shared import bcolors
from physiolabxr.utils.data_utils import get_date_string, mode_by_column
from physiolabxr.utils.buffers import DataBuffer

condition_name_dict = {1: "RSVP", 2: "Carousel", 3: "Visual Search", 4: "Table Search"}
metablock_name_dict = {5: "Classifier Prep", 6: "Identifier Prep"}
colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}

feedback_mode = 'weights'

is_debugging = True
is_simulating_predictions = False
is_simulating_eeg = True
end_of_block_wait_time_in_simulate = 5
num_item_perblock = 30
num_vs_to_train_in_classifier_prep = 2  # for a total of 8 VS blocks in each metablock
num_vs_to_train_in_identifier_prep = 2  # for a total of 8 VS blocks in each metablock

ar_cv_folds = 3

target_threshold_quantile = 0.75

# class ItemEvent():
#     """
#     event types can be
#     1. pop in rsvp
#     2. rotation in carousel
#     3. gaze in any condition
#     4. grab in table search
#
#     duration is a two-item tuple consisting the onset and offset time centered around the event onset
#     """
#     def __init__(self, data, dtn, event_type, duraiton, item_id, item_distance, ):  # TODO
#         pass


class RenaProcessing(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function

        When the meta block changes
        cp -> cp : train the model, report accuracy and collect more data from the next cp metablock
        cp -> ip: train and exp

        always remove the marker that has been processed from the input buffer
        """
        super().__init__(*args, **kwargs)

        self.num_vs_before_training = None
        self.last_block_end_timestamp = None
        self.end_of_block_wait_start = None
        mne.use_log_level(False)
        self.current_metablock = None
        self.meta_block_counter = 0
        self.current_condition = None
        self.current_block_id = None
        self.event_marker_channels = json.load(open("../_presets/LSLPresets/ReNaEventMarker.json", 'r'))["ChannelNames"]
        self.loop_count = 0

        self.item_events = []
        self.item_events_queue = []  # the items (timestamps and metainfo of an event) in this queue is emptied to self.item events, when its data is available

        self.cur_state = 'idle'  # states: idle, recording, (training model), predicting
        self.num_meta_blocks_before_training = 1

        self.next_state = 'idle'

        self.event_marker_head = 0
        self.prediction_feedback_head = 0

        self.vs_block_counter = 0
        self.locking_data = {}
        self.is_inferencing = False
        self.event_ids = None

        self.end_of_block_waited = None
        self.end_of_block_wait_time = 3.5

        self.PCAs = {}
        self.ICAs = {}
        self.ARs = {}
        self.predicted_targets_for_metablock = []
        self.models = {}
        self.models_accs = {}

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.block_reports = defaultdict(dict)

        # outlet for sending predictions
        # RenaPrediction channels: [blockID, isSkipping, predictedTargetItemID] + num_item_perblock for each block item
        info = StreamInfo('RenaPrediction', 'Prediction', num_item_perblock * 2 + 3, pylsl.IRREGULAR_RATE, 'float32', 'RenaFeedbackID1234')
        self.prediction_outlet = StreamOutlet(info)

        # self.running_mode_of_predicted_target = []
        self.predicted_target_index_id_dict = None  # initialized at every new metablock in the main loop

        self.predicted_block_dtn_dict = None
        self.this_block_data_pending_feedback = None
        self.identifier_block_is_training_now = False
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        try:
            # send = np.random.random(3 + num_item_perblock)
            # self.prediction_outlet.push_sample(send)

            self.loop_count += 1
            if 'Unity.ReNa.EventMarkers' in self.inputs.buffer.keys():  # if we receive event markers
                if self.cur_state != 'endOfBlockWaiting':  # will not process new event marker if waiting
                    new_block_id, new_meta_block, new_condition, is_block_end, event_timestamp = self.get_block_update()
                else:
                    new_block_id, new_meta_block, new_condition, is_block_end, event_timestamp = [None] * 5

                if self.cur_state == 'idle':
                    if is_block_end and self.current_metablock is not None:
                        print(f'[{self.loop_count}] System is idle when received block_end, this probably means the last loop took too long')
                        self.next_state = 'endOfBlockWaiting'
                        # if is_simulating_predictions:
                        #     self.next_state = 'endOfBlockWaiting'
                        # else:
                        #     if np.max(self.inputs['Unity.VarjoEyeTrackingComplete'][1]) - event_timestamp > tmax_pupil + epoch_margin:
                        #         print(f'[{self.loop_count}] Eyetracking data has progressed beyond margin, next state will be processing')
                        #         self.next_state = 'endOfBlockProcessing'
                        #     else:
                        #         print(f'[{self.loop_count}] Eyetracking data NOT passed tmax margin, next state will be waiting')
                        #         self.next_state = 'endOfBlockWaiting'
                    if new_meta_block:
                        print(f"[{self.loop_count}] in idle, find new meta block, metablock counter = {self.meta_block_counter}")
                        self.vs_block_counter = 0  # reset the counter for metablock
                        self.predicted_target_index_id_dict = defaultdict(float)
                        # self.running_mode_of_predicted_target = []  # reset running mode of predicted target

                        if new_meta_block == 5:
                            self.num_vs_before_training = num_vs_to_train_in_classifier_prep
                            print(f'[{self.loop_count}] entering classifier prep, num visual search blocks for training will be {num_vs_to_train_in_classifier_prep}')
                        elif new_meta_block == 6:
                            self.num_vs_before_training = num_vs_to_train_in_identifier_prep
                            print(f'[{self.loop_count}] entering identifier and performance evaluation')

                    if new_block_id and self.current_metablock:  # only record if we are in a metablock, this is to ignore the practice
                        print(f"[{self.loop_count}] in idle, find new block id {self.current_block_id}, entering in_block")
                        self.next_state = 'in_block'
                elif self.cur_state == 'in_block':
                    # print('Updating buffer')
                    # self.data_buffer.update_buffers(self.inputs.buffer)
                    if is_block_end:
                        self.next_state = 'endOfBlockWaiting'
                        self.end_of_block_wait_start = time.time()
                elif self.cur_state == 'endOfBlockWaiting':
                    self.end_of_block_waited = time.time() - self.end_of_block_wait_start
                    print(f"[{self.loop_count}] end of block waited {self.end_of_block_waited}")
                    # if self.end_of_block_waited > self.end_of_block_wait_time:
                    if is_simulating_predictions:
                        if self.end_of_block_waited > end_of_block_wait_time_in_simulate:
                            self.next_state = 'endOfBlockProcessing'
                    else:
                        if np.max(self.inputs['Unity.VarjoEyeTrackingComplete'][1]) > self.last_block_end_timestamp + tmax_pupil + epoch_margin:
                            self.next_state = 'endOfBlockProcessing'
                elif self.cur_state == 'endOfBlockProcessing':
                    if self.current_metablock == 5:
                        self.next_state = self.classifier_prep_phase_end_of_block()
                    elif self.current_metablock == 6:
                        self.next_state = self.identifier_prep_phase_end_of_block()
                    else:
                        print(f'[{self.loop_count}] block ended on while in meta block {self.current_metablock}. Skipping end of block processing. Not likely to happen.')
                elif self.cur_state == 'waitingFeedback':
                    self.next_state = self.receive_prediction_feedback()
                    pass

            if self.next_state != self.cur_state:
                print(f'[{self.loop_count}] updating state from {self.cur_state} to {self.next_state}')
                self.cur_state = self.next_state
            # TODO check if there's any request queued
        except Exception as e:
            print(e)
        # print(f"[{self.loop_count}] End of loop ")

    # cleanup is called when the stop button is hit

    def receive_prediction_feedback(self):
        try:
            if 'Unity.ReNa.PredictionFeedback' in self.inputs.keys() and len(self.inputs['Unity.ReNa.PredictionFeedback'][1]) - self.prediction_feedback_head > 0:  # there's new event marker
                if self.predicted_block_dtn_dict is not None:
                    timestamp = self.inputs['Unity.ReNa.PredictionFeedback'][1][self.prediction_feedback_head]
                    feedbacks = self.inputs['Unity.ReNa.PredictionFeedback'][0][:, self.prediction_feedback_head]
                    self.prediction_feedback_head += 1
                    for locking_name in locking_filters.keys():
                        if locking_name in self.predicted_block_dtn_dict.keys():
                            y_pending_feedback = np.copy(self.this_block_data_pending_feedback[locking_name][1])  # get the y array
                            item_indices_pending_feedbacks = [g.item_index for g in self.this_block_data_pending_feedback[locking_name][4]]
                            item_ids_pending_feedbacks = [g.item_id for g in self.this_block_data_pending_feedback[locking_name][4]]
                            for i, feedback_item_dtn in enumerate(feedbacks):
                                # if feedback_item_dtn == 1 and self.predicted_block_dtn_dict[locking_name][i] == 2: # target got flipped to a distractor:
                                # try:
                                #     assert y_pending_feedback[item_indices_pending_feedbacks.index(i)] == 1  # y pred must be 1: target
                                # except AssertionError as e:
                                #     print(f"[{self.loop_count}] ReceivePredictionFeedback: predicted dtn not much y in the block data pending feedback. " + str(e))
                                #     raise e
                                if feedback_item_dtn != 0 and i in item_indices_pending_feedbacks:
                                    if y_pending_feedback[item_indices_pending_feedbacks.index(i)] != feedback_item_dtn - 1:
                                        print(f"[{self.loop_count}] ReceivePredictionFeedback: locking {locking_name}: feedback changed item id {item_ids_pending_feedbacks[item_indices_pending_feedbacks.index(i)]} at index {i} from {y_pending_feedback[item_indices_pending_feedbacks.index(i)]} to {feedback_item_dtn - 1}")
                                    y_pending_feedback[item_indices_pending_feedbacks.index(i)] = feedback_item_dtn - 1
                            if np.any(self.this_block_data_pending_feedback[locking_name][1] != y_pending_feedback):
                                print(f"[{self.loop_count}] ReceivePredictionFeedback: locking {locking_name}: before y is {self.this_block_data_pending_feedback[locking_name][1]}, after is {y_pending_feedback}, for items {item_ids_pending_feedbacks}, at indices {item_indices_pending_feedbacks}")
                                try:
                                    assert np.all(self.this_block_data_pending_feedback[locking_name][2].events[:, 2] == self.this_block_data_pending_feedback[locking_name][3].events[:, 2])
                                    assert np.all(self.this_block_data_pending_feedback[locking_name][3].events[:, 2] == self.this_block_data_pending_feedback[locking_name][1] + 1)
                                except AssertionError:
                                    print(f'[{self.loop_count}] ReceivePredictionFeedback: epoch events do not match y')
                                self.this_block_data_pending_feedback[locking_name][2].events[:, 2] = y_pending_feedback + 1  # update the epoch DTN
                                self.this_block_data_pending_feedback[locking_name][3].events[:, 2] = y_pending_feedback + 1  # update the epoch DTN
                                self.this_block_data_pending_feedback[locking_name][1] = y_pending_feedback
                            else:
                                print(f"[{self.loop_count}] ReceivePredictionFeedback: locking {locking_name}: no y is changed. y is {self.this_block_data_pending_feedback[locking_name][1]}, for items {item_ids_pending_feedbacks}, at indices {item_indices_pending_feedbacks}")
                    self.add_block_data_all_lockings(self.this_block_data_pending_feedback)

                if self.identifier_block_is_training_now:
                    self.train_identification_model()  # the next VS block will probably have wait here, if it ends before this function (training) returns
                return 'idle'
            else:
                return 'waitingFeedback'
        except Exception as e:
            print(f"[{self.loop_count}] ReceivePredictionFeedback: found exception." + str(e))
            raise e

    def classifier_prep_phase_end_of_block(self):
        if self.current_condition == conditions['VS']:
            self.vs_block_counter += 1
            print(f"[{self.loop_count}] ClassifierPrepEndOfBlockProcessing: Incrementing VS block counter to {self.vs_block_counter}")
            try:
                if self.vs_block_counter == num_vs_to_train_in_classifier_prep:  # time to train the model and identify target for this block
                    self.send_skip_prediction()  # epoching the recorded block data
                    self.add_block_data()
                    self.train_identification_model()  # the next VS block will probably have wait here
                else:  # we are still accumulating data
                    self.send_skip_prediction()
                    self.add_block_data()
            except Exception as e:
                print(f"[{self.loop_count}] ClassifierPrepEndOfBlockProcessing: Exception in end-of-block processing with vs counter value {self.vs_block_counter}: ")
                print(e)
        else:
            print(f"[{self.loop_count}] ClassifierPrepEndOfBlockProcessing: not VS block, current condition is {self.current_condition }, skipping")
        return 'idle'

    def identifier_prep_phase_end_of_block(self):
        if self.current_condition == conditions['VS']:
            self.vs_block_counter += 1
            print(f"[{self.loop_count}] IdentifierPrepEndOfBlockProcessing: Incrementing VS block counter to {self.vs_block_counter}")
            try:
                if self.vs_block_counter == self.num_vs_before_training:  # time to train the model and identify target for this block
                    print(f"[{self.loop_count}] IdentifierPrepEndOfBlockProcessing: counter is equal to num blocks before training. Start training with feedback from user and resetting training block counter.")
                    self.vs_block_counter = 0
                    self.this_block_data_pending_feedback = self.add_block_data(append_data=False)
                    self.identifier_block_is_training_now = True
                    # Don't train until have feedback
                else:  # we are not training yet
                    self.this_block_data_pending_feedback = self.add_block_data(append_data=False)  # epoching the recorded block data
                    self.identifier_block_is_training_now = False
                if self.this_block_data_pending_feedback is None:
                    print(f"[{self.loop_count}] IdentifierEndOfBlockProcessing: received this block data as none, probably due to NaN (please check if it's not because of NaN), skipping")
                    self.send_skip_prediction()
                    return 'idle'
                self.predicted_block_dtn_dict = self.target_identification(self.this_block_data_pending_feedback)  # identify target for this block, this will send the identification result

            except Exception as e:
                print(f"[{self.loop_count}]IdentifierPrepEndOfBlockProcessing: Exception in end-of-block processing with vs counter value {self.vs_block_counter}: ")
                print(e)
            return 'waitingFeedback'
        else:
            print(f"[{self.loop_count}] IdentifierEndOfBlockProcessing: not VS block, current condition is {self.current_condition }, skipping")
            return 'idle'

    def cleanup(self):
        print('Cleanup function is called')

    def add_block_data(self, append_data=True):
        try:
            if is_simulating_predictions:
                return dict([(locking_name, [None,  np.random.randint(0, 3, size=num_item_perblock), None, None, None]) for locking_name, _ in locking_filters.items()])

            print(f'[{self.loop_count}] AddingBlockData: Adding block data with id {self.current_block_id} of condition {self.current_condition}')
            rdf = RenaDataFrame()
            data = copy.deepcopy(self.inputs.buffer)  # deep copy the data so our data doesn't get changed
            events = get_item_events(data['Unity.ReNa.EventMarkers'][0], data['Unity.ReNa.EventMarkers'][1], data['Unity.ReNa.ItemMarkers'][0], data['Unity.ReNa.ItemMarkers'][1])
            events += gaze_event_detection_I_DT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])  # TODO no need to resample the headtracking data again
            events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])
            if 'FixationDetection' in data.keys():
                events += gaze_event_detection_PatchSim(data['FixationDetection'][0], data['FixationDetection'][1], events)
            else:
                print(f"[{self.loop_count}] AddingBlockData: WARNING: not FixationDetection stream when trying to add block data")
            rdf.add_participant_session(data, events, '0', 0, None, None, None)
            visualize_block_gaze_event(rdf, participant='0', session=0, block_id=self.current_block_id, generate_video=False, video_fix_alg=None)
            try:
                rdf.preprocess(is_running_ica=True, n_jobs=1, ocular_artifact_mode='proxy')
                if is_simulating_eeg:  # add in fake EEG data
                    rdf.simulate_exg()
            except Exception as e:
                print(f"{bcolors.WARNING}Encountered value error when preprocessing rdf: {str(e)}{bcolors.ENDC}")
                return None
            this_locking_data = {}
            for locking_name, event_filters in locking_filters.items():
                if 'VS' in locking_name:  # TODO only care about VS conditions for now
                    print(f"[{self.loop_count}] AddingBlockData: Finding epochs samples on locking {locking_name}")
                    # if is_debugging: viz_eeg_epochs(rdf, event_names, event_filters, colors, title=f'Block ID {self.current_block_id}, Condition {self.current_condition}, MetaBlock {self.current_metablock}', n_jobs=1)

                    x, y, epochs, event_ids = epochs_to_class_samples(rdf, event_names, event_filters, data_type='both', n_jobs=1, reject=None, plots='full', colors=colors, title=f'{locking_name}, block {self.current_block_id}, condition {self.current_condition}, metablock {self.current_metablock}')
                    if x is None:
                        print(f"{bcolors.WARNING}[{self.loop_count}] AddingBlockData: No event found for locking {locking_name}{bcolors.ENDC}")
                        continue
                    if len(event_ids) == 2:
                        if self.event_ids is None:
                            self.event_ids = event_ids
                    else:
                        # print(f'[{self.loop_count}] AddingBlockData: only found one event {event_ids}, skipping adding epoch')
                        print(f'{bcolors.WARNING}[{self.loop_count}] AddingBlockData: {locking_name} only found one event {event_ids}{bcolors.ENDC}')
                        # continue
                    epoch_events = get_events(event_filters, events, order='time')
                    try:
                        assert np.all(np.array([x.dtn for x in epoch_events])-1 == y)
                    except AssertionError as e:
                        print(f"[{self.loop_count}] AddingBlockData: add_block_data: epoch block events is different from y")
                        raise e

                    target_item_id_count = len(np.unique([e.item_id for e in epoch_events if e.dtn==2.0]))
                    try:
                        assert target_item_id_count == 1 or target_item_id_count == 0
                    except AssertionError as e:
                        print(f"[{self.loop_count}] AddingBlockData: true target item ids not all equal, this should NEVER happen!")
                        raise e

                    if append_data:
                        self._add_block_data_to_locking(locking_name, x, y, epochs[0], epochs[1], epoch_events)
                        print(f"[{self.loop_count}] AddingBlockData: Add {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                        print(f'[{self.loop_count}] AddingBlockData: {locking_name} Has {np.sum(self.locking_data[locking_name][1] == 0)} distractors and {np.sum(self.locking_data[locking_name][1] == 1)} targets')
                    else:
                        print(f"[{self.loop_count}] AddingBlockData: find but not adding {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                    this_locking_data[locking_name] = [x, y, epochs[0], epochs[1], epoch_events]

            print(f"[{self.loop_count}] AddingBlockData: Process completed")
            self.clear_buffer()
            return this_locking_data
        except Exception as e:
            print(f"[{self.loop_count}]AddBlockData: exception when adding block data: " + str(e))
            raise e

    def add_block_data_all_lockings(self, this_block_data: dict):
        try:
            for locking_name, event_filters in locking_filters.items():
                if 'VS' in locking_name:
                    if locking_name in this_block_data.keys():
                        y = this_block_data[locking_name][1]
                        if locking_name in this_block_data.keys():
                            print(f"[{self.loop_count}] AddingBlockDataPostHoc: Add {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                            self._add_block_data_to_locking(locking_name, *this_block_data[locking_name])
                            print(f'[{self.loop_count}] AddingBlockData: {locking_name} Has {np.sum(self.locking_data[locking_name][1] == 0)} distractors and {np.sum(self.locking_data[locking_name][1] == 1)} targets')
                        else:
                            print(f"[{self.loop_count}] AddingBlockDataPostHoc: find but not adding {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                    else:
                        print(f'[{self.loop_count}] AddingBlockDataPostHoc: no data is available for {locking_name}')
        except Exception as e:
            raise e

    def _add_block_data_to_locking(self, locking_name, x, y, epochs_eeg, epochs_pupil, epoch_events):
        if locking_name not in self.locking_data.keys():
            self.locking_data[locking_name] = [x, y, epochs_eeg, epochs_pupil, epoch_events]
        else:  # append the data
            self.locking_data[locking_name][0][0] = np.concatenate([self.locking_data[locking_name][0][0], x[0]], axis=0)  # append EEG data
            self.locking_data[locking_name][0][1] = np.concatenate([self.locking_data[locking_name][0][1], x[1]], axis=0)  # append pupil data
            try:
                self.locking_data[locking_name][2] = mne.concatenate_epochs([self.locking_data[locking_name][2], epochs_eeg])  # append EEG epochs
            except ValueError as e:
                if str(e) == 'Epochs must have same times':
                    self.locking_data[locking_name][2] = concatenate_as_epochArray([self.locking_data[locking_name][2], epochs_eeg])
                else:
                    raise e
            try:
                self.locking_data[locking_name][3] = mne.concatenate_epochs([self.locking_data[locking_name][3], epochs_pupil])  # append pupil epochs
            except ValueError as e:
                if str(e) == 'Epochs must have same times':
                    self.locking_data[locking_name][3] = concatenate_as_epochArray([self.locking_data[locking_name][2], epochs_eeg])
                else:
                    raise e
            self.locking_data[locking_name][1] = np.concatenate([self.locking_data[locking_name][1], y], axis=0)  # append labels
            self.locking_data[locking_name][4] += epoch_events  # append labels

    def clear_buffer(self):
        print(f"[{self.loop_count}] Buffer cleared")
        self.event_marker_head = 0
        self.inputs.clear_up_to(self.last_block_end_timestamp, ignores=['Unity.ReNa.PredictionFeedback'])

    # def check_pupil_data_complete_epoch(self, start_time):
    #     return (np.max(self.inputs['Unity.VarjoEyeTrackingComplete'][1]) - start_time) > (tmax_pupil + epoch_margin)

    def train_identification_model(self):
        if is_simulating_predictions:
            print(f'[{self.loop_count}] TrainIdentificationModel: in simulation, skipping output.')
            return
        try:
            training_start_time = time.time()
            for locking_name, (_, y, epochs_eeg, epochs_pupil, _) in self.locking_data.items():
                print(f'[{self.loop_count}] TrainIdentificationModel: Training models for {locking_name}')
                try:
                    x_eeg, x_pupil, y, self.PCAs[locking_name], self.ICAs[locking_name], self.ARs[locking_name], _ = self.preprocess_block_data(epochs_eeg, epochs_pupil)
                except Exception as e:
                    print(f'{bcolors.WARNING}[{self.loop_count}] TrainIdentificationModel: {locking_name}: error preprocessing block data, skipping. y is {y}. {str(e)}{bcolors.ENDC}')
                    continue
                if len(np.unique(y)) == 1:
                    print(f'{bcolors.WARNING}[{self.loop_count}] TrainIdentificationModel: y for {locking_name} only has class, skipping training. y is {y}{bcolors.ENDC}')
                    continue
                model = EEGPupilCNN(eeg_in_shape=x_eeg.shape, pupil_in_shape=x_pupil.shape, num_classes=2, eeg_in_channels=x_eeg.shape[1])
                model, training_histories, criterion, label_encoder = train_model_pupil_eeg_no_folds([x_eeg, x_pupil], y, model, num_epochs=20, test_name='realtime')
                best_train_acc = np.max(training_histories['train accs'])
                print(f'[{self.loop_count}] TrainIdentificationModel: {locking_name} (with {np.sum(y==0)} distractors and {np.sum(y==1)} targets) gives classification accuracy: {best_train_acc}')
                self.models[locking_name] = model
                self.models_accs[locking_name] = best_train_acc
            print(f'[{self.loop_count}] TrainIdentificationModel: training complete, took {time.time() - training_start_time} seconds')
        except Exception as e:
            raise e

    def target_identification(self, this_block_data):
        try:
            if is_simulating_predictions:
                dummy_predictions = self.send_dummy_prediction()
                return dummy_predictions
            prediction_start_time = time.time()
            predicted_target_ids_lockings_dict = defaultdict(lambda: defaultdict(int))
            item_predictions_dict = {}
            for locking_name, ((_, _), y, epochs_eeg, epochs_pupil, block_events) in this_block_data.items():
                # (x_eeg, x_pupil), y, rejects = reject_combined(epochs_pupil, epochs_eeg, self.event_ids,  n_jobs=1)  # NOT auto reject
                # block_events_cleaned = np.array(block_events)[rejects]
                print(f"[{self.loop_count}] TargetIdentification:  Locking {locking_name} Has {len(y)} epochs, with {np.sum(y==1)} targets and {np.sum(y==0)} distractors")
                try:
                    assert locking_name in self.PCAs.keys() and locking_name in self.ICAs.keys() and locking_name in self.ARs.keys()
                except AssertionError:
                    print(f'{bcolors.WARNING}[{self.loop_count}] TargetIdentification: locking {locking_name} does not have preprocess transforms. Skippinp. Likely this data was not present in entire training data, which should not happen normally.{bcolors.ENDC}')
                    continue
                try:
                    assert locking_name in self.models.keys()
                except AssertionError:
                    print(f'{bcolors.WARNING}[{self.loop_count}] TargetIdentification: locking {locking_name} does not have model. Skippinp. Likely this data only have one class: in the locking data, y is {self.locking_data[locking_name][1]}{bcolors.ENDC}')
                    continue
                # noinspection PyBroadException
                try:
                    x_eeg_reduced, x_pupil, y, _, _, _, rejections = self.preprocess_block_data(epochs_eeg, epochs_pupil)
                except Exception as e:
                    print(f'{bcolors.WARNING}[{self.loop_count}] TargetIdentification: {locking_name}: error preprocessing block data, skipping. y is {y}. {str(e)}{bcolors.ENDC}')
                    continue
                if x_eeg_reduced is None or np.all(y == 0):  # if no data or no target fixations
                    print(f'{bcolors.WARNING}[{self.loop_count}] TargetIdentification: {np.sum(np.all(y == 2))} target epochs remains after rejection. Skipping target identification for {locking_name}.{bcolors.ENDC}')
                    self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['accuracy'] = -1
                    self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['target sensitivity'] = -1
                    self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['target specificity'] = -1
                    continue

                x_eeg_tensor = torch.Tensor(x_eeg_reduced).to(self.device)
                x_pupil_tensor = torch.Tensor(x_pupil).to(self.device)

                self.models[locking_name].eval()
                with torch.no_grad():
                    y_pred = self.models[locking_name]([x_eeg_tensor, x_pupil_tensor])
                    confidences = np.max(y_pred.detach().cpu().numpy(), axis=1)
                    prediction = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

                acc, tpr, tnr = binary_classification_metric(y_true=y, y_pred=prediction)
                if len(tpr) == 1:
                    print(f'{bcolors.WARNING}[{self.loop_count}] TargetIdentification: {locking_name} found only one class for tpr/tnr: y={y}, prediction={prediction}, tpr={tpr}, tnr={tnr}.{bcolors.ENDC}')
                    target_sensitivity = tpr[0]
                    target_specificity = tnr[0]
                else:
                    target_sensitivity = tpr[1]
                    target_specificity = tnr[1]

                cleaned_block_events = np.array(block_events)[rejections]
                target_index_id_predicted_prob = [(x.item_index, x.item_id, prob) for x, prob in zip(cleaned_block_events[prediction == 1], confidences)]
                predicted_item_indices = [x.item_index for x in cleaned_block_events]
                true_target_item_ids = [x.item_id for x in cleaned_block_events[y == 1]]

                # if len(predicted_target_item_ids) == 0 == None:
                #     predicted_target_ids_lockings_dict[locking_name] = None
                # else:
                try:
                    voted_target = stats.mode([x.item_id for x in cleaned_block_events[prediction == 1]]).mode[0]
                except IndexError:
                    voted_target = -1
                    print(f"[{self.loop_count}] TargetIdentification: no target is predicted with locking {locking_name}: predictions are: {prediction}")
                for target_item_index, target_item_id, confidence in target_index_id_predicted_prob:
                    predicted_target_ids_lockings_dict[locking_name][target_item_id] += confidence
                    self.predicted_target_index_id_dict[target_item_id] += confidence * self.models_accs[locking_name]  # weight by the accuracy of the model

                item_index_dtn_predictions = np.zeros(num_item_perblock)
                for item_index, predicted_dtn in zip(predicted_item_indices, prediction):
                    item_index_dtn_predictions[item_index] = predicted_dtn + 1  # plus one to convert between pred and dtn
                item_predictions_dict[locking_name] = item_index_dtn_predictions
                try:
                    assert len(np.unique(true_target_item_ids)) == 1
                except AssertionError as e:
                    print(f"[{self.loop_count}] TargetIdentification: true target item ids not all equal, this should NEVER happen!")
                    raise e
                self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['accuracy'] = acc
                self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['target sensitivity'] = target_sensitivity
                self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['target specificity'] = target_specificity
                print(f"[{self.loop_count}] TargetIdentification: Locking {locking_name} {np.sum(y==1)} accuracy = {acc}, target sensitivity (TPR) = {target_sensitivity}, target specificity (TNR) = {target_specificity}, true target is {true_target_item_ids[0]}, predicted target through mode is {voted_target}")
            print(f'[{self.loop_count}] TargetIdentification: prediction  complete, took {time.time() - prediction_start_time} seconds')
            pickle.dump(self.block_reports, open(f"{get_date_string()}_block_report", 'wb'))
            # TODO return the predicted class for each item, and the inferred target class

            if len(item_predictions_dict) != 0:
                # self.running_mode_of_predicted_target.append(predicted_target_ids_lockings_dict[selected_locking_name])
                block_target_pred_confidences = self.process_block_target_confidence(self.predicted_target_index_id_dict)

                predicts = np.array([predicts for locking_name, predicts in item_predictions_dict.items()])  # calculate modes
                # predicts_mode = stats.mode(predicts, axis=0).mode[0]
                predicts_mode = mode_by_column(predicts, ignore=0)
                # if
                #     print("[{self.loop_count}] TargetIdentification: no target is found for ALL lockings. Predictions are predicts, sending dummy prediction")
                #     dummy_predictions = self.send_dummy_prediction()
                #     return dummy_predictions
                self.send_prediction(-1, predicts_mode, block_target_pred_confidences)  # each value item_predictions_dict contains <num_item_perblock> items, they are the dtn prediction for each item in a block
                return item_predictions_dict
            else:
                print(f"[{self.loop_count}] TargetIdentification: no epochs available for this block: {item_predictions_dict}, sending dummy results.")
                dummy_predictions = self.send_dummy_prediction()
                return dummy_predictions
        except Exception as e:
            print(e)
            raise e

    def process_block_target_confidence(self, predicted_target_index_id_dict: dict):
        try:
            index_target_confidence = - np.ones(num_item_perblock)
            if len(predicted_target_index_id_dict):  # if no target is predicted
                return index_target_confidence
            all_probs = np.array(list(predicted_target_index_id_dict.values()))
            all_ids = np.array([item_id for item_id in predicted_target_index_id_dict.keys()])
            prob_threshold = np.quantile(all_probs, target_threshold_quantile)

            thresholded = [(item_id, prob) for item_id, prob in predicted_target_index_id_dict.items() if prob >= prob_threshold]
            thresholded.sort(key=lambda x: x[1], reverse=True)

            print(f"[{self.loop_count}] ProcessBlockTargetConfidence: process target quantiles: {target_threshold_quantile} give a threshold confidence of {prob_threshold} and {np.sum(all_probs > prob_threshold)} targets")
            print(f'with prob {thresholded}')

            try:
                assert len(thresholded) < num_item_perblock / 2
            except AssertionError:
                print(f'[{self.loop_count}] ProcessBlockTargetConfidence: more than half of <num item per block> ({num_item_perblock/2}) are predicted as target. Trimming to top {num_item_perblock/2} ones.')
                # thresholded = thresholded[:num_item_perblock/2]

            thresholded_probs = [prob for item_id, prob in predicted_target_index_id_dict.items() if prob > prob_threshold]
            if len(thresholded_probs) > 1:
                thresholded_normalized_prob = [(item_id, (prob - np.min(thresholded_probs)) / (np.max(thresholded_probs) - np.min(thresholded_probs))) for item_id, prob in predicted_target_index_id_dict.items() if prob > prob_threshold]
            else:
                thresholded_normalized_prob = [(item_id, 1) for item_id, prob in predicted_target_index_id_dict.items() if prob > prob_threshold]
            for i, (item_id, confidence) in enumerate(thresholded_normalized_prob):
                index_target_confidence[i] = item_id
                index_target_confidence[i + int(num_item_perblock / 2)] = confidence
            print(f"[{self.loop_count}] ProcessBlockTargetConfidence: find thresholded & normalized confidence: {thresholded_normalized_prob}, with indices index_target_confidence {index_target_confidence} or ids {all_ids}, and all probs {all_probs}")

            return index_target_confidence
        except Exception as e:
            print(f"[{self.loop_count}] ProcessBlockTargetConfidence: find exception: " + str(e))
            raise e

    def send_prediction(self, predicted_target_id, block_item_prediciton, item_index_pred_target_confidence):
        try:
            send = np.zeros(3 + num_item_perblock * 2)
            send[0] = self.current_block_id  # Unity will check the block ID matches
            send[2] = predicted_target_id  # set predicted target item id
            send[3:] = np.concatenate((block_item_prediciton, item_index_pred_target_confidence))
            self.prediction_outlet.push_sample(send)
            print("Prediction send successful")
        except Exception as e:
            raise e

    def send_dummy_prediction(self):
        try:
            send = np.zeros(3 + num_item_perblock * 2)
            send[0] = self.current_block_id  # Unity will check the block ID matches
            send[2] = -1  # set predicted target item id
            send[3:] = np.random.randint(1, 3, size=num_item_perblock * 2)
            self.prediction_outlet.push_sample(send)
            print("Dummy prediction send successful")
        except Exception as e:
            raise e

    def send_skip_prediction(self):
        try:
            send = np.zeros(3 + 2 * num_item_perblock)
            send[0] = self.current_block_id  # Unity will check the block ID matches
            send[1] = 1  # set the skip (second value to 1)
            self.prediction_outlet.push_sample(send)
            print("Skip prediction sent")
        except Exception as e:
            raise e

    def preprocess_block_data(self, epochs_eeg, epochs_pupil, pca=None, ica=None, ar=None):
        if len(epochs_eeg) > 1:
            try:
                x_eeg, x_pupil, y, ar, rejections = reject_combined(epochs_pupil, epochs_eeg, self.event_ids, n_jobs=1, n_folds=ar_cv_folds, ar=ar, return_rejections=True)  # apply auto reject
            except ValueError as e:
                print(f"[{self.loop_count}] preprocess_block_data: error in rejection, most likely the there are too few samples of fixations for it folds")
                raise e
        elif len(epochs_eeg) == 1:
            print(f"[{self.loop_count}] preprocess_block_data: only find 1 eeg epoch, skip rejection")
            x_eeg, x_pupil, y = _epochs_to_samples_eeg_pupil(epochs_pupil, epochs_eeg, self.event_ids)
            ar, rejections = [None] * 2
        else:
            raise ValueError(f"[{self.loop_count}] preprocess_block_data: zero epoch is given to preprocess_block_data")
        print(f'[{self.loop_count}] target_identification: {len(epochs_eeg) - len(x_eeg)} epochs were auto rejected. Now with {np.sum(y == 1)} targets and {np.sum(y == 0)} distractors')
        if len(y) == 0:  # no data remains after rejection
            return [None] * 6

        x_eeg, pca, ica = compute_pca_ica(x_eeg, n_components=20, pca=pca, ica=ica)
        return x_eeg, x_pupil, y, pca, ica, ar, rejections

    def get_block_update(self):
        try:
            # there cannot be multiple condition updates in one block update, which runs once a second
            is_block_end = False
            new_block_id = None
            new_meta_block = None
            new_condition = None
            this_event_timestamp = None
            if len(self.inputs['Unity.ReNa.EventMarkers'][1]) - self.event_marker_head > 0:  # there's new event marker
                this_event_timestamp = self.inputs['Unity.ReNa.EventMarkers'][1][self.event_marker_head]
                block_id_start_end = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("BlockIDStartEnd"), self.event_marker_head]
                block_marker = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("BlockMarker"), self.event_marker_head]
                self.event_marker_head += 1

                if block_id_start_end > 0:  # if there's a new block id
                    new_block_id = None if block_id_start_end == 0 or block_id_start_end < 0 else block_id_start_end  # block id less than 0 is also when the block ends
                    new_condition = block_marker if block_marker in condition_name_dict.keys() else None
                elif block_id_start_end < 0:  # if this is an end of a block
                    try:
                        assert block_id_start_end == -self.current_block_id
                    except AssertionError:
                        raise Exception(f"[{self.loop()}] get_block_update: Did not receive block end signal. This block end {block_id_start_end}. Current block id {self.current_block_id}")
                    try:
                        assert self.current_block_id is not None
                    except AssertionError:
                        raise Exception(f"[{self.loop()}] get_block_update: self.current_block_id is None when block_end signal ({block_id_start_end}) comes, that means a block start is never received")
                    print("[{0}] get_block_update: Block with ID {1} ended. ".format(self.loop_count, self.current_block_id))
                    # self.current_block_id = None  # IMPORTANT, current_block_id will retain its value until new block id is received
                    self.last_block_end_timestamp = this_event_timestamp
                    is_block_end = True
                else:  # the only other possibility is that this is a meta block marker
                    new_meta_block = block_marker if block_marker in metablock_name_dict.keys() else None

            if new_meta_block:
                # self.current_block_id = new_block_id
                self.meta_block_counter += 1
                print("[{0}] get_block_update: Entering new META block {1}, metablock count is {2}".format(self.loop_count, metablock_name_dict[new_meta_block], self.meta_block_counter))
                self.current_metablock = new_meta_block
                # self.inputs.clear_buffer()  # should clear buffer at every metablock, so we don't need to deal with practice round data
            if new_block_id:
                self.current_block_id = new_block_id
                # message = "[{0}] Entering new block with ID {1}".format(self.loop_count, self.current_block_id) + \
                # ". No metablock is given, assuming in practice rounds." if not self.current_metablock else ""
                print("[{0}] get_block_update: Entering new block with ID {1}".format(self.loop_count, self.current_block_id) + (". No metablock is given, assuming in practice rounds." if not self.current_metablock else ""))
            if new_condition:
                self.current_block_id = new_block_id
                self.current_condition = new_condition
                print("[{0}] get_block_update: Block {2} is setting current condition to {1}".format(self.loop_count, condition_name_dict[new_condition], self.current_block_id))
            return new_block_id, new_meta_block, new_condition, is_block_end, this_event_timestamp  # tells if there's new block and meta block
        except Exception as e:
            raise e
    # def get_item_event(self):
    #     for i in range(len(self.inputs['Unity.ReNa.EventMarkers'][1])):
    #         block_id_start_end = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("BlockIDStartEnd"), i]


def concatenate_as_epochArray(epochs_array):
    """
    Concatenate epochs_eeg and epochs_pupil into one EpochArray
    :param epochs_eeg: EpochArray
    :param epochs_pupil: EpochArray
    :param event_ids: dict
    :return: EpochArray
    """
    arrays = np.concatenate([x.get_data() for x in epochs_array], axis=0)
    event_arrays = np.concatenate([x.events for x in epochs_array], axis=0)

    epochs_concatenated = EpochsArray(arrays, epochs_array[0].info, events=event_arrays, event_id=epochs_array[0].event_id)
    return epochs_concatenated