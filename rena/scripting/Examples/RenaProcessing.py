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
from pylsl import StreamInfo, StreamOutlet, pylsl
from renaanalysis.eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim
from renaanalysis.learning.models import EEGPupilCNN
from renaanalysis.learning.train import train_model_pupil_eeg, train_model_pupil_eeg_no_folds
from renaanalysis.params.params import conditions, dtnn_types, tmax_pupil, random_seed
from renaanalysis.utils.Event import get_events
from renaanalysis.utils.RenaDataFrame import RenaDataFrame
from renaanalysis.utils.data_utils import epochs_to_class_samples, compute_pca_ica, reject_combined, \
    binary_classification_metric
from renaanalysis.utils.utils import get_item_events, viz_eeg_epochs
from scipy.stats import stats
from sklearn.metrics import confusion_matrix

from rena.scripting.Examples.RenaProcessingParameters import locking_filters, event_names, epoch_margin
from rena.scripting.RenaScript import RenaScript
from rena.utils.data_utils import get_date_string
from rena.utils.general import DataBuffer

condition_name_dict = {1: "RSVP", 2: "Carousel", 3: "Visual Search", 4: "Table Search"}
metablock_name_dict = {5: "Classifier Prep", 6: "Identifier Prep"}

is_debugging = True
is_simulating_predictions = False
end_of_block_wait_time_in_simulate = 5
num_item_perblock = 30
selected_locking_name = 'VS-FLGI'
num_vs_to_train_in_classifier_prep = 8  # for a total of 8 VS blocks in each metablock
num_vs_to_train_in_identifier_prep = 3  # for a total of 8 VS blocks in each metablock

ar_cv_folds = 3

target_threshold_quantile = 0.75

class ItemEvent():
    """
    event types can be
    1. pop in rsvp
    2. rotation in carousel
    3. gaze in any condition
    4. grab in table search

    duration is a two-item tuple consisting the onset and offset time centered around the event onset
    """
    def __init__(self, data, dtn, event_type, duraiton, item_id, item_distance, ):  # TODO
        pass


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

        self.last_block_end_timestamp = None
        self.end_of_block_wait_start = None
        mne.use_log_level(False)
        self.current_metablock = None
        self.meta_block_counter = 0
        self.current_condition = None
        self.current_block_id = None
        self.event_marker_channels = json.load(open("../Presets/LSLPresets/ReNaEventMarker.json", 'r'))["ChannelNames"]
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

        self.predicted_block_dtn = None
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
                            print(f'[{self.loop_count}] entering classifier prep, num visual search blocks for training will be {self.num_vs_before_training}')
                        elif new_meta_block == 6:
                            self.num_vs_before_training = num_vs_to_train_in_identifier_prep
                            print(f'[{self.loop_count}] entering identifier and performance evaluation, num visual search blocks for training will be {self.num_vs_before_training}')

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
            if 'Unity.ReNa.PredictionFeedback' in self.inputs.keys() and len(self.inputs['Unity.ReNa.PredictionFeedback'][1]) - self.prediction_feedback_head > 0: # there's new event marker
                timestamp = self.inputs['Unity.ReNa.PredictionFeedback'][1][self.prediction_feedback_head]
                feedbacks = self.inputs['Unity.ReNa.PredictionFeedback'][0][:, self.prediction_feedback_head]
                self.prediction_feedback_head += 1

                flipped_count = 0
                for i, (predicted_item_dtn, feedback_item_dtn) in enumerate(zip(self.predicted_block_dtn, feedbacks)):
                    if feedback_item_dtn == 1 and predicted_item_dtn == 2: # target got flipped to a distractor:
                        flipped_count += 1
                        for locking_name, _ in locking_filters.items():
                            y = np.copy(self.this_block_data_pending_feedback[locking_name][1])
                            try:
                                assert y == 1  # y pred must be 1: target
                            except AssertionError as e:
                                print("predicted dtn not much y in the block data pending feedback. " + str(e))
                            y[i] = 0
                            self.this_block_data_pending_feedback[locking_name][1] = y

                self.all_block_data_all_lockings(self.this_block_data_pending_feedback)
                if self.identifier_block_is_training_now:
                    self.train_identification_model()  # the next VS block will probably have wait here, if it ends before this function (training) returns
                return 'idle'
            else:
                return 'waitingFeedback'
        except Exception as e:
            print(f"[{self.loop_count}] ReceivePredictionFeedback: found exception." + str(e))

    def classifier_prep_phase_end_of_block(self):
        if self.current_condition == conditions['VS']:
            self.vs_block_counter += 1
            print(f"[{self.loop_count}] ClassifierPrepEndOfBlockProcessing: Incrementing VS block counter to {self.vs_block_counter}")
            try:
                if self.vs_block_counter == self.num_vs_before_training:  # time to train the model and identify target for this block
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
            print(f"[{self.loop_count}] EndOfBlockProcessing: not VS block, current condition is {self.current_condition }, skipping")
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
                    self.predicted_block_dtn = self.target_identification(self.this_block_data_pending_feedback)  # identify target for this block, this will send the identification result
                    self.identifier_block_is_training_now = True
                    # Don't train until have feedback
                else:  # we are not training yet
                    self.this_block_data_pending_feedback = self.add_block_data(append_data=False)  # epoching the recorded block data
                    self.predicted_block_dtn = self.target_identification(self.this_block_data_pending_feedback)  # identify target for this block, this will send the identification result
                    self.identifier_block_is_training_now = False

            except Exception as e:
                print(f"[{self.loop_count}]IdentifierPrepEndOfBlockProcessing: Exception in end-of-block processing with vs counter value {self.vs_block_counter}: ")
                print(e)
            return 'waitingFeedback'
        else:
            print(f"[{self.loop_count}] EndOfBlockProcessing: not VS block, current condition is {self.current_condition }, skipping")
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
            events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events)
            events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])
            if 'FixationDetection' in data.keys():
                events += gaze_event_detection_PatchSim(data['FixationDetection'][0], data['FixationDetection'][1], events)
            else:
                print(f"[{self.loop_count}] AddingBlockData: WARNING: not FixationDetection stream when trying to add block data")
            rdf.add_participant_session(data, events, 0, 0, None, None, None)
            rdf.preprocess(is_running_ica=False, n_jobs=1)

            this_locking_data = {}
            for locking_name, event_filters in locking_filters.items():
                if 'VS' in locking_name:  # TODO only care about VS conditions for now
                    print(f"[{self.loop_count}] AddingBlockData: Finding epochs samples on locking {locking_name}")
                    # if is_debugging: viz_eeg_epochs(rdf, event_names, event_filters, colors, title=f'Block ID {self.current_block_id}, Condition {self.current_condition}, MetaBlock {self.current_metablock}', n_jobs=1)
                    x, y, epochs, event_ids = epochs_to_class_samples(rdf, event_names, event_filters, data_type='both', n_jobs=1, reject=None)
                    if x is None:
                        print(f"[{self.loop_count}] AddingBlockData: No event found for locking {locking_name}")
                        continue
                    if len(event_ids) == 2:
                        if self.event_ids == None:
                            self.event_ids = event_ids
                    else:
                        print(f'[{self.loop_count}] AddingBlockData: only found one event {event_ids}, skipping adding epoch')
                        continue
                    epoch_events = get_events(event_filters, events, order='time')
                    try:
                        assert np.all(np.array([x.dtn for x in epoch_events])-1 == y)
                    except:
                        print(f"[{self.loop_count}] AddingBlockData: add_block_data: epoch block events is different from y")
                        raise ValueError
                    if append_data:
                        self._add_block_data_to_locking(locking_name, x, y, epochs[0], epochs[1], epoch_events)
                        print(f"[{self.loop_count}] AddingBlockData: Add {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                    else:
                        print(f"[{self.loop_count}] AddingBlockData: find but not adding {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                    this_locking_data[locking_name] = [x, y, epochs[0], epochs[1], epoch_events]
                    print(f'[{self.loop_count}] AddingBlockData: {self.locking_data} Has {np.sum(self.locking_data[locking_name][1] == 0)} distractors and {np.sum(self.locking_data[locking_name][1] == 1)} targets')

            print(f"[{self.loop_count}] AddingBlockData: Process completed")
            self.clear_buffer()
            return this_locking_data
        except Exception as e:
            print(f"[{self.loop_count}]AddBlockData: exception when adding block data: " + str(e))

    def all_block_data_all_lockings(self, this_block_data: dict):
        for locking_name, event_filters in locking_filters.items():
            y = this_block_data[1]
            if locking_name in this_block_data.keys():
                print(f"[{self.loop_count}] AddingBlockDataPostHoc: Add {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                self._add_block_data_to_locking(locking_name, *this_block_data[locking_name])
            else:
                print(f"[{self.loop_count}] AddingBlockDataPostHoc: find but not adding {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")

    def _add_block_data_to_locking(self, locking_name, x, y, epochs_eeg, epochs_pupil, epoch_events):
        if locking_name not in self.locking_data.keys():
            self.locking_data[locking_name] = [x, y, epochs_eeg, epochs_pupil, epoch_events]
        else:  # append the data
            self.locking_data[locking_name][0][0] = np.concatenate([self.locking_data[locking_name][0][0], x[0]], axis=0)  # append EEG data
            self.locking_data[locking_name][0][1] = np.concatenate([self.locking_data[locking_name][0][1], x[1]], axis=0)  # append pupil data
            self.locking_data[locking_name][2] = mne.concatenate_epochs([self.locking_data[locking_name][2], epochs_eeg])  # append EEG epochs
            self.locking_data[locking_name][3] = mne.concatenate_epochs([self.locking_data[locking_name][3], epochs_pupil])  # append pupil epochs

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
                x_eeg, x_pupil, y, self.PCAs[locking_name], self.ICAs[locking_name], self.ARs[locking_name], _ = self.preprocess_block_data(epochs_eeg, epochs_pupil)

                model = EEGPupilCNN(eeg_in_shape=x_eeg.shape, pupil_in_shape=x_pupil.shape, num_classes=2, eeg_in_channels=x_eeg.shape[1])
                model, training_histories, criterion, label_encoder = train_model_pupil_eeg_no_folds([x_eeg, x_pupil], y, model, num_epochs=20, test_name='realtime')
                best_train_acc = np.max(training_histories['train accs'])
                print(f'[{self.loop_count}] TrainIdentificationModel: {locking_name} gives classification accuracy: {best_train_acc}')
                self.models[locking_name] = model
                self.models_accs[locking_name] = best_train_acc
            print(f'[{self.loop_count}] TrainIdentificationModel: training complete, took {time.time() - training_start_time} seconds')
        except Exception as e:
            print(e)

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
                    print(f'[{self.loop_count}] TargetIdentification: locking {locking_name} does not have preprocess transforms. Skippinp. Likely this data was not present in entire training data, which should not happen normally.')
                    continue
                x_eeg_reduced, x_pupil, y, _, _, _, rejections = self.preprocess_block_data(epochs_eeg, epochs_pupil)

                if x_eeg_reduced is None or np.all(y == 0):  # if no data or no target fixations
                    print(f'[{self.loop_count}] TargetIdentification: {np.sum(np.all(y == 2))} target epochs remains after rejection. Skipping target identification for {locking_name}.')
                    self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['accuracy'] = -1
                    self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['target sensitivity'] = -1
                    self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['target specificity'] = -1
                    continue

                x_eeg_tensor = torch.Tensor(x_eeg_reduced).to(self.device)
                x_pupil_tensor = torch.Tensor(x_pupil).to(self.device)

                self.models[locking_name].eval()
                with torch.no_grad():
                    y_pred = self.models[locking_name]([x_eeg_tensor, x_pupil_tensor])
                    confidences = y_pred.max(dim=1).detach().cpu().numpy()
                    prediction = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

                acc, tpr, tnr = binary_classification_metric(y_true=y, y_pred=prediction)
                target_sensitivity = tpr[1]

                target_specificity = tnr[1]

                cleaned_block_events = np.array(block_events)[rejections]
                target_index_id_predicted_prob = [(x.item_idnex, x.item_id, prob) for x, prob in zip(cleaned_block_events[prediction == 1], confidences)]
                predicted_item_indices = [x.item_index for x in cleaned_block_events]

                true_target_item_ids = [x.item_id for x in cleaned_block_events[y == 1]]

                # if len(predicted_target_item_ids) == 0 == None:
                #     predicted_target_ids_lockings_dict[locking_name] = None
                # else:
                for target_item_index, target_item_id, confidence in zip(predicted_item_indices, target_index_id_predicted_prob):
                    predicted_target_ids_lockings_dict[locking_name][target_item_id] += confidence
                    self.predicted_target_index_id_dict[target_item_index] += confidence * self.models_accs[locking_name]  # weight by the accuracy of the model

                item_index_dtn_predictions = np.zeros(num_item_perblock)
                for item_index, predicted_dtn in zip(predicted_item_indices, prediction):
                    item_index_dtn_predictions[item_index] = predicted_dtn + 1  # plus one to convert between pred and dtn
                item_predictions_dict[locking_name] = item_index_dtn_predictions
                try:
                    assert len(np.unique(true_target_item_ids)) == 1
                except AssertionError as e:
                    print(f"[{self.loop_count}] TargetIdentification: true target item ids not all equal, this should NEVER happen!")
                    raise(e)
                self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['accuracy'] = acc
                self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['target sensitivity'] = target_sensitivity
                self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['target specificity'] = target_specificity
                print(f"[{self.loop_count}] TargetIdentification: Locking {locking_name} {np.sum(y==1)} accuracy = {acc}, target sensitivity (TPR) = {target_sensitivity}, target specificity (TNR) = {target_specificity}, true target is {true_target_item_ids[0]}")
            print(f'[{self.loop_count}] TargetIdentification: prediction  complete, took {time.time() - prediction_start_time} seconds')
            pickle.dump(self.block_reports, open(f"{get_date_string()}_block_report", 'wb'))
            # TODO return the predicted class for each item, and the inferred target class

            try:
                # self.running_mode_of_predicted_target.append(predicted_target_ids_lockings_dict[selected_locking_name])
                # voted_target = stats.mode(self.running_mode_of_predicted_target).mode[0]
                block_target_pred_confidences = self.process_block_target_confidence(self.predicted_target_index_id_dict)
                self.send_prediction(-1, item_predictions_dict[selected_locking_name], block_target_pred_confidences)  # each value item_predictions_dict contains <num_item_perblock> items, they are the dtn prediction for each item in a block
                return item_predictions_dict[selected_locking_name]
            except KeyError:
                print(f"[{self.loop_count}] TargetIdentification: no epochs available for selected target-identification locking {selected_locking_name}, sending dummy results.")
                dummy_predictions = self.send_dummy_prediction()
                return dummy_predictions
        except Exception as e:
            print(e)
            raise e

    def process_block_target_confidence(self, predicted_target_index_id_dict: dict):
        try:
            index_target_confidence = - np.ones(num_item_perblock)
            all_probs = np.array(predicted_target_index_id_dict.values())
            all_ids = np.array([id for id in predicted_target_index_id_dict.keys()])
            prob_threshold = np.quantile(all_probs, target_threshold_quantile)

            thresholded =[(id, prob) for id, prob in predicted_target_index_id_dict.items() if prob > prob_threshold]
            thresholded.sort(key=lambda x: x[1], reverse=True)

            print(f"[{self.loop_count}] ProcessBlockTargetConfidence: process target quantiles: {target_threshold_quantile} give a threshold confidence of {prob_threshold} and {np.sum(all_probs > prob_threshold)} targets")
            print(f'with prob {thresholded}')

            try:
                assert len(thresholded) < num_item_perblock / 2
            except AssertionError:
                print(f'[{self.loop_count}] ProcessBlockTargetConfidence: more than half of <num item perblock> ({num_item_perblock/2}) are predicted as target. Trimming to top {num_item_perblock/2} ones.')
                thresholded = thresholded[:num_item_perblock/2]

            thresholded_probs =[prob for id, prob in predicted_target_index_id_dict.items() if prob > prob_threshold]
            thresholded_normalized_prof =[(id, (prob - np.min(thresholded_probs)) / (np.max(thresholded_probs) - np.min(thresholded_probs))) for id, prob in predicted_target_index_id_dict.items() if prob > prob_threshold]

            for i, (id, confidence) in enumerate(thresholded_normalized_prof):
                index_target_confidence[i] = id
                index_target_confidence[i + int(num_item_perblock / 2)] = condition_name_dict
            return index_target_confidence
        except Exception as e:
            print(f"[{self.loop_count}] ProcessBlockTargetConfidence: find exception: " + str(e))


    def send_prediction(self, predicted_target_id, block_item_prediciton, item_index_pred_target_confidence):
        try:
            send = np.zeros(3 + num_item_perblock)
            send[0] = self.current_block_id  # Unity will check the block ID matches
            send[2] = predicted_target_id  # set predicted target item id
            send[3:] = np.concatenate((block_item_prediciton, item_index_pred_target_confidence))
            self.prediction_outlet.push_sample(send)
        except Exception as e:
            raise e

    def send_dummy_prediction(self):
        pass
        # try:asdfwe need to change the dummy dtn to dummy confidence scores
        #     item_predictions = np.random.randint(0, 3, size=num_item_perblock)
        #     send = np.zeros(3 + num_item_perblock)
        #     send[0] = self.current_block_id  # Unity will check the block ID matches
        #     send[2] = -1  # set predicted target item id
        #     send[3:] = np.random.randint(1, 3, size=num_item_perblock)
        #     self.prediction_outlet.push_sample(send)
        # except Exception as e:
        #     raise e
        # return item_predictions

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

        try:
            x_eeg, x_pupil, y, ar, rejections = reject_combined(epochs_pupil, epochs_eeg, self.event_ids, n_jobs=1, n_folds=ar_cv_folds, ar=ar, return_rejections=True)  # apply auto reject
        except ValueError as e:
            print(f"[{self.loop_count}] preprocess_block_data: error in rejection, most likely the there are too few samples of fixations for it folds")
            raise e
        print(f'[{self.loop_count}] target_identification: {len(epochs_eeg) - len(x_eeg)} epochs were auto rejected. Now with {np.sum(y == 1)} targets and {np.sum(y == 0)} distractors')
        if len(y) == 0:  # no data remains after rejection
            return [None] * 6

        x_eeg, pca, ica = compute_pca_ica(x_eeg, n_components=20, pca=pca, ica=None)
        return x_eeg, x_pupil, y, pca, ica, ar, rejections

    def get_block_update(self):
        try:
            # there cannot be multiple condition updates in one block update, which runs once a second
            is_block_end = False
            new_block_id = None
            new_meta_block = None
            new_condition = None
            this_event_timestamp = None
            if len(self.inputs['Unity.ReNa.EventMarkers'][1]) - self.event_marker_head > 0: # there's new event marker
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
                    # self.current_block_id = None  # IMPORTANT, current_block_id will retain the its value until new block id is received
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

