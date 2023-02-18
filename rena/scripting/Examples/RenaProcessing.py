import copy
import json
import pickle
import time
from collections import defaultdict

import mne
import numpy as np
import torch
from renaanalysis.eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim
from renaanalysis.learning.models import EEGPupilCNN
from renaanalysis.learning.train import train_model_pupil_eeg, train_model_pupil_eeg_no_folds
from renaanalysis.params.params import conditions, dtnn_types, tmax_pupil
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

        self.num_vs_blocks_before_training = 4  # for a total of 8 VS blocks in each metablock
        self.vs_block_counter = 0
        self.locking_data = {}
        self.is_inferencing = False
        self.event_ids = None

        self.end_of_block_waited = None
        self.end_of_block_wait_time = 3.5

        self.PCAs = {}
        self.ICAs = {}
        self.current_target_item_id = None
        self.models = {}

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.block_reports = defaultdict(dict)

        self.ar = None

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        try:
            self.loop_count += 1
            if 'Unity.ReNa.EventMarkers' in self.inputs.buffer.keys():  # if we receive event markers
                if self.cur_state != 'endOfBlockWaiting':  # will not process new event marker if waiting
                    new_block_id, new_meta_block, new_condition, is_block_end, event_timestamp = self.get_block_update()
                else:
                    new_block_id, new_meta_block, new_condition, is_block_end, event_timestamp = [None] * 5

                if self.cur_state == 'idle':
                    if is_block_end and self.current_metablock is not None:
                        print(f'[{self.loop_count}] System is idle when received block_end, this probably means the last loop took too long')
                        if np.max(self.inputs['Unity.VarjoEyeTrackingComplete'][1]) - event_timestamp > tmax_pupil + epoch_margin:
                            print(f'[{self.loop_count}] Eyetracking data has progressed beyond margin, next state will be processing')
                            self.next_state = 'endOfBlockProcessing'
                        else:
                            print(f'[{self.loop_count}] Eyetracking data NOT passed tmax margin, next state will be waiting')
                            self.next_state = 'endOfBlockWaiting'
                    if new_meta_block:
                        print(f"[{self.loop_count}] in idle, find new meta block, metablock counter = {self.meta_block_counter}")
                        self.vs_block_counter = 0  # renew the counter for metablock
                        # if self.meta_block_counter > self.num_meta_blocks_before_training:
                        #     print("Should start training now !!!")  # TODO add rena analysis scripts here
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
                    if np.max(self.inputs['Unity.VarjoEyeTrackingComplete'][1]) > self.last_block_end_timestamp + tmax_pupil + epoch_margin:
                        self.next_state = 'endOfBlockProcessing'
                elif self.cur_state == 'endOfBlockProcessing':
                    if self.current_condition == conditions['VS']:
                        self.vs_block_counter += 1
                        print(f"[{self.loop_count}] EndOfBlockProcessing: Incrementing VS block counter to {self.vs_block_counter}")
                        try:
                            if self.vs_block_counter == self.num_vs_blocks_before_training:  # time to train the model and identify target for this block
                                self.add_block_data()
                                self.train_identification_model()
                            elif self.vs_block_counter > self.num_vs_blocks_before_training:  # identify target for this block
                                this_block_data = self.add_block_data()
                                self.target_identification(this_block_data)
                            else:
                                self.add_block_data()  # epoching the recorded block data
                        except Exception as e:
                            print(f"[{self.loop_count}] Exception in end-of-block processing with vs counter value {self.vs_block_counter}: ")
                            print(e)
                    else:
                        print(f"[{self.loop_count}] EndOfBlockProcessing: not VS block, skipping")
                    self.next_state = 'idle'

            if self.next_state != self.cur_state:
                print(f'[{self.loop_count}] updating state from {self.cur_state} to {self.next_state}')
                self.cur_state = self.next_state
            # TODO check if there's any request queued
        except Exception as e:
            print(e)
        # print(f"[{self.loop_count}] End of loop ")

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    def add_block_data(self):
        print(f'[{self.loop_count}] Adding block data with id {self.current_block_id} of condition {self.current_condition}')
        rdf = RenaDataFrame()
        data = copy.deepcopy(self.inputs.buffer)  # deep copy the data so our data doesn't get changed
        events = get_item_events(data['Unity.ReNa.EventMarkers'][0], data['Unity.ReNa.EventMarkers'][1], data['Unity.ReNa.ItemMarkers'][0], data['Unity.ReNa.ItemMarkers'][1])
        events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events)
        events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])
        if 'FixationDetection' in data.keys():
            events += gaze_event_detection_PatchSim(data['FixationDetection'][0], data['FixationDetection'][1], events)
        else:
            print(f"[{self.loop_count}] WARNING: not FixationDetection stream when trying to add block data")
        rdf.add_participant_session(data, events, 0, 0, None, None, None)
        rdf.preprocess(is_running_ica=False, n_jobs=1)

        this_locking_data = {}
        for locking_name, event_filters in locking_filters.items():
            if 'VS' in locking_name:  # TODO only care about VS conditions for now
                print(f"[{self.loop_count}] Finding epochs samples on locking {locking_name}")
                # if is_debugging: viz_eeg_epochs(rdf, event_names, event_filters, colors, title=f'Block ID {self.current_block_id}, Condition {self.current_condition}, MetaBlock {self.current_metablock}', n_jobs=1)
                x, y, epochs, event_ids = epochs_to_class_samples(rdf, event_names, event_filters, data_type='both', n_jobs=1, reject=None)
                if x is None:
                    print(f"[{self.loop_count}] No event found for locking {locking_name}")
                    continue
                if len(event_ids) == 2:
                    if self.event_ids == None:
                        self.event_ids = event_ids
                else:
                    print(f'[{self.loop_count}] only found one event {event_ids}, skipping adding epoch')
                    continue
                epoch_events = get_events(event_filters, events)
                self._add_block_data_to_locking(x, y, epochs, locking_name, epoch_events)
                print(f"[{self.loop_count}] Add {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                self.report_locking_class_nums(locking_name)
                this_locking_data[locking_name] = [x, y, epochs[0], epochs[1], epoch_events]
        print(f"[{self.loop_count}] Process completed")
        self.clear_buffer()
        return this_locking_data

    def clear_buffer(self):
        print(f"[{self.loop_count}] Buffer cleared")
        self.event_marker_head = 0
        self.inputs.clear_up_to(self.last_block_end_timestamp)

    # def check_pupil_data_complete_epoch(self, start_time):
    #     return (np.max(self.inputs['Unity.VarjoEyeTrackingComplete'][1]) - start_time) > (tmax_pupil + epoch_margin)

    def train_identification_model(self):
        try:
            training_start_time = time.time()
            for locking_name, (_, y, epochs_eeg, epochs_pupil, _) in self.locking_data.items():
                print(f'[{self.loop_count}] Training models for {locking_name}')
                x_eeg, x_pupil, y, self.PCAs[locking_name], self.ICAs[locking_name] = self.preprocess_block_data(epochs_eeg, epochs_pupil)

                model = EEGPupilCNN(eeg_in_shape=x_eeg.shape, pupil_in_shape=x_pupil.shape, num_classes=2, eeg_in_channels=x_eeg.shape[1])
                model, training_histories, criterion, label_encoder = train_model_pupil_eeg_no_folds([x_eeg, x_pupil], y, model, num_epochs=20, test_name='realtime')
                best_train_acc = np.max(training_histories['train accs'])
                print(f'[{self.loop_count}] {locking_name} gives classification accuracy: {best_train_acc}')
                self.models[locking_name] = model
            print(f'[{self.loop_count}] training complete, took {time.time() - training_start_time} seconds')
        except Exception as e:
            print(e)

    def target_identification(self, this_block_data):
        try:
            prediction_start_time = time.time()
            for locking_name, ((_, _), y, epochs_eeg, epochs_pupil, block_events) in this_block_data.items():
                # (x_eeg, x_pupil), y, rejects = reject_combined(epochs_pupil, epochs_eeg, self.event_ids,  n_jobs=1)  # NOT auto reject
                # block_events_cleaned = np.array(block_events)[rejects]
                print(f"[{self.loop_count}] Locking {locking_name} Has {len(y)} epochs, with {np.sum(y==1)} targets and {np.sum(y==0)} distractors")
                assert locking_name in self.PCAs.keys() and locking_name in self.ICAs.keys()
                x_eeg, x_pupil, y = reject_combined(epochs_pupil, epochs_eeg, self.event_ids, n_jobs=1, ar=self.ar)  # apply auto reject
                print(f'[{self.loop_count}] target_identification: {len(epochs_eeg) - len(x_eeg)} epochs were auto rejected.')
                x_eeg_reduced, _, _ = compute_pca_ica(x_eeg, n_components=20, pca=self.PCAs[locking_name], ica=self.ICAs[locking_name])

                x_eeg_tensor = torch.Tensor(x_eeg_reduced).to(self.device)
                x_pupil_tensor = torch.Tensor(x_pupil).to(self.device)

                pred = self.models[locking_name]([x_eeg_tensor, x_pupil_tensor])
                pred = torch.argmax(pred, dim=1).detach().cpu().numpy()

                acc, tpr, tnr = binary_classification_metric(y_true=y, y_pred=pred)
                target_sensitivity = tpr[1]
                target_specificity = tnr[1]

                predicted_target_item_ids = [x.item_id for x in np.array(block_events)[pred==1]]
                true_target_item_ids = [x.item_id for x in np.array(block_events)[y==1]]
                predicted_target_item_id = None if len(predicted_target_item_ids)==0 else stats.mode(predicted_target_item_ids).mode[0]
                try:
                    assert len(np.unique(true_target_item_ids)) == 1
                except AssertionError as e:
                    print(f"[{self.loop_count}] true target item ids not all equal, this should NEVER happen!")
                    print(e)
                self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['accuracy'] = acc
                self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['target sensitivity'] = target_sensitivity
                self.block_reports[(self.meta_block_counter, self.current_block_id, locking_name)]['target specificity'] = target_specificity
                print(f"[{self.loop_count}] Locking {locking_name} {np.sum(y==1)} accuracy = {acc}, target sensitivity (TPR) = {target_sensitivity}, target specificity (TNR) = {target_specificity}, predicts the target is {predicted_target_item_id}, true target is {true_target_item_ids[0]}")
            print(f'[{self.loop_count}] prediction  complete, took {time.time() - prediction_start_time} seconds')
            pickle.dump(self.block_reports, open(f"{get_date_string()}_block_report", 'wb'))
        except Exception as e:
            print(e)

    def preprocess_block_data(self, epochs_eeg, epochs_pupil):
        x_eeg, x_pupil, y, self.ar = reject_combined(epochs_pupil, epochs_eeg, self.event_ids, n_jobs=1)  # apply auto reject
        x_eeg, pca, ica = compute_pca_ica(x_eeg, n_components=20)
        return x_eeg, x_pupil, y, pca, ica

    def _add_block_data_to_locking(self, x, y, epochs, locking_name, epoch_events):
        epochs_eeg, epochs_pupil = epochs
        if locking_name not in self.locking_data.keys():
            self.locking_data[locking_name] = [x, y, epochs_eeg, epochs_pupil, epoch_events]
        else:  # append the data
            self.locking_data[locking_name][0][0] = np.concatenate([self.locking_data[locking_name][0][0], x[0]], axis=0)  # append EEG data
            self.locking_data[locking_name][0][1] = np.concatenate([self.locking_data[locking_name][0][1], x[1]], axis=0)  # append pupil data
            self.locking_data[locking_name][2] = mne.concatenate_epochs([self.locking_data[locking_name][2], epochs_eeg])  # append EEG epochs
            self.locking_data[locking_name][3] = mne.concatenate_epochs([self.locking_data[locking_name][3], epochs_pupil])  # append pupil epochs

            self.locking_data[locking_name][1] = np.concatenate([self.locking_data[locking_name][1], y], axis=0)  # append labels
            self.locking_data[locking_name][4] += epoch_events  # append labels

    def report_locking_class_nums(self, locking_name):
        print(f'[{self.loop_count}] Has {np.sum(self.locking_data[locking_name][1] == 0)} distractors and {np.sum(self.locking_data[locking_name][1] == 1)} targets')

    def get_block_update(self):
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
            # self.inputs.clear_buffer()  # should clear buffer at every metablock so we don't need to deal with practice round data
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

    def get_item_event(self):
        for i in range(len(self.inputs['Unity.ReNa.EventMarkers'][1])):
            block_id_start_end = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("BlockIDStartEnd"), i]