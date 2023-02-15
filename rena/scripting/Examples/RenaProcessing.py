import copy
import json

import mne
import numpy as np
from renaanalysis.eye.eyetracking import gaze_event_detection_I_VT
from renaanalysis.learning.models import EEGPupilCNN
from renaanalysis.learning.train import train_model_pupil_eeg
from renaanalysis.params.params import conditions, dtnn_types
from renaanalysis.utils.RenaDataFrame import RenaDataFrame
from renaanalysis.utils.data_utils import epochs_to_class_samples, compute_pca_ica, reject_combined
from renaanalysis.utils.utils import get_item_events, viz_eeg_epochs

from rena.scripting.Examples.RenaProcessingParameters import locking_filters, event_names
from rena.scripting.RenaScript import RenaScript
from rena.utils.general import DataBuffer

ONLY_RECORD_META_BLOCKS = True
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
        self.current_metablock = None
        self.current_condition = None
        self.current_block_id = None
        self.event_marker_channels = json.load(open("../Presets/LSLPresets/ReNaEventMarker.json", 'r'))["ChannelNames"]
        self.loop_count = 0

        self.item_events = []
        self.item_events_queue = []  # the items (timestamps and metainfo of an event) in this queue is emptied to self.item events, when its data is available

        self.cur_state = 'idle'  # states: idle, recording, (training model), predicting
        self.num_meta_blocks_before_training = 1
        self.meta_block_counter = 0

        self.next_state = 'idle'

        self.event_marker_head = 0

        self.num_vs_blocks_before_training = 4  # for a total of 8 VS blocks in each metablock
        self.vs_block_counter = 0
        self.locking_data = {}
        self.is_inferencing = False
        self.event_ids = None

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        # self.data_buffer.update_buffers(self.inputs.buffer)  # update the buffer with the most recent events
        # self.inputs.clear_buffer()

        self.loop_count += 1
        if 'Unity.ReNa.EventMarkers' in self.inputs.buffer.keys():  # if we receive event markers
            new_block_id, new_meta_block, new_condition, is_block_end = self.get_block_update()

            if self.cur_state == 'idle':
                if is_block_end and (self.current_metablock or not ONLY_RECORD_META_BLOCKS):
                    raise Exception("System must be recording when a block ends")
                if new_meta_block:
                    self.vs_block_counter = 0  # renew the counter for metablock
                    self.meta_block_counter += 1
                    if self.meta_block_counter > self.num_meta_blocks_before_training:
                        print("Should start training now !!!")  # TODO add rena analysis scripts here
                # state transition logic
                if new_block_id and (self.current_metablock or not ONLY_RECORD_META_BLOCKS):  # only record if we are in a metablock
                    # self.data_buffer = DataBuffer()  # init the data buffer to record data of this block
                    # self.data_buffer.update_buffers(self.inputs.buffer)  # update the buffer with the most recent events
                    self.next_state = 'in_block'

            elif self.cur_state == 'in_block':
                # print('Updating buffer')
                # self.data_buffer.update_buffers(self.inputs.buffer)
                if is_block_end:
                    if self.current_condition == conditions['VS']:
                        self.vs_block_counter += 1
                        print(f"Incrementing VS block counter to {self.vs_block_counter}")
                        try:
                            if self.vs_block_counter == self.num_vs_blocks_before_training:  # time to train the model and identify target for this block
                                self.train_identification_model()
                                this_block_data = self.add_block_data()
                                self.target_identification(this_block_data)
                            elif self.vs_block_counter > self.num_vs_blocks_before_training:  # identify target for this block
                                this_block_data = self.add_block_data()
                                self.target_identification(this_block_data)
                            else:   # epoching the recorded block data
                                self.add_block_data()
                                pass
                        except Exception as e:
                            print("Hi")

                    self.next_state = 'idle'

        if self.next_state != self.cur_state:
            print(f'updating state from {self.cur_state} to {self.next_state}')
            self.cur_state = self.next_state
        # TODO check if there's any request queued

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    def add_block_data(self):
        print(f'Adding block data with id {self.current_block_id} of condition {self.current_condition}')
        rdf = RenaDataFrame()
        data = copy.deepcopy(self.inputs.buffer)  # deep copy the data so our data doesn't get changed
        events = get_item_events(data['Unity.ReNa.EventMarkers'][0], data['Unity.ReNa.EventMarkers'][1], data['Unity.ReNa.ItemMarkers'][0], data['Unity.ReNa.ItemMarkers'][1])
        events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events)
        events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])
        rdf.add_participant_session(data, events, 0, 0, None, None, None)
        rdf.preprocess(is_running_ica=False, n_jobs=1)

        this_locking_data = {}
        for locking_name, event_filters in locking_filters.items():
            if 'VS' in locking_name:  # TODO only care about VS conditions for now
                print(f"Working on locking {locking_name}")
                # if is_debugging: viz_eeg_epochs(rdf, event_names, event_filters, colors, title=f'Block ID {self.current_block_id}, Condition {self.current_condition}, MetaBlock {self.current_metablock}', n_jobs=1)
                x, y, epochs, event_ids = epochs_to_class_samples(rdf, event_names, event_filters, data_type='both', n_jobs=1, reject=None)
                if x is None:
                    print(f"No event found for locking {locking_name}")
                    continue
                if self.event_ids == None:
                    self.event_ids = event_ids
                else:
                    try:
                        assert self.event_ids == event_ids
                    except AssertionError:
                        raise ValueError(f'Event ids mismatch from previous blocks, current is {event_ids}, previous is {self.event_ids}')
                self._add_block_data_to_locking(x, y, epochs, locking_name)
                print(f"Add {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                self.report_locking_class_nums(locking_name)
                this_locking_data[locking_name] = [x, y, epochs[0], epochs[1]]
        print("Process completed")
        self.clear_buffer()
        return this_locking_data

    def clear_buffer(self):
        self.event_marker_head = 0
        self.inputs.clear_buffer()

    def train_identification_model(self):
        for locking_name, (_, _, epochs_eeg, epochs_pupil) in self.locking_data:
            (x_eeg, x_pupil), y = reject_combined(epochs_pupil, epochs_eeg, self.event_ids, n_jobs=1)  # apply auto reject
            x_eeg = compute_pca_ica(x_eeg, n_components=20)
            model = EEGPupilCNN(eeg_in_shape=x_eeg.shape, pupil_in_shape=x_pupil.shape, num_classes=2, eeg_in_channels=x_eeg.shape[1])
            model, training_histories, criterion, label_encoder = train_model_pupil_eeg([x_eeg, x_pupil], y, model,test_name='realtime')
        pass

    def target_identification(self, this_block_data):
        pass
    def  _add_block_data_to_locking(self, x, y, epochs, locking_name):
        epochs_eeg, epochs_pupil = epochs
        if locking_name not in self.locking_data.keys():
            self.locking_data[locking_name] = [x, y, epochs_eeg, epochs_pupil]
        else:  # append the data
            self.locking_data[locking_name][0][0] = np.concatenate([self.locking_data[locking_name][0][0], x[0]], axis=0)  # append EEG data
            self.locking_data[locking_name][0][1] = np.concatenate([self.locking_data[locking_name][0][1], x[1]], axis=0)  # append pupil data
            self.locking_data[locking_name][2] = mne.concatenate_epochs([self.locking_data[locking_name][2], epochs_eeg])  # append EEG epochs
            self.locking_data[locking_name][3] = mne.concatenate_epochs([self.locking_data[locking_name][3], epochs_pupil])  # append pupil epochs

            self.locking_data[locking_name][1] = np.concatenate([self.locking_data[locking_name][1], y], axis=0)  # append labels

    def report_locking_class_nums(self, locking_name):
        print(f'Has {np.sum(self.locking_data[locking_name][1] == 0)} distractors and {np.sum(self.locking_data[locking_name][1] == 1)} targets')

    def get_block_update(self):
        # there cannot be multiple condition updates in one block update, which runs once a second
        is_block_end = False
        new_block_id = None
        new_meta_block = None
        new_condition = None

        if len(self.inputs['Unity.ReNa.EventMarkers'][1]) - self.event_marker_head > 0: # there's new event marker
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
                    raise Exception(f"Did not receive block end signal. This block end {block_id_start_end}. Current block id {self.current_block_id}")
                try:
                    assert self.current_block_id is not None
                except AssertionError:
                    raise Exception(f"self.current_block_id is None when block_end signal ({block_id_start_end}) comes, that means a block start is never received")
                print("[{0}] Block with ID {1} ended. ".format(self.loop_count, self.current_block_id))
                self.current_block_id = None
                is_block_end = True
            else:  # the only other possibility is that this is a meta block marker
                new_meta_block = block_marker if block_marker in metablock_name_dict.keys() else None

        if new_meta_block:
            self.current_block_id = new_block_id
            print("[{0}] Entering new META block {1}, clearing buffer".format(self.loop_count, metablock_name_dict[new_meta_block]))
            self.current_metablock = new_meta_block
            # self.inputs.clear_buffer()  # should clear buffer at every metablock so we don't need to deal with practice round data
        if new_block_id:
            self.current_block_id = new_block_id
            # message = "[{0}] Entering new block with ID {1}".format(self.loop_count, self.current_block_id) + \
            # ". No metablock is given, assuming in practice rounds." if not self.current_metablock else ""
            print("[{0}] Entering new block with ID {1}".format(self.loop_count, self.current_block_id) + ". No metablock is given, assuming in practice rounds." if not self.current_metablock else "")
        if new_condition:
            self.current_block_id = new_block_id
            self.current_condition = new_condition
            print("[{0}] Setting current condition to {1}".format(self.loop_count, condition_name_dict[new_condition]))

        return new_block_id, new_meta_block, new_condition, is_block_end  # tells if there's new block and meta block

    def get_item_event(self):
        for i in range(len(self.inputs['Unity.ReNa.EventMarkers'][1])):
            block_id_start_end = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("BlockIDStartEnd"), i]