"""
You will find plenty of try, except Exception in the functions defined here. This is to help debugging with breakpoints
in case a function raises exception. We don't have to dig back into the function to find what's wrong
"""

import copy
import json
import pickle
import struct
import time
from collections import defaultdict, deque
import re

import cv2
import mne
import numpy as np
import torch
import zmq
import msgpack as mp
import msgpack_numpy as mp_np
from lpips import lpips
from mne import EpochsArray
from pylsl import StreamInfo, StreamOutlet, pylsl
from pathlib import Path

#----------- add RLPF to python path--------------
from rlpf.eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim, gaze_event_detection_I_DT
from rlpf.params.params import conditions, tmax_pupil, random_seed
from rlpf.utils.Event import get_events
from rlpf.utils.RenaDataFrame import RenaDataFrame
from rlpf.utils.rdf_utils import rena_epochs_to_class_samples_rdf
from rlpf.utils.utils import get_item_events
from rlpf.utils.viz_utils import visualize_block_gaze_event
from scipy.stats import stats
from sklearn.utils import resample

from physiolabxr.rpc.decorator import rpc, async_rpc

#----------------- physiolabxr utils --------------------
from physiolabxr.scripting.attention_bci.RenaCameraUtils import receive_fixation_decode,get_cam_socket,receive_decode_image,draw_bboxes
from physiolabxr.scripting.attention_bci.RenaFixationDataset import FixationDataset
from physiolabxr.scripting.attention_bci.RenaProcessingParameters import locking_filters, event_names, epoch_margin
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.configs.shared import bcolors
from physiolabxr.utils.buffers import DataBuffer
from physiolabxr.utils.data_utils import get_date_string, mode_by_column
from physiolabxr.scripting.attention_bci.RenaDataLockingUtils import to_bytes, from_bytes

# should add the SS condition to the processing file
condition_name_dict = {1: "RSVP", 2: "Carousel", 3: "Visual Search", 4: "Table Search", 8: "Table Search gnd", 9: "Table Search Identifier", 10: "Space Shooter", 11: "SS gnd", 12: "SS orc PDecoder", 13: "SS full", 14: "Conclusion"}
metablock_name_dict = {5: "Classifier Prep", 6: "Identifier Prep"}
colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}

is_simulating_predictions = False
is_simulating_eeg = False
is_simulating_VS = True
num_item_perblock = 30

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


        self.data_buffer = DataBuffer()
        self.num_vs_before_training = None
        self.last_block_end_timestamp = None
        self.end_of_block_wait_start = None
        mne.use_log_level(False)
        self.current_metablock = None
        self.meta_block_counter = 0
        self.current_condition = None
        self.current_block_id = None

        # the json is under physiolabxr/_presets
        ROOT = Path(__file__).resolve().parents[2]
        json_path = ROOT/"_presets"/ "LSLPresets"/"ReNaEventMarker.json"
        self.event_marker_channels = json.load(open(json_path, 'r'))["ChannelNames"]
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

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        '''----------------------------The part for fixation detection images-----------------------------------------'''
        # fix detection parameters  #######################################
        self.loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
        self.previous_img_patch = None
        self.fixation_frame_counter = 0
        self.distance = 0

        mp_np.patch()

        '''-------------------- the channel setup for locking data ----------------------------'''
        # ctx = zmq.Context.instance()
        # self.locking_socket = ctx.socket(zmq.PUB)
        # self.locking_socket.setsockopt(zmq.IPV6, 0)
        # self.locking_socket.bind('tcp://127.0.0.1:1002')


    # Start will be called once when the run button is hit.
    def init(self):
        #//////// for debugging only, remove when finished//////////////////
        self.add_block_data()
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        try:
            self.loop_count += 1

            # feature: the state machine for event markers
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
                        # self.predicted_target_index_id_dict = defaultdict(float)
                        # self.running_mode_of_predicted_target = []  # reset running mode of predicted target

                        # TODO: change the state for classifier prep and identifier prep
                        if new_meta_block == 5:
                            # self.num_vs_before_training = num_vs_to_train_in_classifier_prep
                            # print(f'[{self.loop_count}] entering classifier prep, num visual search blocks for training will be {num_vs_to_train_in_classifier_prep}')
                            pass
                        elif new_meta_block == 6:
                            # self.num_vs_before_training = num_vs_to_train_in_identifier_prep
                            # print(f'[{self.loop_count}] entering identifier and performance evaluation')
                            pass

                        # warning: should add a new meta block for the type of space shooter

                    if new_block_id and self.current_metablock:  # only record if we are in a metablock, this is to ignore the practice
                        print(f"[{self.loop_count}] in idle, find new block id {self.current_block_id}, entering in_block")
                        self.next_state = 'in_block'


                elif self.cur_state == 'in_block':
                    # feature: update the data buffer while in block state
                    # print('Updating buffer')
                    self.data_buffer.update_buffers(self.inputs.buffer)
                    if is_block_end:
                        self.next_state = 'endOfBlockWaiting'
                        self.end_of_block_wait_start = time.time()

                elif self.cur_state == 'endOfBlockWaiting':
                    self.end_of_block_waited = time.time() - self.end_of_block_wait_start
                    # print(f"[{self.loop_count}] end of block waited {self.end_of_block_waited}")
                    # if self.end_of_block_waited > self.end_of_block_wait_time:
                    if is_simulating_predictions:
                        # if self.end_of_block_waited > end_of_block_wait_time_in_simulate:
                        #     self.next_state = 'endOfBlockProcessing'
                        pass
                    else:
                        if np.max(self.inputs['Unity.VarjoEyeTrackingComplete'][1]) > self.last_block_end_timestamp + tmax_pupil + epoch_margin:
                            self.next_state = 'endOfBlockProcessing'

                elif self.cur_state == 'endOfBlockProcessing':
                    if self.current_metablock == 6:
                        # self.next_state = self.classifier_prep_phase_end_of_block()
                        pass
                    elif self.current_metablock == 7:
                        # self.next_state = self.identifier_prep_phase_end_of_block()
                        pass
                    else:
                        print(f'[{self.loop_count}] block ended on while in meta block {self.current_metablock}. Skipping end of block processing. Not likely to happen.')


            # the state switcher
            if self.next_state != self.cur_state:
                print(f'[{self.loop_count}] updating state from {self.cur_state} to {self.next_state}')
                self.cur_state = self.next_state
        except Exception as e:
            print(e)

    '''############################################ the block related function call #####################################'''

    def cleanup(self):
        print('Cleanup function is called')


    # TODO make this an async RPC, called by Unity at the end of a visual search
    # this way you can remove all the state tracking logic in here
    @async_rpc
    def add_block_data(self):
        append_data = True
        try:
            print(f'[{self.loop_count}] AddingBlockData: Adding block data')

            if "BioSemi" in self.inputs.buffer.keys():
                rdf = RenaDataFrame(eeg_srate=2048)
            elif "BAlert" in self.inputs.buffer.keys():
                rdf = RenaDataFrame(eeg_srate=256)
            else:
                rdf = RenaDataFrame()

            #///////// function to simulate VS block, turn off when running experiment////////////
            if not is_simulating_VS:
                data = copy.deepcopy(self.inputs.buffer)  # deep copy the data so our data doesn't get changed. This data only have the last block's data because at the end of this function, we clear the buffer
            else:
                with open('real_data.pkl', 'rb') as f:
                    data = pickle.load(f)

            # load the data based on the .pkl file content
            events = get_item_events(data['Unity.ReNa.EventMarkers'][0], data['Unity.ReNa.EventMarkers'][1], data['Unity.ReNa.ItemMarkers'][0], data['Unity.ReNa.ItemMarkers'][1])
            events += gaze_event_detection_I_DT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])  # TODO no need to resample the headtracking data again
            events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])

            # if 'FixationDetection' in data.keys():
            #     events += gaze_event_detection_PatchSim(data['FixationDetection'][0], data['FixationDetection'][1], events)
            # else:
            #     print(f"[{self.loop_count}] AddingBlockData: not FixationDetection stream when trying to add block data")

            rdf.add_participant_session(data, events, '0', 0, None, None, None, None, None)
            try:
                rdf.preprocess(is_running_ica=True, n_jobs=1, ocular_artifact_mode='proxy',exg_resample_rate = 256)
            except Exception as e:
                print(f"Encountered value error when preprocessing rdf: {str(e)}")
                return None
            this_locking_data = {}
            for locking_name, event_filters in locking_filters.items():
                if 'VS' in locking_name:
                    # TODO: double check if the bio semi send any streams
                    x, y, epochs, event_ids, meta_data = rena_epochs_to_class_samples_rdf(rdf, event_names, event_filters, data_type='both', n_jobs=1, reject=None, plots='full', colors=colors, title=f'{locking_name}', picks_eeg=('Fz', 'Cz', 'Pz', 'POz'))
                    if x is None:
                        print(f"{bcolors.WARNING}[{self.loop_count}] AddingBlockData: No event found for locking {locking_name}{bcolors.ENDC}")
                        continue
                    if len(event_ids) == 2:
                        if self.event_ids is None:
                            self.event_ids = event_ids
                    else:
                        print(f'{bcolors.WARNING}[{self.loop_count}] AddingBlockData: {locking_name} only found one event {event_ids}{bcolors.ENDC}')
                    epoch_events = get_events(event_filters, events, order='time')
                    try:
                        assert np.all(np.array([x.dtn for x in epoch_events])-1 == y)
                    except AssertionError as e:
                        print(f"[{self.loop_count}] AddingBlockData: add_block_data: epoch block events is different from y")
                        raise e

                    for bid in np.unique([e.block_id for e in epoch_events]):
                        block_e = [e for e in epoch_events if e.block_id == bid]
                        target_item_id_count = len(np.unique([e.item_id for e in block_e if e.dtn == 2.0]))
                        try:
                            assert target_item_id_count == 1 or target_item_id_count == 0
                        except AssertionError as e:
                            print(f"[{self.loop_count}] AddingBlockData: true target item ids not all equal, this should NEVER happen!")
                            raise e

                    # this is for single-block
                    target_item_id_count = len(np.unique([e.item_id for e in epoch_events if e.dtn==2.0]))

                    # try:
                    #     assert target_item_id_count == 1 or target_item_id_count == 0
                    # except AssertionError as e:
                    #     print(f"[{self.loop_count}] AddingBlockData: true target item ids not all equal, this should NEVER happen!")
                    #     raise e


                    if append_data:
                        self._add_block_data_to_locking(locking_name, x, y, epochs[0], epochs[1], epoch_events)
                        print(f"[{self.loop_count}] AddingBlockData: Add {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                        print(f'[{self.loop_count}] AddingBlockData: {locking_name} Has {np.sum(self.locking_data[locking_name][1] == 0)} distractors and {np.sum(self.locking_data[locking_name][1] == 1)} targets')
                    else:
                        print(f"[{self.loop_count}] AddingBlockData: find but not adding {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
                    this_locking_data[locking_name] = [x, y, epochs[0], epochs[1], epoch_events]

            print(f"[{self.loop_count}] AddingBlockData: Process completed")
            self.clear_buffer()  # clear the buffer so that next time we run this function,

            # serialize the locking data to byte format
            locking_payload = to_bytes(this_locking_data)

            # send the serialized data to the tunneled server
            # self.locking_socket.send_multipart([b"locking", locking_payload])
            print('The locking payload packed and sent to the server')
        except Exception as e:
            print(f"[{self.loop_count}]AddBlockData: exception when adding block data: " + str(e))
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

    # the function for get the update for the block
    def get_block_update(self):
        try:
            # there cannot be multiple condition updates in one block update, which runs once a second
            is_block_end = False
            new_block_id = None
            new_meta_block = None
            new_condition = None
            this_event_timestamp = None

            # This is the checker for new event
            if len(self.inputs['Unity.ReNa.EventMarkers'][1]) - self.event_marker_head > 0:  # there's new event marker
                this_event_timestamp = self.inputs['Unity.ReNa.EventMarkers'][1][self.event_marker_head]
                block_id_start_end = self.inputs['Unity.ReNa.EventMarkers'][0][
                    self.event_marker_channels.index("BlockIDStartEnd"), self.event_marker_head]
                block_marker = self.inputs['Unity.ReNa.EventMarkers'][0][
                    self.event_marker_channels.index("BlockMarker"), self.event_marker_head]
                self.event_marker_head += 1

                if block_id_start_end > 0:  # if there's a new block id
                    new_block_id = None if block_id_start_end == 0 or block_id_start_end < 0 else block_id_start_end  # block id less than 0 is also when the block ends
                    new_condition = block_marker if block_marker in condition_name_dict.keys() else None
                elif block_id_start_end < 0:  # if this is an end of a block
                    try:
                        assert block_id_start_end == -self.current_block_id
                    except AssertionError:
                        raise Exception(
                            f"[{self.loop()}] get_block_update: Did not receive block end signal. This block end {block_id_start_end}. Current block id {self.current_block_id}")
                    try:
                        assert self.current_block_id is not None
                    except AssertionError:
                        raise Exception(
                            f"[{self.loop()}] get_block_update: self.current_block_id is None when block_end signal ({block_id_start_end}) comes, that means a block start is never received")
                    print("[{0}] get_block_update: Block with ID {1} ended. ".format(self.loop_count,
                                                                                     self.current_block_id))
                    # self.current_block_id = None  # IMPORTANT, current_block_id will retain its value until new block id is received
                    self.last_block_end_timestamp = this_event_timestamp
                    is_block_end = True
                else:  # the only other possibility is that this is a meta block marker
                    new_meta_block = block_marker if block_marker in metablock_name_dict.keys() else None

            if new_meta_block:
                # self.current_block_id = new_block_id
                self.meta_block_counter += 1
                print("[{0}] get_block_update: Entering new META block {1}, metablock count is {2}".format(
                    self.loop_count, metablock_name_dict[new_meta_block], self.meta_block_counter))
                self.current_metablock = new_meta_block
                # self.inputs.clear_buffer()  # should clear buffer at every metablock, so we don't need to deal with practice round data

            if new_block_id:
                self.current_block_id = new_block_id
                # message = "[{0}] Entering new block with ID {1}".format(self.loop_count, self.current_block_id) + \
                # ". No metablock is given, assuming in practice rounds." if not self.current_metablock else ""
                print("[{0}] get_block_update: Entering new block with ID {1}".format(self.loop_count,
                                                                                      self.current_block_id) + (
                          ". No metablock is given, assuming in practice rounds." if not self.current_metablock else ""))

            if new_condition:
                self.current_block_id = new_block_id
                self.current_condition = new_condition
                print(
                    "[{0}] get_block_update: Block {2} is setting current condition to {1}".format(self.loop_count,
                                                                                                   condition_name_dict[
                                                                                                       new_condition],
                                                                                                   self.current_block_id))
            return new_block_id, new_meta_block, new_condition, is_block_end, this_event_timestamp  # tells if there's new block and meta block
        except Exception as e:
            raise e

    def clear_buffer(self):
        print(f"[{self.loop_count}] Buffer cleared")
        self.event_marker_head = 0
        self.inputs.clear_up_to(self.last_block_end_timestamp, ignores=['Unity.ReNa.PredictionFeedback'])

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
