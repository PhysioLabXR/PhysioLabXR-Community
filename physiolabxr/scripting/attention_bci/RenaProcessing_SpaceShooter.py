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

#----------- add RLPF to python path--------------
from rlpf.eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim, gaze_event_detection_I_DT
from rlpf.params.params import conditions, tmax_pupil, random_seed
from rlpf.utils.Event import get_events
from rlpf.utils.RenaDataFrame import RenaDataFrame
from rlpf.utils.rdf_utils import rena_epochs_to_class_samples_rdf
from rlpf.utils.utils import get_item_events
from rlpf.utils.viz_utils import visualize_block_gaze_event
from scipy.stats import stats
from torch.distributed.rpc import rpc_async
from physiolabxr.rpc.decorator import rpc, async_rpc

#----------------- physiolabxr utils --------------------
from physiolabxr.scripting.attention_bci.RenaCameraUtils import receive_fixation_decode,get_cam_socket,receive_decode_image,draw_bboxes
from physiolabxr.scripting.attention_bci.RenaFixationDataset import FixationDataset
from physiolabxr.scripting.attention_bci.RenaProcessingParameters import locking_filters, event_names, epoch_margin
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.configs.shared import bcolors
from physiolabxr.utils.data_utils import get_date_string, mode_by_column
from physiolabxr.scripting.attention_bci.RenaDataLockingUtils import to_bytes, from_bytes

# should add the SS condition to the processing file
condition_name_dict = {1: "RSVP", 2: "Carousel", 3: "Visual Search", 4: "Table Search", 8: "Table Search gnd", 9: "Table Search Identifier", 10: "Space Shooter", 11: "SS gnd", 12: "SS orc PDecoder", 13: "SS full", 14: "Conclusion"}
metablock_name_dict = {5: "Classifier Prep", 6: "Identifier Prep"}
colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}

is_simulating_predictions = False
is_simulating_eeg = False
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

        self.last_block_end_timestamp = None
        mne.use_log_level(False)

        self.loop_count = 0

        self.event_marker_head = 0

        self.locking_data = {}
        self.event_ids = None

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        '''----------------------------The part for fixation detection images-----------------------------------------'''
        # fix detection parameters  #######################################
        self.loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
        self.previous_img_patch = None
        self.fixation_frame_counter = 0
        self.distance = 0
        self.fixation_outlet = StreamOutlet(StreamInfo("FixationDetection", 'FixationDetection', 3, 30, 'float32'))

        # # TODO: -------------the camera capture should be from the local host--------------
        # self.fixation_cam_socket = get_cam_socket("tcp://127.0.0.1:5556", 'ColorDepthCamGazePositionBBox')
        # self.fixation_cam_socket.setsockopt(zmq.CONFLATE,1)
        # self.right_cam_socket =    get_cam_socket("tcp://127.0.0.1:5557", 'ColorDepthCamRight')
        # self.right_cam_socket.setsockopt(zmq.CONFLATE,1)
        # self.left_cam_socket =     get_cam_socket("tcp://127.0.0.1:5558", 'ColorDepthCamLeft')
        # self.left_cam_socket.setsockopt(zmq.CONFLATE, 1)
        # self.back_cam_socket =     get_cam_socket("tcp://127.0.0.1:5559", 'ColorDepthCamBack')
        # self.back_cam_socket.setsockopt(zmq.CONFLATE, 1)
        # print(f'Image sockets connected.')
        #
        # # TODO: ------------- the processed data from OVTR detection------------------------
        # self.ovtr_fixation_cam_socket = get_cam_socket("tcp://127.0.0.1:5560", 'OVTRCamFixation')
        # self.ovtr_fixation_cam_socket.setsockopt(zmq.CONFLATE, 1)
        # self.ovtr_right_cam_socket = get_cam_socket("tcp://127.0.0.1:5561", 'OVTRCamRight')
        # self.ovtr_right_cam_socket.setsockopt(zmq.CONFLATE, 1)
        # self.ovtr_left_cam_socket = get_cam_socket("tcp://127.0.0.1:5562", 'OVTRCamLeft')
        # self.ovtr_left_cam_socket.setsockopt(zmq.CONFLATE, 1)
        # self.ovtr_back_cam_socket = get_cam_socket("tcp://127.0.0.1:5563", 'OVTRCamBack')
        # self.ovtr_back_cam_socket.setsockopt(zmq.CONFLATE, 1)
        # print(f'OVTR image sockets connected.')

        '''--------------------------- Declaration of fixation dataset ---------------------------------------'''
        # self.fixation_dataset = FixationDataset()
        
        mp_np.patch()

        '''-------------------- the channel setup for locking data ----------------------------'''
        ctx = zmq.Context.instance()
        self.locking_socket = ctx.socket(zmq.PUB)
        self.locking_socket.bind('tcp://127.0.0.1:5564')




    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        try:
            self.loop_count += 1
        except Exception as e:
            print(e)

    '''############################################ the block related function call #####################################'''

    def cleanup(self):
        print('Cleanup function is called')


    # TODO make this an async RPC, called by Unity at the end of a visual search
    # this way you can remove all the state tracking logic in here
    @async_rpc
    def add_block_data(self, append_data=True, ):
        try:
            print(f'[{self.loop_count}] AddingBlockData: Adding block data')

            rdf = RenaDataFrame()
            data = copy.deepcopy(self.inputs.buffer)  # deep copy the data so our data doesn't get changed. This data only have the last block's data because at the end of this function, we clear the buffer
            events = get_item_events(data['Unity.ReNa.EventMarkers'][0], data['Unity.ReNa.EventMarkers'][1], data['Unity.ReNa.ItemMarkers'][0], data['Unity.ReNa.ItemMarkers'][1])
            events += gaze_event_detection_I_DT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])  # TODO no need to resample the headtracking data again
            events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])
            if 'FixationDetection' in data.keys():
                events += gaze_event_detection_PatchSim(data['FixationDetection'][0], data['FixationDetection'][1], events)
            else:
                print(f"[{self.loop_count}] AddingBlockData: WARNING: not FixationDetection stream when trying to add block data")
            rdf.add_participant_session(data, events, '0', 0, None, None, None)
            try:
                rdf.preprocess(is_running_ica=True, n_jobs=1, ocular_artifact_mode='proxy')

            except Exception as e:
                print(f"Encountered value error when preprocessing rdf: {str(e)}")
                return None
            this_locking_data = {}
            for locking_name, event_filters in locking_filters.items():
                if 'VS' in locking_name:  # TODO only care about VS conditions for now
                    x, y, epochs, event_ids = rena_epochs_to_class_samples_rdf(rdf, event_names, event_filters, data_type='both', n_jobs=1, reject=None, plots='full', colors=colors, title=f'{locking_name}')
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
            self.clear_buffer()  # clear the buffer so that next time we run this function,

            # serialize the locking data to byte format
            locking_payload = to_bytes(this_locking_data)

            # send the serialized data to the tunneled server
            self.locking_socket.send_multipart([b"locking", locking_payload])
            print('The locking payload packed and sent to the server')
            return this_locking_data
        except Exception as e:
            print(f"[{self.loop_count}]AddBlockData: exception when adding block data: " + str(e))
            raise e

    # def add_block_data_all_lockings(self, this_block_data: dict):
    #     try:
    #         for locking_name, event_filters in locking_filters.items():
    #             if 'VS' in locking_name:
    #                 if locking_name in this_block_data.keys():
    #                     y = this_block_data[locking_name][1]
    #                     if locking_name in this_block_data.keys():
    #                         print(f"[{self.loop_count}] AddingBlockDataPostHoc: Add {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
    #                         self._add_block_data_to_locking(locking_name, *this_block_data[locking_name])
    #                         print(f'[{self.loop_count}] AddingBlockData: {locking_name} Has {np.sum(self.locking_data[locking_name][1] == 0)} distractors and {np.sum(self.locking_data[locking_name][1] == 1)} targets')
    #                     else:
    #                         print(f"[{self.loop_count}] AddingBlockDataPostHoc: find but not adding {len(y)} samples to {locking_name} with {np.sum(y == 0)} distractors and {np.sum(y == 1)} targets")
    #                 else:
    #                     print(f'[{self.loop_count}] AddingBlockDataPostHoc: no data is available for {locking_name}')
    #     except Exception as e:
    #         raise e

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
    return epochs_concatenated\
