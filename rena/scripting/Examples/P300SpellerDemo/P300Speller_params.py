# P300SpellerGameStateControlMarker
START_TRAINING_MARKER = 1
END_TRAINING_MARKER = 2
START_TESTING_MARKER = 3
END_TESTING_MARKER = 4
InterruptExperimentMarker = -1

# P300SpellerTrailStartEndMarker
TRAIL_START_MARKER = 1
TRAIL_END_MARKER = 2

# P300SpellerStartFlashingMarker
FLASHING_MARKER = 1

# P300SpellerFlashingRowOrColumMarker
ROW_FLASHING_MARKER = 1
COL_FLASHING_MARKER = 2

# P300SpellerTargetNonTargetMarker
NONTARGET_MARKER = 1
TARGET_MARKER = 2




##########
TRAIN_RESPONSE_MARKER = 1
TEST_RESPONSE_MARKER = 2

###########################################
IDLE_STATE = 0
RECORDING_STATE = 1
TRAINING_STATE = 99
TESTING_STATE = 100
############################################

EEG_SAMPLING_RATE = 250.0

Time_Window = 1.1  # second

OpenBCIStreamName = 'OpenBCI_Cython_8_LSL'

P300EventStreamName = 'P300Speller'

sampling_rate = 250
data_duration = 2
channel_num = 8

test_size = 0.2

p300_speller_event_marker_channel_index = {
    "P300SpellerGameStateControlMarker": 0,
    "P300SpellerStartTrailStartEndMarker": 1,
    "P300SpellerFlashingMarker": 2,
    "P300SpellerFlashingRowOrColumMarker": 3,
    "P300SpellerFlashingRowColumIndexMarker": 4,
    "P300SpellerTargetNonTargetMarker": 5
}

channel_types = ['eeg'] * 8
# channel_names = [
#     "Fp1",
#     "Fp2",
#     "C3",
#     "C4",
#     "P7",
#     "P8",
#     "O1",
#     "O2"
# ]
channel_names = [
    "Fz",
    "Cz",
    "Pz",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1"
]

event_id = {'non_target': NONTARGET_MARKER, 'target': TARGET_MARKER}
event_color = {'non_target': 'blue', 'target': 'red'}
montage = 'standard_1005'

##########################

import itertools
import json
from os import path

import mne
import numpy as np
import pkg_resources

from renaanalysis.utils.Bidict import Bidict

# base_root = "C:/Users/LLINC-Lab/Dropbox/ReNa/data/ReNaPilot-2022Fall/"
# base_root = "/Users/Leo/Dropbox/ReNa/data/ReNaPilot-2022Fall"
# base_root = "D:/Dropbox/Dropbox/ReNa/data/ReNaPilot-2022Fall"
base_root = "C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2022Fall"
data_directory = "Subjects"
export_data_root = 'C:/Data'
# data_directory = "Subjects-Test"
# data_directory = "Subjects-Test-IncompleteBlock"

# event_ids_dict = {'EventMarker': {'DistractorPops': 1, 'TargetPops': 2, 'NoveltyPops': 3},
#             'GazeRayIntersect': {'GazeRayIntersectsDistractor': 4, 'GazeRayIntersectsTarget': 5, 'GazeRayIntersectsNovelty': 6},
#             'GazeBehavior': {'FixationDistractor': 7, 'FixationTarget': 8, 'FixationNovelty': 9, 'FixationNull': 10,
#                               'Saccade2Distractor': 11, 'Saccade2Target': 12, 'Saccade2Novelty': 13,
#                               'Saccade2Null': 14},
#                   }  # event_ids_for_interested_epochs
# color_dict = {
#               'DistractorPops': 'blue', 'TargetPops': 'red', 'NoveltyPops': 'orange',
#               'Fixation': 'blue', 'Saccade': 'orange',
#                 "GazeRayIntersectsDistractor": 'blue', "GazeRayIntersectsTarget": 'red', "GazeRayIntersectsNovelty": 'orange',
#               'FixationDistractor': 'blue', 'FixationTarget': 'red', 'FixationNovelty': 'orange', 'FixationNull': 'grey',
#               'Saccade2Distractor': 'blue', 'Saccade2Target': 'red', 'Saccade2Novelty': 'orange', 'Saccade2Null': 'yellow'}

dtn_color_dict = {None: 'grey', 1: 'blue', 2: 'red', 3: 'orange', 4: 'grey'}

event_viz = 'GazeRayIntersect'

conditions = Bidict({'RSVP': 1., 'Carousel': 2., 'VS': 3., 'TS': 4., 'TS-gnd': 8, 'TS-id': 9})
dtnn_types = Bidict({'Distractor': 1, 'Target': 2, 'Novelty': 3, 'Null': 4})
meta_blocks = Bidict({'cp': 5, 'ip': 7})

# load presets
eeg_chs = ["Trig1", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14",
           "A15", "A16", "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28",
           "A29", "A30", "A31", "A32", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12",
           "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20", "B21", "B22", "B23", "B24", "B25", "B26",
           "B27", "B28", "B29", "B30", "B31", "B32", "EX1", "EX2", "EX3", "EX4", "EX5", "EX6", "EX7", "EX8",
           "AUX1", "AUX2", "AUX3", "AUX4", "AUX5", "AUX6", "AUX7", "AUX8", "AUX9", "AUX10", "AUX11", "AUX12",
           "AUX13", "AUX14", "AUX15", "AUX16"]
eventmarker_chs = [
    "BlockMarker",
    "BlockIDStartEnd",
    "DTN",
    "itemID",
    "objDistFromPlayer",
    "CarouselSpeed",
    "CarouselAngle",
    "TSHandLeft", "TSHandRight",
    "Likert"]

headtracker_chs = [
    "Head Yaw", "Head Pitch", "Head Roll",
    "Head Displacement X", "Head Displacement Y", "Head Displacement Z",
    "Head Position X", "Head Position Y", "Head Position Z"
]
varjoEyetracking_stream_name = "Unity.VarjoEyeTrackingComplete"
varjoEyetracking_chs = [
    "raw_timestamp",
    "log_time",
    "focus_distance",
    "frame_number",
    "stability",
    "status",
    "Angle2CameraUp",
    "gaze_forward_x",
    "gaze_forward_y",
    "gaze_forward_z",
    "gaze_origin_x",
    "gaze_origin_y",
    "gaze_origin_z",
    "HMD_position_x",
    "HMD_position_y",
    "HMD_position_z",
    "HMD_rotation_x",
    "HMD_rotation_y",
    "HMD_rotation_z",
    "left_forward_x",
    "left_forward_y",
    "left_forward_z",
    "left_origin_x",
    "left_origin_y",
    "left_origin_z",
    "left_pupil_size",
    "left_status",
    "right_forward_x",
    "right_forward_y",
    "right_forward_z",
    "right_origin_x",
    "right_origin_y",
    "right_origin_z",
    "right_pupil_size",
    "right_status"
]
# eeg_preset = json.load(pkg_resources.resource_stream(__name__, 'BioSemi.json'))
# eventmarker_preset = json.load(pkg_resources.resource_stream(__name__, 'ReNaEventMarker.json'))
# headtracker_preset = json.load(pkg_resources.resource_stream(__name__, 'UnityHeadTracking.json'))

tmin_pupil = -1
tmax_pupil = 3.
tmin_pupil_viz = -0.1
tmax_pupil_viz = 3.

tmin_eeg = -1.2
tmax_eeg = 2.4

tmin_eeg_viz = -0.1
tmax_eeg_viz = 1.

eyetracking_srate = 200
eyetracking_resample_srate = 20
exg_srate = 2048
exg_resample_srate = 128

eeg_picks = [
    "Fz",
    "Cz",
    "Pz",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1"
]

eeg_montage = mne.channels.make_standard_montage('biosemi64')
eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
ecg_ch_name = 'ECG00'

note = "test_v3"

FIXATION_MINIMAL_TIME = 1e-3 * 141.42135623730952

START_OF_BLOCK_ENCODING = 4
END_OF_BLOCK_ENCODING = 5

# ITEM_TYPE_ENCODING = {event_ids_dict['GazeRayIntersect']['GazeRayIntersectsDistractor']: 'distractor',
#                       event_ids_dict['GazeRayIntersect']['GazeRayIntersectsTarget']: 'target',
#                       event_ids_dict['GazeRayIntersect']['GazeRayIntersectsNovelty']: 'novelty'}


'''
The core events, each core event will have some meta information associated with it

RSVP-pop:
'''

# classifier_prep_markers = ['{}-{}-{}-{}'.format(a, b, c ,d) for a, b, c ,d in itertools.product(['practice', 'exp'], ['RSVP', 'Carousel'], ['Distractor', 'Target', 'Novelty'], ['Pop', 'IDTFixGaze', 'FixDetectGaze'])]
# identifier_prep_markers = ['{}-VS-{}-{}'.format(a, b, c) for a, b, c in itertools.product(['practice', 'exp'], ['Distractor', 'Target', 'Novelty'], ['IDTFixGaze', 'FixDetectGaze'])]
# events = ['BlockStart', 'BlockEnd'] + classifier_prep_markers + identifier_prep_markers
#
#
# events = Bidict(dict([(e, i) for i, e in enumerate(events)]))

item_marker_names = ['itemDTNType', 'ItemIndexInBlock', 'itemID', 'foveateAngle', 'isInFrustum', 'isGazeRayIntersected',
                     'distFromPlayer', 'transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z']

SACCADE_CODE = 1
FIXATION_CODE = 2

num_items_per_constrainted_block = 30

reject = dict(eeg=100e-6)  # DO NOT reject or we will have a mismatch between EEG and pupil

is_regenerate_ica = False
debug = True

eeg_epoch_ticks = np.array([0, 0.3, 0.6, 0.8])
pupil_epoch_ticks = np.array([0, 0.5, 1., 1.5, 2., 2.5, 3])

lr = 1e-3
batch_size = 64
epochs = 5000
patience = 75
train_ratio = 0.8
model_save_dir = 'renaanalysis/learning/saved_models'
l2_weight = 1e-5

random_seed = 42

# HDCA parameters
split_window_eeg = 100e-3
split_window_pupil = 500e-3
num_folds = 10
num_top_compoenents = 20
