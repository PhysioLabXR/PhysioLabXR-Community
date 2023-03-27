
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
# Visualization Settings
tmin_eeg_viz = -0.1
tmax_eeg_viz = 1.
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

