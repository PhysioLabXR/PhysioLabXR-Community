

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

###########################################
TRAIN_RESPONSE_MARKER = 1
TEST_RESPONSE_MARKER = 2

###########################################
# State representation
IDLE_STATE = 0
RECORDING_STATE = 1
TRAINING_STATE = 2
TESTING_STATE = 3
############################################

# IDLE_STATE
EEG_SAMPLING_RATE = 250.0



P300EventStreamName = 'P300Speller'

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

montage = 'standard_1005'

event_id = {'non_target': NONTARGET_MARKER, 'target': TARGET_MARKER}
event_color = {'non_target': 'blue', 'target': 'red'}


##########################
# Visualization Settings
tmin_eeg_viz = -0.1
tmax_eeg_viz = 1.

