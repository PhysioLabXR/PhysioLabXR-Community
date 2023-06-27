# LSL Stream Name from the P300Speller game
P300EventStreamName = 'P300Speller'

p300_speller_event_marker_channel_index = {
    "P300SpellerGameStateControlMarker": 0,
    "P300SpellerStartTrailStartEndMarker": 1,
    "P300SpellerFlashingMarker": 2,
    "P300SpellerFlashingRowOrColumMarker": 3,
    "P300SpellerFlashingRowColumIndexMarker": 4,
    "P300SpellerTargetNonTargetMarker": 5
}

# P300SpellerGameStateControlMarker
START_TRAINING_MARKER = 1
END_TRAINING_MARKER = 2
START_TESTING_MARKER = 3
END_TESTING_MARKER = 4
InterruptExperimentMarker = -1

# P300SpellerStartTrailStartEndMarker
TRAIL_START_MARKER = 1
TRAIL_END_MARKER = 2

# P300SpellerFlashingMarker
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

event_id = {
    'non_target': NONTARGET_MARKER,
    'target': TARGET_MARKER
}

# training
test_size = 0.2

###########################################
# State representation
IDLE_STATE = 0
RECORDING_STATE = 1
TRAINING_STATE = 2
TESTING_STATE = 3

# Other Settings
#############################################
# Visualization Settings
tmin_eeg_viz = -0.1
tmax_eeg_viz = 1.
event_color = {'non_target': 'blue', 'target': 'red'}

# montage = 'standard_1005'






