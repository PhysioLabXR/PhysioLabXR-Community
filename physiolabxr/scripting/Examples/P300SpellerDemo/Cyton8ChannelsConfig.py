

# OpenBCI Stream Name
OpenBCIStreamName = 'OpenBCI_Cython_8_LSL'

# Sampling Rate
eeg_sampling_rate = 250

# all channel names
openbci_cython_8_channels = [
    "PackageNum",

    "Fz",
    "Cz",
    "Pz",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1"

    "X", "Y", "Z",
    "Other1", "Other2", "Other3", "Other4", "Other5", "Other6", "Other7",
    "Analog1", "Analog2", "Analog3",
    "TimeStamp",
    "Marker"
]

channel_num = 8

# eeg channel index for the experiment. The Cython8 board has 8 eeg channels
eeg_channel_index = [1, 2, 3, 4, 5, 6, 7, 8]

# eeg channels from cython 8 board
eeg_channel_names = [
    "Fz",
    "Cz",
    "Pz",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1"
]

# channel types are 8 eeg channels
channel_types = ['eeg'] * 8

