import os
import numpy as np
from PyQt6.QtCore import QSettings

'''
########################################################################################################################
User parameters:
Use these parameters to set the RN App to your preference
'''
REFRESH_FREQUENCY_RETAIN_FRAMES = 50

VISUALIZATION_REFRESH_FREQUENCY_RETAIN_FRAMES = 20  # the duration use for frequency calculation (the back track duration partially depends on the refresh rate)

MAIN_WINDOW_META_DATA_REFRESH_INTERVAL = 500
VIZ_DISPLAY_DURATION = 10.  # in seconds, how long a history do the plots keep

SCRIPTING_UPDATE_REFRESH_INTERVAL = 15
STOP_PROCESS_KILL_TIMEOUT = 2000  # wait up to 2 second after sending the stop command,
REQUEST_REALTIME_INFO_TIMEOUT = 2000  # wait up to 2 second after sending the stop command,

downsample_method_mean_sr_threshold = 256
'''
########################################################################################################################
Advanced parameters:
do not change these unless you know what you are doing
'''
DOWNSAMPLE_MULTIPLY_THRESHOLD = 5

'''
########################################################################################################################
data file parameters:
data recording settings
'''
# if platform == "linux" or platform == "linux2":
#     # linux
# elif platform == "darwin":
#     # OS X
# elif platform == "win32":
#     # Windows...
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser('~/Documents'), 'Recordings')
FILE_FORMATS = ["Rena Native (.dats)", "MATLAB (.m)", "Pickel (.p)"]
# FILE_FORMATS = ["Rena Native (.dats)", "MATLAB (.m)", "Pickel (.p)", "Comma separate values (.CSV)"]

DEFAULT_EXPERIMENT_NAME = 'my experiment'
DEFAULT_SUBJECT_TAG = 'someone'
DEFAULT_SESSION_TAG = '0'

'''
########################################################################################################################
Dev parameters:
parameters whose functions are still under development
'''
OPENBCI_EEG_CHANNEL_SIZE = 31
OPENBCI_EEG_USEFUL_CHANNELS = slice(1, 17)
OPENBCI_EEG_SAMPLING_RATE = 125.
# OPENBCI_EEG_USEFUL_CHANNELS_NUM = slice_len_for(OPENBCI_EEG_USEFUL_CHANNELS, OPENBCI_EEG_CHANNEL_SIZE)

INFERENCE_REFRESH_INTERVAL = 200  # in milliseconds

EYE_INFERENCE_WINDOW_SEC = 4
EYE_INFERENCE_WINDOW_NUM = 2  # we look at 2 windows at a time
EYE_WINDOW_STRIDE_SEC = 0.2 # unit is in seconds

EYE_INFERENCE_MODEL_PATH = 'model/'
EYE_INFERENCE_Y_ENCODER_PATH = 'encoder.p'
EYE_INFERENCE_MIN_MAX_PATH = 'minmax.p'
INFERENCE_CLASS_NUM = 2

INFERENCE_LSL_RESULTS_NAME = 'Python.Inference.Results'
INFERENCE_LSL_RESULTS_TYPE = 'Python.Inference.Results'

INFERENCE_LSL_NAME = 'Python.Samples'
INFERENCE_LSL_TYPE = 'Python.Samples'
'''
########################################################################################################################
Deprecated parameters:
to be removed in future iterations
'''
UNITY_LSL_CHANNEL_SIZE = 17
UNITY_LSL_SAMPLING_RATE = 70.
UNITY_LSL_USEFUL_CHANNELS = slice(1, 10)
# UNITY_LSL_USEFUL_CHANNELS_NUM = slice_len_for(UNITY_LSL_USEFUL_CHANNELS, UNITY_LSL_CHANNEL_SIZE)

sensors = ['OpenBCICyton', 'RNUnityEyeLSL', 'B-Alert X24']

'''
########################################################################################################################
Calculated parameters:
do not change these unless you know what you are doing
'''
EYE_WINDOW_STRIDE_TIMESTEMPS = int(UNITY_LSL_SAMPLING_RATE * EYE_WINDOW_STRIDE_SEC)
EYE_INFERENCE_WINDOW_TIMESTEPS = int(UNITY_LSL_SAMPLING_RATE * EYE_INFERENCE_WINDOW_SEC)
EYE_INFERENCE_TOTAL_TIMESTEPS = int(EYE_INFERENCE_WINDOW_TIMESTEPS * EYE_INFERENCE_WINDOW_NUM)
EYE_SAMPLES_PER_INFERENCE = int((EYE_INFERENCE_TOTAL_TIMESTEPS - EYE_INFERENCE_WINDOW_TIMESTEPS) / EYE_WINDOW_STRIDE_TIMESTEMPS)
EYE_TOTAL_POINTS_PER_INFERENCE = EYE_SAMPLES_PER_INFERENCE * EYE_INFERENCE_WINDOW_TIMESTEPS * 2  # 2 for two eyes


# USER_SETTINGS_PATH = "../UserConfig.json"
# if not os.path.exists(USER_SETTINGS_PATH):
#     # create default user config
#     USER_SETTINGS = {"USER_DATA_DIR": DEFAULT_DATA_DIR}
#     json.dump(USER_SETTINGS, open(USER_SETTINGS_PATH, 'w'))
# else:
#     USER_SETTINGS = json.load(open(USER_SETTINGS_PATH))
DEFAULT_CHANNEL_DISPLAY_NUM = 40

settings = QSettings('TeamRena', 'RenaLabApp')  # load the user settings

# rena_server_port = 9999999
# rena_server_name = 'RENA_SERVER'
# rena_server_worker_ports = np.arange(1,100)

rena_server_add_dsp_worker_request = 1
rena_server_update_worker_request = 2
rena_server_remove_worker_request = 3
rena_server_exit_request = 4

# stream num
MAX_TS_CHANNEL_NUM = 4000


# Scripting
scripting_port = 8000
CONSOLE_LOG_MAX_NUM_ROWS = 1000
script_fps_counter_buffer_size = 10000

valid_preset_categories = ['other', 'video', 'exp']

stream_availability_wait_time = 2  # in seconds

# viz
plot_fps_range = (1, 60)

default_group_name = 'default group name '

app_data_name = 'RenaLabApp'

def valid_networking_interfaces():
    return None