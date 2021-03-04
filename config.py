from utils.general import slice_len_for

'''
########################################################################################################################
User parameters:
Use these parameters to set the RN App to your preference
'''
REFRESH_INTERVAL = 1  # in milliseconds, how often does the sensor/LSL pulls data from their designated sources
VISUALIZATION_REFRESH_INTERVAL = 66  # in milliseconds, how often does the plots refresh. If your app is laggy, you will want a larger value here
PLOT_RETAIN_HISTORY = 10.  # in seconds, how long a history do the plots keep

'''
########################################################################################################################
Dev parameters:
these parameters, and what they controls are still under development
'''
OPENBCI_EEG_CHANNEL_SIZE = 31
OPENBCI_EEG_USEFUL_CHANNELS = slice(1, 17)
OPENBCI_EEG_SAMPLING_RATE = 125.
OPENBCI_EEG_USEFUL_CHANNELS_NUM = slice_len_for(OPENBCI_EEG_USEFUL_CHANNELS, OPENBCI_EEG_CHANNEL_SIZE)

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
UNITY_LSL_USEFUL_CHANNELS_NUM = slice_len_for(UNITY_LSL_USEFUL_CHANNELS, UNITY_LSL_CHANNEL_SIZE)

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
