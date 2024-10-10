import random
import string
import time

import numpy as np
from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop
from scipy.signal import resample
from tensorflow.python.keras.models import load_model

from physiolabxr.configs import config

import pickle
import os


# user parameters
from physiolabxr.utils.data_utils import interp_negative

data_stream_name = 'Unity.ViveSREyeTracking'
inference_chann_count = 1
data_chann_count = 16
use_channels = np.s_[0:2]
f_data = 120
f_resample = 20
time_window = 1.5  # in seconds
adaptive_min_max_time_window = 3

reconnect_patience = 1.  # in seconds

def load_model_params():
    y_encoder = pickle.load(open(os.path.join(os.getcwd(), config.EYE_INFERENCE_Y_ENCODER_PATH), 'rb'))
    data_downsampled_min, data_downsampled_max = pickle.load(
        open(os.path.join(os.getcwd(), config.EYE_INFERENCE_MIN_MAX_PATH), 'rb'))
    model = load_model(os.path.join(os.getcwd(), config.EYE_INFERENCE_MODEL_PATH))
    return model, y_encoder, data_downsampled_min, data_downsampled_max


def preprocess_eye_samples(samples, f_resample, data_downsampled_min=None, data_downsampled_max=None):
    # interp missing value (i.e., negative values)
    try:
        samples_preprocessed = np.array([[interp_negative(e) for e in x] for x in samples])
    except ValueError:
        # catch when array of sample points is all negative
        samples_preprocessed = samples
    samples_preprocessed = resample(samples_preprocessed, int(
        config.EYE_INFERENCE_WINDOW_TIMESTEPS * f_resample / config.UNITY_LSL_SAMPLING_RATE),
                                    axis=1)  # resample to 20 hz
    # min-max normalize
    if data_downsampled_min and data_downsampled_max:
        samples_preprocessed = (samples_preprocessed - np.min(data_downsampled_min)) / (
                    np.max(data_downsampled_max) - np.min(data_downsampled_min))
    return samples_preprocessed


def inference(samples_preprocessed, y_encoder=None):
    results = make_inference_on_sample(samples_preprocessed)
    results_decoded = y_encoder.inverse_transform(results) if y_encoder else None
    return results, results_decoded


def make_inference_on_sample(samples_preprocessed):
    return np.mean(samples_preprocessed, axis=(1, 2))


def main():
    inlet = None
    timestamp_accumulated = None
    data_accumulated = None
    no_data_duration = None

    # Create a outlet to relay the scripting results  ##################################################################
    if len(resolve_byprop('name', config.INFERENCE_LSL_RESULTS_NAME, timeout=0.5)) > 0:
        print(
            'Inference stream with name {0} alreayd exists, cannot start. Check if there are other same script running'.format(
                config.INFERENCE_LSL_RESULTS_NAME))
        raise Exception('Inference stream already exists')

    lsl_data_type = config.INFERENCE_LSL_RESULTS_TYPE
    lsl_data_name = config.INFERENCE_LSL_RESULTS_NAME
    info = StreamInfo(lsl_data_name, lsl_data_type, channel_count=inference_chann_count,
                      channel_format='float32',
                      source_id=(''.join(random.choice(string.digits) for i in range(8))), nominal_srate=110)
    info.desc().append_child_value("apocalyvec", "RealityNavigation")
    chns = info.desc().append_child("scripting")
    channel_names = ['class' + str(i) for i in range(inference_chann_count)]
    for label in channel_names:
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)
    outlet = StreamOutlet(info, max_buffered=360)
    print('Created scripting results out stream ...')

    # Main Loop ##################################################################################
    while True:
        if inlet:
            data, timestamp = inlet.pull_sample(timeout=1e-2)
            if data:
                no_data_duration = 0  # received data,
                timestamp_accumulated.append(timestamp)  # accumulate timestamps
                data_accumulated = np.concatenate([data_accumulated, np.expand_dims(data, axis=-1)], axis=-1)  # take the most recent samples

                # conduct scripting
                # simply take the tail of data
                if data_accumulated.shape[-1] > f_data * time_window:
                    try:
                        _frame_min_max_adaptive = [interp_negative(x) for x in data_accumulated[use_channels, -int(f_data * adaptive_min_max_time_window):]]
                        data_downsampled_min = np.min(_frame_min_max_adaptive)
                        data_downsampled_max = np.max(_frame_min_max_adaptive)
                    except ValueError:  # use default
                        data_downsampled_min = 2.
                        data_downsampled_max = 8.
                    samples = np.expand_dims(data_accumulated[use_channels, -int(f_data * time_window):], axis=0)
                    samples_preprocessed = preprocess_eye_samples(samples, f_resample=f_resample, data_downsampled_min=data_downsampled_min, data_downsampled_max=data_downsampled_max)
                    results, _ = inference(samples_preprocessed)

                    # send the scripting results via LSL, only send out the not-decoded results
                    outlet.push_sample(results)
                    inference_per_second = len(timestamp_accumulated) / (
                                timestamp_accumulated[-1] - timestamp_accumulated[0]) if timestamp_accumulated[-1] - \
                                                                                         timestamp_accumulated[
                                                                                             0] != 0. else 0.
                    print('Inference per second is ' + str(inference_per_second), end='\r', flush=True)
            else:
                no_data_duration += time.time() - current_time
                if no_data_duration > reconnect_patience:
                    print('No data seen on data stream with name {0}. Assume it is lost, trying to reconnect'.format(data_stream_name))
                    inlet = None

        else:
            print('Waiting for data stream with name {0} ...'.format(data_stream_name))
            streams = resolve_byprop('name', data_stream_name)
            if len(streams) < 0:
                print('No stream found with name {0}, cannot start.'.format(data_stream_name))
                raise Exception('No stream found for given name')
            # create a new inlet to read from the stream  ######################################################################
            inlet = StreamInlet(streams[0])
            print('Found data stream with name {0}'.format(data_stream_name))

            # model, y_encoder, data_downsampled_min, data_downsampled_max = load_model_params()

            # data buffers #####################################################################
            timestamp_accumulated = []
            data_accumulated = np.empty(shape=(data_chann_count, 0))
            no_data_duration = 0.
            print('Entering scripting loop')
            current_time = time.time()

        current_time = time.time()

if __name__ == '__main__':
    main()
