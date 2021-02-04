import numpy as np
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet, local_clock
from scipy.signal import resample
from tensorflow.python.keras.models import load_model

import config

import pickle
import os

def load_model_params():
    y_encoder = pickle.load(open(os.path.join(os.getcwd(), config.EYE_INFERENCE_Y_ENCODER_PATH), 'rb'))
    data_downsampled_min, data_downsampled_max = pickle.load(open(os.path.join(os.getcwd(), config.EYE_INFERENCE_MIN_MAX_PATH), 'rb'))
    model = load_model(os.path.join(os.getcwd(), config.EYE_INFERENCE_MODEL_PATH))
    return model, y_encoder, data_downsampled_min, data_downsampled_max

def preprocess_eye_samples(samples, data_downsampled_max, data_downsampled_min, downsample_to_hz=20):
    samples_downsampled = resample(samples, int(config.EYE_INFERENCE_WINDOW_TIMESTEPS * downsample_to_hz / config.UNITY_LSL_SAMPLING_RATE),
                                   axis=1)  # resample to 20 hz
    # min-max normalize
    samples_downsampled_minmaxnormlized = (samples_downsampled - np.min(data_downsampled_min)) / (np.max(data_downsampled_max) - np.min(data_downsampled_min))
    return samples_downsampled_minmaxnormlized

def inference(samples_preprocessed, model, y_encoder):
    results = model.predict(samples_preprocessed)
    results_decoded = y_encoder.inverse_transform(results)
    return results, results_decoded

def main():
    print("looking for stream with type " + config.INFERENCE_LSL_NAME)
    streams = resolve_stream('type', config.INFERENCE_LSL_NAME)

    # create a new inlet to read from the stream  ######################################################################
    inlet = StreamInlet(streams[0])

    # create a outlet to relay the inference results  ##################################################################
    lsl_data_type = config.INFERENCE_LSL_RESULTS_TYPE
    lsl_data_name = config.INFERENCE_LSL_RESULTS_NAME

    # TODO need to change the channel count when adding eeg
    info = StreamInfo(lsl_data_name, lsl_data_type, channel_count=2, channel_format='float32', source_id='myuid2424')
    info.desc().append_child_value("apocalyvec", "RealityNavigation")

    chns = info.desc().append_child("eye")
    channel_names = ['left_pupil_diameter_sample', 'right_pupil_diameter_sample']
    for label in channel_names:
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "mm")
        ch.append_child_value("type", "eye")

    self.outlet = StreamOutlet(info, max_buffered=360)
    self.start_time = local_clock()

    # Load inference parameters ########################################################################################
    model, y_encoder, data_downsampled_min, data_downsampled_max = load_model_params()

    # data buffers #####################################################################
    eye_data_accumulated = np.empty((0, 2))
    timestamp_accumulated = []

    print('Entering inference loop')
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        try:
            data, timestamps = inlet.pull_chunk(max_samples=config.EYE_TOTAL_POINTS_PER_INFERENCE)
            if len(data) > 0:
                timestamp_accumulated += timestamps  # accumulate timestamps
                eye_data_accumulated = np.concatenate([eye_data_accumulated, data])

            if len(eye_data_accumulated) >= config.EYE_TOTAL_POINTS_PER_INFERENCE:
                print(len(eye_data_accumulated))
                print(eye_data_accumulated.shape[0] / config.EYE_TOTAL_POINTS_PER_INFERENCE)
                samples = np.reshape(eye_data_accumulated[-config.EYE_TOTAL_POINTS_PER_INFERENCE:], newshape=(config.EYE_SAMPLES_PER_INFERENCE, config.EYE_INFERENCE_WINDOW_TIMESTEPS, -1))  # take the most recent samples

                # conduct inference
                samples_preprocessed = preprocess_eye_samples(samples, data_downsampled_max, data_downsampled_min)
                results, results_decoded = inference(samples_preprocessed, model, y_encoder)

                # send the inference results via LSL, only send out the not-decoded results


                # put the reminder
                print('cutting')
                eye_data_accumulated = eye_data_accumulated[:-config.EYE_TOTAL_POINTS_PER_INFERENCE]  # this is mostly not needed because each call to pull_chunk will return a fixed chunk size of config.EYE_TOTAL_POINTS_PER_INFERENCE
                print(eye_data_accumulated.shape[0] / config.EYE_TOTAL_POINTS_PER_INFERENCE)
                print()

        except KeyboardInterrupt:
            pass
            # if len(samples) > 1:
            #     f_sample = 1. / ((samples[-1][0] - samples[0][0]) / len(samples))
            #     print('Interrupted, f_sample is ' + str(f_sample))
            # break

if __name__ == '__main__':
    main()