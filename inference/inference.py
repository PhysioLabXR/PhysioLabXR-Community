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
    info = StreamInfo(lsl_data_name, lsl_data_type, channel_count=config.INFERENCE_CLASS_NUM, channel_format='float32', source_id='myuid1234')
    info.desc().append_child_value("apocalyvec", "RealityNavigation")

    chns = info.desc().append_child("inference")
    channel_names = ['class' + str(i) for i in range(config.INFERENCE_CLASS_NUM)]
    for label in channel_names:
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)

    outlet = StreamOutlet(info, max_buffered=360)
    start_time = local_clock()

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
            data, timestamps = inlet.pull_chunk(timeout=1e-2, max_samples=config.EYE_TOTAL_POINTS_PER_INFERENCE)
            if len(data) > 0:
                timestamp_accumulated += timestamps  # accumulate timestamps
                eye_data_accumulated = np.concatenate([eye_data_accumulated, data])

            if len(eye_data_accumulated) == config.EYE_TOTAL_POINTS_PER_INFERENCE:
                print(len(eye_data_accumulated))
                print(eye_data_accumulated.shape[0] / config.EYE_TOTAL_POINTS_PER_INFERENCE)
                samples = np.reshape(eye_data_accumulated[:config.EYE_TOTAL_POINTS_PER_INFERENCE], newshape=(config.EYE_SAMPLES_PER_INFERENCE, config.EYE_INFERENCE_WINDOW_TIMESTEPS, -1))  # take the most recent samples

                # conduct inference
                samples_preprocessed = preprocess_eye_samples(samples, data_downsampled_max, data_downsampled_min)
                results, results_decoded = inference(samples_preprocessed, model, y_encoder)

                # send the inference results via LSL, only send out the not-decoded results
                results_moving_average = np.mean(results, axis=0)
                print(results_moving_average)
                outlet.push_sample(results_moving_average)

                # put the reminder
                eye_data_accumulated = np.empty((0, 2))  # this is mostly not needed because each call to pull_chunk will return a fixed chunk size of config.EYE_TOTAL_POINTS_PER_INFERENCE
                print(eye_data_accumulated.shape[0] / config.EYE_TOTAL_POINTS_PER_INFERENCE)
                print()
            elif len(eye_data_accumulated) > config.EYE_TOTAL_POINTS_PER_INFERENCE:
                print('missed, this should never happen; you probably want to use a smaller batch size')
        except KeyboardInterrupt:
            pass
            # if len(samples) > 1:
            #     f_sample = 1. / ((samples[-1][0] - samples[0][0]) / len(samples))
            #     print('Interrupted, f_sample is ' + str(f_sample))
            # break

if __name__ == '__main__':
    main()