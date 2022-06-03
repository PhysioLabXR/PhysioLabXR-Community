import json

import numpy as np
import tensorflow as tf

from rena.inference.RealTimeModels.RealTimeModel import RealTimeModel
from rena.utils.data_utils import window_slice


class PupilTensorflowModel(RealTimeModel):
    expected_preprocessed_input_size = (621, 1)

    def __init__(self, model_path):
        super().__init__()
        self.window_size = 621
        self.stride = 310
        self.prepare_model(model_path, None)
        self.srate = 200
        self.baseline_index_window = (0, int(0.1 * self.srate))


    def preprocess(self, input):
        """
        :param: input: this is the data buffer
        :rtype: object
        """
        # slice the input buffer
        x = input[0][self.left_right_pupil_indices, :]  # first item is the data
        x = np.expand_dims(np.mean(x, axis=0), axis=-1)
        try:
            x = window_slice(x, self.window_size, self.stride, channel_mode='channel_last')
        except AssertionError:
            return None
        # baselining
        baseline_value = np.repeat(np.mean(x[:, self.baseline_index_window[0]:self.baseline_index_window[1]], axis=1), self.window_size).reshape(x.shape)
        return x - baseline_value

    def prepare_model(self, model_path, preprocess_params_path):
        self.model = tf.keras.models.load_model(model_path)
        channels: list = json.load(open('../Presets/LSLPresets/VarjoEyeDataComplete.json', 'r'))['ChannelNames']
        self.left_right_pupil_indices = channels.index('left_pupil_size'), channels.index('right_pupil_size')