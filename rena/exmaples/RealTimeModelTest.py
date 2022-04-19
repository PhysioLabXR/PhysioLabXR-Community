import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from rena.ui.InferenceTab import RealTimeModel
from rena.utils.data_utils import window_slice


class PupilTensorflowModel(RealTimeModel):
    expected_preprocessed_input_size = (621, 1)

    def __init__(self, model_path):
        super().__init__()
        self.window_size = 621
        self.stride = 621
        self.prepare_model(model_path, None)
        self.srate = 200
        self.baseline_index_window = (0, int(0.1 * self.srate))


    def preprocess(self, input, ):
        """
        :param: input: this is the data buffer
        :rtype: object
        """
        # slice the input buffer
        try:
            x = window_slice(input, self.window_size, self.stride, channel_mode='channel_last')
        except AssertionError:
            return None
        # baselining
        baseline_value = np.repeat(np.mean(x[:, self.baseline_index_window[0]:self.baseline_index_window[1]], axis=1), self.window_size).reshape(x.shape)
        return x - baseline_value

    def prepare_model(self, model_path, preprocess_params_path):
        self.model = tf.keras.models.load_model(model_path)

X = np.load('C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/SingleTrials/epochs_pupil_raw_condition_RSVP_DL.npy')
y = np.load('C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/SingleTrials/epochs_pupil_raw_condition_RSVP_DL_labels.npy')

model = PupilTensorflowModel('D:\PycharmProjects\ReNaAnalysis\Learning\Model\Pupil_ANN')

buffer_data = np.concatenate(X[:5], axis=0)  # dummy buffer data

preprocess_data = model.preprocess(buffer_data)
y_pred = model.predict(preprocess_data)

model.model.evaluate(x = X, y = y)
#
# [plt.plot(xx) for xx in x]
# plt.show()