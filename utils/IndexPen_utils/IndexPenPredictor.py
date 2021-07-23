import time
from collections import deque

import numpy as np
import tensorflow as tf

from utils.data_utils import is_broken_frame, clutter_removal


class IndexPenRealTimePredictor:
    def __init__(self, model_path, classes, debouncer_threshold, data_buffer_len):

        self.predict_interval = 3
        self.predict_interval_counter = 0
        self.IndexPenRealTimePreprocessor = IndexPenRealTimePreprocessor()
        self.model_path = model_path
        self.classes = classes
        self.debouncer_threshold = debouncer_threshold
        self.model = None
        self.debouncer_buffer = dict.fromkeys(classes, 0)

        self.load_model(self.model_path)
        self.activated = False

        self.rd_hist_buffer = deque(maxlen=data_buffer_len)
        self.ra_hist_buffer = deque(maxlen=data_buffer_len)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, current_rd, currrent_ra):
        # preprocessing
        rd_cr, ra_cr = self.IndexPenRealTimePreprocessor.data_preprocessing(current_rd=current_rd,
                                                                            current_ra=currrent_ra)

        if rd_cr is not None and ra_cr is not None:
            # append to hist buffer
            self.rd_hist_buffer.append(rd_cr)
            self.ra_hist_buffer.append(ra_cr)

        if self.ra_hist_buffer.__len__() == self.ra_hist_buffer.maxlen:
            # start prediction
            if self.predict_interval_counter < self.predict_interval:
                self.predict_interval_counter += 1
                return None
            self.predict_interval_counter = 0

            start = time.time()
            y_pred_prob = self.model.predict(
                [np.expand_dims(np.array(self.rd_hist_buffer), 0),
                 np.expand_dims(np.array(self.ra_hist_buffer), 0)])
            y_pred_index = np.argmax(y_pred_prob, axis=1)[0]
            y_pred_class = self.classes[y_pred_index]
            cost = time.time() - start
            # print('Predict_label: ', y_pred_class, cost)

            if not self.activated:
                # wait for activation:
                if y_pred_class == 'Act':
                    self.debouncer_buffer['Act'] += 1
                else:
                    # reset dict to 0
                    self.reset_debouncer_buffer()

                if self.debouncer_buffer['Act'] >= self.debouncer_threshold:
                    self.activated = True
                    self.reset_debouncer_buffer()
                    print('Activated')
                    return y_pred_class
                else:
                    return None

            else:
                # start prediction after activated == True
                self.debouncer_buffer[y_pred_class] += 1

                # zero out all classes except the current prediction
                for indexPen_class in self.debouncer_buffer:
                    if indexPen_class != y_pred_class:
                        self.debouncer_buffer[indexPen_class] = 0

                if self.debouncer_buffer[y_pred_class] >= self.debouncer_threshold:
                    # if the char is  Act: deactivate and result, else reset and return
                    self.reset_debouncer_buffer()
                    if y_pred_class == 'Act':
                        self.activated = False
                        print('Deactivated')
                        return y_pred_class

                    print('detect', y_pred_class)
                    return y_pred_class

        else:
            print('Not enough sample to predict')
            return None

    def reset_debouncer_buffer(self):
        self.debouncer_buffer = dict.fromkeys(self.debouncer_buffer, 0)


class IndexPenRealTimePreprocessor:
    def __init__(self, data_buffer_len=3, rd_cr_ratio=0.8, ra_cr_ratio=0.8, rd_threshold=(-1000, 1500),
                 ra_threshold=(0, 2500)):
        self.data_buffer_len = data_buffer_len
        self.rd_buffer = deque(maxlen=data_buffer_len)
        self.ra_buffer = deque(maxlen=data_buffer_len)
        self.rd_cr_ratio = rd_cr_ratio
        self.ra_cr_ratio = ra_cr_ratio

        self.rd_threshold = rd_threshold
        self.ra_threshold = ra_threshold

        self.rd_clutter = None
        self.ra_clutter = None

    def data_preprocessing(self, current_rd, current_ra):
        # check index 1 data is corrupt or not
        self.rd_buffer.append(current_rd)
        self.ra_buffer.append(current_ra)
        if len(self.rd_buffer) == self.data_buffer_len:

            # corrupt frame removal
            self.rd_buffer = self.data_padding(self.rd_buffer, threshold=self.rd_threshold)
            self.ra_buffer = self.data_padding(self.ra_buffer, threshold=self.ra_threshold)

            # return index 0 data with clutter removal
            rd_cr_frame, self.rd_clutter = clutter_removal(self.rd_buffer[0], self.rd_clutter, self.rd_cr_ratio)
            ra_cr_frame, self.ra_clutter = clutter_removal(self.ra_buffer[0], self.ra_clutter, self.ra_cr_ratio)

            return rd_cr_frame, ra_cr_frame
        else:
            return None, None

    def data_padding(self, data_buffer, threshold):
        if is_broken_frame(data_buffer[1], min_threshold=threshold[0], max_threshold=threshold[1]) \
                and not is_broken_frame(data_buffer[2], min_threshold=threshold[0], max_threshold=threshold[1]):
            data_buffer[1] = (data_buffer[0] + data_buffer[2]) * 0.5
            print('broken frame pad with frame before and after')
        elif is_broken_frame(data_buffer[1], min_threshold=threshold[0], max_threshold=threshold[1]) \
                and is_broken_frame(data_buffer[2], min_threshold=threshold[0], max_threshold=threshold[1]):
            data_buffer[1] = data_buffer[0]
            print('two continuous borken frame, equalize with previous one')

        return data_buffer
