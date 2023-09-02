import time
from collections import deque

import numpy as np
import tensorflow as tf

from physiolabxr.scripting.RenaScript import RenaScript


class IndexPen(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        self.rd_hist_buffer = deque(maxlen=120)
        self.ra_hist_buffer = deque(maxlen=120)
        self.IndexPenRealTimePreprocessor = IndexPenRealTimePreprocessor()
        self._interpreter = tf.lite.Interpreter(model_path=self.params['model_path'])
        self._interpreter.allocate_tensors()
        self.input1_index = self._interpreter.get_input_details()[0]["index"]
        self.input2_index = self._interpreter.get_input_details()[1]["index"]
        self.output_index = self._interpreter.get_output_details()[0]["index"]
        print('Successfully load IndexPen Model: ', self.params['model_path'])

    # loop is called <Run Frequency> times per second
    def loop(self):
        for frame in self.inputs['TImmWave_6843AOP'].transpose():
            current_rd = np.array(frame[0:128]).reshape((8, 16))
            current_ra = np.array(frame[128:640]).reshape((8, 64))
            rd_cr, ra_cr = self.IndexPenRealTimePreprocessor.data_preprocessing(current_rd=current_rd,
                                                                                current_ra=current_ra)
            if rd_cr is not None and ra_cr is not None:
                # append to hist buffer
                self.rd_hist_buffer.append(rd_cr)
                self.ra_hist_buffer.append(ra_cr)

        if self.ra_hist_buffer.__len__() == self.ra_hist_buffer.maxlen:
            self._interpreter.set_tensor(self.input1_index,
                                         np.expand_dims(np.array(self.rd_hist_buffer), axis=(0, -1)).astype(
                                             np.float32))
            self._interpreter.set_tensor(self.input2_index,
                                         np.expand_dims(np.array(self.ra_hist_buffer), axis=(0, -1)).astype(
                                             np.float32))
            self._interpreter.invoke()
            # print('Invoking duration: ', invoke_start_time - time.time())

            output = self._interpreter.tensor(self.output_index)
            soft_max_out = np.array(output()[0])
            self.outputs['indexpen'] = soft_max_out

    def cleanup(self):
        print('Cleanup function is called')


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
            self.rd_buffer = data_padding(self.rd_buffer, threshold=self.rd_threshold)
            self.ra_buffer = data_padding(self.ra_buffer, threshold=self.ra_threshold)

            # return index 0 data with clutter removal
            rd_cr_frame, self.rd_clutter = clutter_removal(self.rd_buffer[0], self.rd_clutter, self.rd_cr_ratio)
            ra_cr_frame, self.ra_clutter = clutter_removal(self.ra_buffer[0], self.ra_clutter, self.ra_cr_ratio)

            return rd_cr_frame, ra_cr_frame
        else:
            return None, None

def data_padding(data_buffer, threshold):
    if is_broken_frame(data_buffer[1], min_threshold=threshold[0], max_threshold=threshold[1]) \
            and not is_broken_frame(data_buffer[2], min_threshold=threshold[0], max_threshold=threshold[1]):
        data_buffer[1] = (data_buffer[0] + data_buffer[2]) * 0.5
        print('broken frame pad with frame before and after')
    elif is_broken_frame(data_buffer[1], min_threshold=threshold[0], max_threshold=threshold[1]) \
            and is_broken_frame(data_buffer[2], min_threshold=threshold[0], max_threshold=threshold[1]):
        data_buffer[1] = data_buffer[0]
        print('two continuous borken frame, equalize with previous one')
    return data_buffer

def clutter_removal(cur_frame, clutter, signal_clutter_ratio):
    if clutter is None:
        clutter = cur_frame
    else:
        clutter = signal_clutter_ratio * clutter + (1 - signal_clutter_ratio) * cur_frame
    return cur_frame - clutter, clutter

def is_broken_frame(frame, min_threshold=np.NINF, max_threshold=np.PINF):
    if np.min(frame) < min_threshold or np.max(frame) > max_threshold:
        return True
    else:
        return False

