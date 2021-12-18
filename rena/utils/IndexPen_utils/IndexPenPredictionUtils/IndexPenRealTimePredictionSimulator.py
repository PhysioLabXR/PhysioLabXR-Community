import os
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy
import numpy as np
import tensorflow as tf

import config_signal
from utils.IndexPen_utils.IndexPenPredictionUtils.IndexPenPredictor import IndexPenRealTimePredictor
from utils.data_utils import RNStream, replace_special, levenshtein_ratio_and_distance
from utils.IndexPen_utils.preprocessing_utils import load_idp_raw
from utils.data_utils import is_broken_frame, clutter_removal, corrupt_frame_padding, RNStream

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# class IndexPenRealTimePredictionSimulator:
#     def __init__(self, model_path=None, debouncer_threshold=60, data_buffer_len=120):
#         self.model_path = model_path
#         self.IndexPenRealTimePredictor = IndexPenRealTimePredictor(model_path=os.path.abspath(model_path),
#                                                                    classes=config_signal.indexpen_classes,
#                                                                    debouncer_threshold=debouncer_threshold,
#                                                                    data_buffer_len=data_buffer_len)
#
#     def realtime_prediction_sim(self, data_path, DataStreamName = 'TImmWave_6843AOP'):
#         # load data with load data dict
#         reshape_dict = {
#             'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
#         }
#         data = rs_stream.stream_in(reshape_stream_dict=reshape_dict, jitter_removal=False)
#         data[DataStreamName][0][0] = np.moveaxis(data[DataStreamName][0][0], -1, 0)
#         data[DataStreamName][0][1] = np.moveaxis(data[DataStreamName][0][1], -1, 0)
#
#         data[DataStreamName][0][0] = corrupt_frame_padding(data[DataStreamName][0][0], min_threshold=-1000, max_threshold=1500, frame_channel_first=True)
#         data[DataStreamName][0][1] = corrupt_frame_padding(data[DataStreamName][0][1], min_threshold=0, max_threshold=2500, frame_channel_first=True)
#
#         if rd_cr_ratio:
#             data[DataStreamName][0][0] = time_series_static_clutter_removal(data[DataStreamName][0][0],
#                                                                             signal_clutter_ratio=rd_cr_ratio)
#             print('rd_cr_ratio: ', rd_cr_ratio)
#         if ra_cr_ratio:
#             data[DataStreamName][0][1] = time_series_static_clutter_removal(data[DataStreamName][0][1],
#                                                                             signal_clutter_ratio=ra_cr_ratio)
#             print('ra_cr_ratio: ', ra_cr_ratio)
#
#         return None
DataStreamName = 'TImmWave_6843AOP'
# data_dir_path = 'C:\Recordings\John_F-J'
data_file_path = 'C:/Users/Haowe/OneDrive/Desktop/IndexPen_User_Study_Data/IndexPen_6000_Samples/Sub1_hw/Senario_2_Office/Day2/07_11_2021_18_42_10-Exp_Senario_2_Office-Sbj_hw-Ssn_3.dats'
# data_dir_path = 'C:/Recordings/John_A-J_test/2'

exp_info_dict_json_path = 'C:/Users/Haowe/PycharmProjects/RealityNavigation/utils/IndexPen_utils/IndexPenExp.json'
reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
}

indexpen_raw, time_stamp, labels, label_start_time_stamp_indexes = load_idp_raw(
    data_file_path=data_file_path,
    DataStreamName=DataStreamName,
    reshape_dict=reshape_dict,
    exp_info_dict_json_path=exp_info_dict_json_path,
    rd_cr_ratio=0.8,
    ra_cr_ratio=0.8,
    all_categories=None,
    session_only=True
)
rd_map_series = indexpen_raw[0]
ra_map_series = indexpen_raw[1]

labels_index = np.array(labels).astype(int)-1
grdt_chars = np.array(config_signal.indexpen_classes)[labels_index]

detect_chars_buffer = []
pred_prob_hist_buffer = None

model_path = os.path.abspath('../../../resource/mmWave/indexPen_model/2021-07-17_22-18-53.145732.h5')
model = tf.keras.models.load_model(model_path)

rd_hist_buffer = deque(maxlen=120)
ra_hist_buffer = deque(maxlen=120)
debouncer = np.zeros(31)
debouncerFrameThreshold = 60
debouncerProbThreshold = 0.8

relaxPeriod = 5
relaxCounter = 0
inactivateClearThreshold = 10

for index in range(0, 400):
    # push in sample
    rd_hist_buffer.append(rd_map_series[index])
    ra_hist_buffer.append(ra_map_series[index])
    print(index)
    if rd_hist_buffer.__len__() == rd_hist_buffer.maxlen:
        # start prediction
        if relaxCounter == relaxPeriod:

            yPred = model.predict(
                [np.expand_dims(np.array(rd_hist_buffer), 0),
                 np.expand_dims(np.array(ra_hist_buffer), 0)])

            # add to history
            if index == 120 + relaxPeriod - 1:
                pred_prob_hist_buffer = yPred
            else:
                pred_prob_hist_buffer = numpy.append(pred_prob_hist_buffer, yPred, axis=0)

            # yPred = yPred[0]
            breakIndices = np.argwhere(yPred >= debouncerProbThreshold)
            debouncer[breakIndices[:, 1]] += 1

            detects = np.argwhere(np.array(debouncer) >= debouncerFrameThreshold)

            # zero out the debouncer that inactivated for debouncerProbThreshold frames

            if len(detects) > 0:
                print(detects)
                detect_char = config_signal.indexpen_classes[detects[0][0]]
                print(detect_char)
                detect_chars_buffer.append(detect_char)
                debouncer = np.zeros(31)
                relaxCounter = 0

        else:
            relaxCounter += 1

pred_prob_hist_buffer = pred_prob_hist_buffer.transpose()

existing_char_only = True

plotted_char_prob = []


fig = plt.figure(figsize=(60, 10))
ax = fig.add_subplot(111)

for index, char in enumerate(config_signal.indexpen_classes):
    if existing_char_only:
        if char in grdt_chars:
            ax.plot(pred_prob_hist_buffer[index], label=char)
            plotted_char_prob.append(config_signal.indexpen_classes[index])
        else:
            pass
    else:
        ax.plot(pred_prob_hist_buffer[index], label=char)
        plotted_char_prob.append(config_signal.indexpen_classes[index])

# ax.legend(bbox_to_anchor=(1.1, 1), loc=5, borderaxespad=0.)
ax.set_aspect(aspect=80)
plt.show()



# levenshtein_ratio_and_distance
special_replacement = {'Act': '0', 'Spc': '1', 'Bspc': '2', 'Ent': '3', 'Nois': '4'}

pred_string = replace_special(''.join(detect_chars_buffer), special_replacement)
grdt_string = replace_special(''.join(grdt_chars), special_replacement)

str_dist = levenshtein_ratio_and_distance(pred_string, grdt_string, ratio_calc=True)






