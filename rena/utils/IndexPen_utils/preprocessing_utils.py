import os
import numpy as np
import json
from utils.data_utils import RNStream, integer_one_hot, corrupt_frame_padding, time_series_static_clutter_removal
from sklearn.preprocessing import OneHotEncoder


def load_idp(data_dir_path, DataStreamName, reshape_dict, exp_info_dict_json_path, sample_num, rd_cr_ratio=None, ra_cr_ratio=None, all_categories=None):
    exp_info_dict = json.load(open(exp_info_dict_json_path))
    ExpID = exp_info_dict['ExpID']
    ExpLSLStreamName = exp_info_dict['ExpLSLStreamName']
    ExpStartMarker = exp_info_dict['ExpStartMarker']
    ExpEndMarker = exp_info_dict['ExpEndMarker']
    ExpLabelMarker = exp_info_dict['ExpLabelMarker']
    ExpInterruptMarker = exp_info_dict['ExpInterruptMarker']
    ExpErrorMarker = exp_info_dict['ExpErrorMarker']

    # one-hot encoder
    if all_categories is None:
        all_categories = list(ExpLabelMarker.values())
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.reshape(all_categories, (-1, 1)))

    X_dict = dict()
    Y = []

    for file_name in os.listdir(data_dir_path):
        file_path = os.path.join(data_dir_path, file_name)
        rs_stream = RNStream(file_path)

        # lsl_data = rs_stream.stream_in(only_stream=DataStreamName, reshape_stream_dict=reshape_dict,
        #                                jitter_removal=False)
        # marker_data = rs_stream.stream_in(only_stream=ExpLSLStreamName, reshape_stream_dict=reshape_dict,
        #                                   jitter_removal=False)
        #
        # data = {}
        # data[DataStreamName] = lsl_data[DataStreamName]
        # data[marker_data] = marker_data[ExpLSLStreamName]
        #

        data = rs_stream.stream_in(reshape_stream_dict=reshape_dict, ignore_stream=('monitor1', '0'), jitter_removal=False)
        data[DataStreamName][0][0] = np.moveaxis(data[DataStreamName][0][0], -1, 0)
        data[DataStreamName][0][1] = np.moveaxis(data[DataStreamName][0][1], -1, 0)

        # corrupt frame removal
        data[DataStreamName][0][0] = corrupt_frame_padding(data[DataStreamName][0][0], min_threshold=-1000, max_threshold=1500, frame_channel_first=True)
        data[DataStreamName][0][1] = corrupt_frame_padding(data[DataStreamName][0][1], min_threshold=0, max_threshold=2500, frame_channel_first=True)

        # clutter removal
        if rd_cr_ratio:
            data[DataStreamName][0][0] = time_series_static_clutter_removal(data[DataStreamName][0][0],
                                                                            signal_clutter_ratio=rd_cr_ratio)
            print('rd_cr_ratio: ', rd_cr_ratio)
        if ra_cr_ratio:
            data[DataStreamName][0][1] = time_series_static_clutter_removal(data[DataStreamName][0][1],
                                                                            signal_clutter_ratio=ra_cr_ratio)
            print('ra_cr_ratio: ', ra_cr_ratio)

        index_buffer = []
        label_buffer = []

        event_markers = data[ExpLSLStreamName][0][0]
        session_start_marker_indexes = np.where(event_markers == ExpStartMarker)[0]

        for start_marker_index in session_start_marker_indexes:
            # forward track the event marker
            session_index_buffer = []
            session_label_buffer = []
            for index in range(start_marker_index + 1, len(event_markers)):
                # stop the forward tracking and go for the next session if interrupt Marker found
                if event_markers[index] == ExpInterruptMarker:
                    break
                elif event_markers[index] == ExpID:
                    break
                elif event_markers[index] == ExpEndMarker:
                    # only attach the event marker with regular exit
                    index_buffer.extend(session_index_buffer)
                    label_buffer.extend(session_label_buffer)
                    break

                # remove last element from the list
                if event_markers[index] == ExpErrorMarker and len(session_index_buffer) != 0:
                    del session_index_buffer[-1]
                    del session_label_buffer[-1]
                    continue

                session_index_buffer.append(index)
                session_label_buffer.append(event_markers[index])

        # get all useful timestamps using index list
        label_start_time_stamps = data[ExpLSLStreamName][1][index_buffer]
        # loop through each label time stamp and find starting index for each label

        label_start_time_stamp_indexes = []
        for time_stamp in label_start_time_stamps:  # index greater or equal to
            label_start_time_stamp_indexes.append(np.where(data[DataStreamName][1] > time_stamp)[0][0])

        # extract n frames for each stream after each time stamp

        for ts_index in label_start_time_stamp_indexes:
            for channel in data[DataStreamName][0]:
                if channel in X_dict:
                    # append_data = np.array(data[DataStreamName][0][channel][ts_index:ts_index + sample_num])
                    X_dict[channel] = np.concatenate([X_dict[channel],
                                                      np.expand_dims(np.array(data[DataStreamName][0][channel]
                                                                              [ts_index:ts_index + sample_num]), axis=0)
                                                      ])
                else:
                    X_dict[channel] = np.expand_dims(
                        np.array(data[DataStreamName][0][channel][ts_index:ts_index + sample_num]), axis=0)

        Y.extend(label_buffer)

    Y = encoder.transform(np.reshape(Y, (-1, 1))).toarray()

    return X_dict, Y, encoder


def load_idp_raw(data_file_path, DataStreamName, reshape_dict, exp_info_dict_json_path, rd_cr_ratio=None,
                ra_cr_ratio=None, all_categories=None, session_only=False):
    exp_info_dict = json.load(open(exp_info_dict_json_path))
    ExpID = exp_info_dict['ExpID']
    ExpLSLStreamName = exp_info_dict['ExpLSLStreamName']
    ExpStartMarker = exp_info_dict['ExpStartMarker']
    ExpEndMarker = exp_info_dict['ExpEndMarker']
    ExpLabelMarker = exp_info_dict['ExpLabelMarker']
    ExpInterruptMarker = exp_info_dict['ExpInterruptMarker']
    ExpErrorMarker = exp_info_dict['ExpErrorMarker']

    # one-hot encoder
    if all_categories is None:
        all_categories = list(ExpLabelMarker.values())
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.reshape(all_categories, (-1, 1)))

    rs_stream = RNStream(data_file_path)

    data = rs_stream.stream_in(reshape_stream_dict=reshape_dict, ignore_stream=('monitor1', '0'), jitter_removal=False)
    data[DataStreamName][0][0] = np.moveaxis(data[DataStreamName][0][0], -1, 0)
    data[DataStreamName][0][1] = np.moveaxis(data[DataStreamName][0][1], -1, 0)

    # corrupt frame removal
    data[DataStreamName][0][0] = corrupt_frame_padding(data[DataStreamName][0][0], min_threshold=-1000,
                                                       max_threshold=1500, frame_channel_first=True)
    data[DataStreamName][0][1] = corrupt_frame_padding(data[DataStreamName][0][1], min_threshold=0, max_threshold=2500,
                                                       frame_channel_first=True)

    # clutter removal
    if rd_cr_ratio:
        data[DataStreamName][0][0] = time_series_static_clutter_removal(data[DataStreamName][0][0],
                                                                        signal_clutter_ratio=rd_cr_ratio)
        print('rd_cr_ratio: ', rd_cr_ratio)
    if ra_cr_ratio:
        data[DataStreamName][0][1] = time_series_static_clutter_removal(data[DataStreamName][0][1],
                                                                        signal_clutter_ratio=ra_cr_ratio)
        print('ra_cr_ratio: ', ra_cr_ratio)


    if session_only:
        event_markers = data[ExpLSLStreamName][0][0]

        session_start_marker_index = np.where(event_markers == ExpStartMarker)[0]
        session_start_marker_index = np.array(session_start_marker_index)[-1]
        session_start_time_stamp = data[ExpLSLStreamName][1][session_start_marker_index]

        session_end_marker_index = np.where(event_markers == ExpEndMarker)[0]
        session_end_marker_index = np.array(session_end_marker_index)[-1]
        session_end_time_stamp = data[ExpLSLStreamName][1][session_end_marker_index]

        session_start_data_index = np.where(data[DataStreamName][1] > session_start_time_stamp)[0][0]
        session_end_data_index = np.where(data[DataStreamName][1] > session_end_time_stamp)[0][0]

        for stream in data[DataStreamName][0]:
            data[DataStreamName][0][stream] = data[DataStreamName][0][stream][session_start_data_index:session_end_data_index+120]
        data[DataStreamName][1] = data[DataStreamName][1][session_start_data_index:session_end_data_index+120]

    else:
        return data

    index_buffer = []
    label_buffer = []

    event_markers = data[ExpLSLStreamName][0][0]
    session_start_marker_indexes = np.where(event_markers == ExpStartMarker)[0]


    for start_marker_index in session_start_marker_indexes:
        # forward track the event marker
        session_index_buffer = []
        session_label_buffer = []
        for index in range(start_marker_index + 1, len(event_markers)):
            # stop the forward tracking and go for the next session if interrupt Marker found
            if event_markers[index] == ExpInterruptMarker:
                break
            elif event_markers[index] == ExpID:
                break
            elif event_markers[index] == ExpEndMarker:
                # only attach the event marker with regular exit
                index_buffer.extend(session_index_buffer)
                label_buffer.extend(session_label_buffer)
                break

            # remove last element from the list
            if event_markers[index] == ExpErrorMarker and len(session_index_buffer) != 0:
                del session_index_buffer[-1]
                del session_label_buffer[-1]
                continue

            session_index_buffer.append(index)
            session_label_buffer.append(event_markers[index])

    # get all useful timestamps using index list
    label_start_time_stamps = data[ExpLSLStreamName][1][index_buffer]
    # loop through each label time stamp and find starting index for each label

    label_start_time_stamp_indexes = []
    for time_stamp in label_start_time_stamps:  # index greater or equal to
        label_start_time_stamp_indexes.append(np.where(data[DataStreamName][1] > time_stamp)[0][0])

    # return rd_map_series, ra_map_series, ts, label_start_time_stamp_indexes
    return data[DataStreamName][0], \
           data[DataStreamName][1], \
           label_buffer, \
           label_start_time_stamp_indexes


def load_idp_file(file_path, DataStreamName, reshape_dict, exp_info_dict_json_path, sample_num, rd_cr_ratio=None, ra_cr_ratio=None, all_categories=None):
    exp_info_dict = json.load(open(exp_info_dict_json_path))
    ExpID = exp_info_dict['ExpID']
    ExpLSLStreamName = exp_info_dict['ExpLSLStreamName']
    ExpStartMarker = exp_info_dict['ExpStartMarker']
    ExpEndMarker = exp_info_dict['ExpEndMarker']
    ExpLabelMarker = exp_info_dict['ExpLabelMarker']
    ExpInterruptMarker = exp_info_dict['ExpInterruptMarker']
    ExpErrorMarker = exp_info_dict['ExpErrorMarker']

    # one-hot encoder
    if all_categories is None:
        all_categories = list(ExpLabelMarker.values())
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.reshape(all_categories, (-1, 1)))

    X_dict = dict()
    Y = []



    rs_stream = RNStream(file_path)

    # lsl_data = rs_stream.stream_in(only_stream=DataStreamName, reshape_stream_dict=reshape_dict,
    #                                jitter_removal=False)
    # marker_data = rs_stream.stream_in(only_stream=ExpLSLStreamName, reshape_stream_dict=reshape_dict,
    #                                   jitter_removal=False)
    #
    # data = {}
    # data[DataStreamName] = lsl_data[DataStreamName]
    # data[marker_data] = marker_data[ExpLSLStreamName]
    #

    data = rs_stream.stream_in(reshape_stream_dict=reshape_dict, ignore_stream=('monitor1', '0'), jitter_removal=False)
    data[DataStreamName][0][0] = np.moveaxis(data[DataStreamName][0][0], -1, 0)
    data[DataStreamName][0][1] = np.moveaxis(data[DataStreamName][0][1], -1, 0)

    # corrupt frame removal
    data[DataStreamName][0][0] = corrupt_frame_padding(data[DataStreamName][0][0], min_threshold=-1000, max_threshold=1500, frame_channel_first=True)
    data[DataStreamName][0][1] = corrupt_frame_padding(data[DataStreamName][0][1], min_threshold=0, max_threshold=2500, frame_channel_first=True)

    # clutter removal
    if rd_cr_ratio:
        data[DataStreamName][0][0] = time_series_static_clutter_removal(data[DataStreamName][0][0],
                                                                        signal_clutter_ratio=rd_cr_ratio)
        print('rd_cr_ratio: ', rd_cr_ratio)
    if ra_cr_ratio:
        data[DataStreamName][0][1] = time_series_static_clutter_removal(data[DataStreamName][0][1],
                                                                        signal_clutter_ratio=ra_cr_ratio)
        print('ra_cr_ratio: ', ra_cr_ratio)

    index_buffer = []
    label_buffer = []

    event_markers = data[ExpLSLStreamName][0][0]
    session_start_marker_indexes = np.where(event_markers == ExpStartMarker)[0]

    for start_marker_index in session_start_marker_indexes:
        # forward track the event marker
        session_index_buffer = []
        session_label_buffer = []
        for index in range(start_marker_index + 1, len(event_markers)):
            # stop the forward tracking and go for the next session if interrupt Marker found
            if event_markers[index] == ExpInterruptMarker:
                break
            elif event_markers[index] == ExpID:
                break
            elif event_markers[index] == ExpEndMarker:
                # only attach the event marker with regular exit
                index_buffer.extend(session_index_buffer)
                label_buffer.extend(session_label_buffer)
                break

            # remove last element from the list
            if event_markers[index] == ExpErrorMarker and len(session_index_buffer) != 0:
                del session_index_buffer[-1]
                del session_label_buffer[-1]
                continue

            session_index_buffer.append(index)
            session_label_buffer.append(event_markers[index])

    # get all useful timestamps using index list
    label_start_time_stamps = data[ExpLSLStreamName][1][index_buffer]
    # loop through each label time stamp and find starting index for each label

    label_start_time_stamp_indexes = []
    for time_stamp in label_start_time_stamps:  # index greater or equal to
        label_start_time_stamp_indexes.append(np.where(data[DataStreamName][1] > time_stamp)[0][0])

    # extract n frames for each stream after each time stamp

    for ts_index in label_start_time_stamp_indexes:
        for channel in data[DataStreamName][0]:
            if channel in X_dict:
                # append_data = np.array(data[DataStreamName][0][channel][ts_index:ts_index + sample_num])
                X_dict[channel] = np.concatenate([X_dict[channel],
                                                  np.expand_dims(np.array(data[DataStreamName][0][channel]
                                                                          [ts_index:ts_index + sample_num]), axis=0)
                                                  ])
            else:
                X_dict[channel] = np.expand_dims(
                    np.array(data[DataStreamName][0][channel][ts_index:ts_index + sample_num]), axis=0)

    Y.extend(label_buffer)

    Y = encoder.transform(np.reshape(Y, (-1, 1))).toarray()

    return X_dict, Y, encoder
