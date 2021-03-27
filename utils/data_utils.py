import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import resample

from utils.sig_proc_utils import notch_filter, baseline_correction, butter_bandpass_filter


def window_slice(data, window_size, stride, channel_mode='channel_last'):
    assert len(data.shape) == 2
    if channel_mode == 'channel_first':
        data = np.transpose(data)
    elif channel_mode == 'channel_last':
        pass
    else:
        raise Exception('Unsupported channel mode')
    assert window_size <= len(data)
    assert stride > 0
    rtn = np.expand_dims(data, axis=0) if window_size == len(data) else []
    for i in range(window_size, len(data), stride):
        rtn.append(data[i - window_size:i])
    return np.array(rtn)


# constant
magic = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
max_label_len = 32
max_dtype_len = 8
dim_bytes_len = 8
shape_bytes_len = 8

endianness = 'little'
encoding = 'utf-8'

ts_dtype = 'float64'


class RNStream:
    def __init__(self, file_path):
        self.fn = file_path

    def stream_out(self, buffer):
        out_file = open(self.fn, "ab")
        stream_label_bytes, dtype_bytes, dim_bytes, shape_bytes, data_bytes, ts_bytes = \
            b'', b'', b'', b'', b'', b''
        for stream_label, data_ts_array in buffer.items():
            data_array, ts_array = data_ts_array[0], data_ts_array[1]
            stream_label_bytes = \
                bytes(stream_label[:max_label_len] + "".join(
                    " " for x in range(max_label_len - len(stream_label))), encoding)
            try:
                dtype_str = str(data_array.dtype)
                assert len(dtype_str) < max_dtype_len
            except AssertionError:
                raise Exception('dtype encoding exceeds 8 characters, please contact support')
            dtype_bytes = bytes(dtype_str + "".join(" " for x in range(max_dtype_len - len(dtype_str))),
                                encoding)
            try:
                dim_bytes = len(data_array.shape).to_bytes(dim_bytes_len, 'little')
                shape_bytes = b''.join(
                    [s.to_bytes(shape_bytes_len, 'little') for s in data_array.shape])  # the last axis is time
            except OverflowError:
                raise Exception('RN requires its stream to have number of dimensions less than 2^40, '
                                'and the size of any dimension to be less than the same number ')
            data_bytes = data_array.tobytes()
            ts_bytes = ts_array.tobytes()
            out_file.write(magic)
            out_file.write(stream_label_bytes)
            out_file.write(dtype_bytes)
            out_file.write(dim_bytes)
            out_file.write(shape_bytes)
            out_file.write(data_bytes)
            out_file.write(ts_bytes)
        out_file.close()
        return len(magic + stream_label_bytes + dtype_bytes + dim_bytes + shape_bytes + data_bytes + ts_bytes)

    def stream_in(self, ignore_stream=None, only_stream=None, jitter_removal=True):
        """
        different from LSL XDF importer, this jitter removal assumes no interruption in the data
        :param ignore_stream:
        :param only_stream:
        :param jitter_removal:
        :return:
        """
        total_bytes = float(os.path.getsize(self.fn))  # use floats to avoid scalar type overflow
        buffer = {}
        read_bytes_count = 0.
        with open(self.fn, "rb") as file:
            while True:
                print('Streaming in progress {0}%'.format(str(round(100 * read_bytes_count / total_bytes, 2))), sep=' ',
                      end='\r', flush=True)
                # read magic
                read_bytes = file.read(len(magic))
                read_bytes_count += len(read_bytes)
                if len(read_bytes) == 0:
                    break
                try:
                    assert read_bytes == magic
                except AssertionError:
                    raise Exception('Data invalid, magic sequence not found')
                # read stream_label
                read_bytes = file.read(max_label_len)
                read_bytes_count += len(read_bytes)
                stream_name = str(read_bytes, encoding).strip(' ')
                # read read_bytes
                read_bytes = file.read(max_dtype_len)
                read_bytes_count += len(read_bytes)
                stream_dytpe = str(read_bytes, encoding).strip(' ')
                # read number of dimensions
                read_bytes = file.read(dim_bytes_len)
                read_bytes_count += len(read_bytes)
                dims = int.from_bytes(read_bytes, 'little')
                # read number of np shape
                shape = []
                for i in range(dims):
                    read_bytes = file.read(shape_bytes_len)
                    read_bytes_count += len(read_bytes)
                    shape.append(int.from_bytes(read_bytes, 'little'))

                data_array_num_bytes = np.prod(shape) * np.dtype(stream_dytpe).itemsize
                timestamp_array_num_bytes = shape[-1] * np.dtype(ts_dtype).itemsize

                this_in_only_stream = (stream_name in only_stream) if only_stream else True
                not_ignore_this_stream = (stream_name not in ignore_stream) if ignore_stream else True
                if not_ignore_this_stream and this_in_only_stream:
                    # read data array
                    read_bytes = file.read(data_array_num_bytes)
                    read_bytes_count += len(read_bytes)
                    data_array = np.frombuffer(read_bytes, dtype=stream_dytpe)
                    data_array = np.reshape(data_array, newshape=shape)
                    # read timestamp array
                    read_bytes = file.read(timestamp_array_num_bytes)
                    ts_array = np.frombuffer(read_bytes, dtype=ts_dtype)

                    if stream_name not in buffer.keys():
                        buffer[stream_name] = [np.empty(shape=tuple(shape[:-1]) + (0,), dtype=stream_dytpe),
                                               np.empty(shape=(0,))]  # data first, timestamps second
                    buffer[stream_name][0] = np.concatenate([buffer[stream_name][0], data_array], axis=-1)
                    buffer[stream_name][1] = np.concatenate([buffer[stream_name][1], ts_array])
                else:
                    file.read(data_array_num_bytes + timestamp_array_num_bytes)
                    read_bytes_count += data_array_num_bytes + timestamp_array_num_bytes
        if jitter_removal:
            i = 1
            for stream_name, (d_array, ts_array) in buffer.items():
                print('Removing jitter for streams {0}/{1}'.format(i, len(buffer)), sep=' ',
                      end='\r', flush=True)
                coefs = np.polyfit(list(range(len(ts_array))), ts_array, 1)
                smoothed_ts_array = np.array([i * coefs[0] + coefs[1] for i in range(len(ts_array))])
                buffer[stream_name][1] = smoothed_ts_array

        print("Stream-in completed: {0}".format(self.fn))
        return buffer

    def get_stream_names(self):
        total_bytes = float(os.path.getsize(self.fn))  # use floats to avoid scalar type overflow
        stream_names = []
        read_bytes_count = 0.
        with open(self.fn, "rb") as file:
            while True:
                print('Scanning stream in progress {}%'.format(str(round(100 * read_bytes_count / total_bytes, 2))),
                      sep=' ', end='\r', flush=True)
                # read magic
                read_bytes = file.read(len(magic))
                read_bytes_count += len(read_bytes)
                if len(read_bytes) == 0:
                    break
                try:
                    assert read_bytes == magic
                except AssertionError:
                    raise Exception('Data invalid, magic sequence not found')
                # read stream_label
                read_bytes = file.read(max_label_len)
                read_bytes_count += len(read_bytes)
                stream_label = str(read_bytes, encoding).strip(' ')
                # read read_bytes
                read_bytes = file.read(max_dtype_len)
                read_bytes_count += len(read_bytes)
                stream_dytpe = str(read_bytes, encoding).strip(' ')
                # read number of dimensions
                read_bytes = file.read(dim_bytes_len)
                read_bytes_count += len(read_bytes)
                dims = int.from_bytes(read_bytes, 'little')
                # read number of np shape
                shape = []
                for i in range(dims):
                    read_bytes = file.read(shape_bytes_len)
                    read_bytes_count += len(read_bytes)
                    shape.append(int.from_bytes(read_bytes, 'little'))

                data_array_num_bytes = np.prod(shape) * np.dtype(stream_dytpe).itemsize
                timestamp_array_num_bytes = shape[-1] * np.dtype(ts_dtype).itemsize

                file.read(data_array_num_bytes + timestamp_array_num_bytes)
                read_bytes_count += data_array_num_bytes + timestamp_array_num_bytes

                stream_names.append(stream_label)
        print("Scanning stream completed: {0}".format(self.fn))
        return stream_names

    def generate_video(self, video_stream_name, output_path=''):
        """
        if output path is not specified, the output video will be place in the same directory as the
        stream .dats file with a tag to its stream name
        :param stream_name:
        :param output_path:
        """
        print('Load video stream...')
        data_fn = self.fn.split('/')[-1]
        data_root = Path(self.fn).parent.absolute()
        data = self.stream_in(only_stream=(video_stream_name,))

        video_frame_stream = data[video_stream_name][0]
        frame_count = video_frame_stream.shape[-1]

        timestamp_stream = data[video_stream_name][1]
        frate = len(timestamp_stream) / (timestamp_stream[-1] - timestamp_stream[0])
        try:
            assert len(video_frame_stream.shape) == 4 and video_frame_stream.shape[2] == 3
        except AssertionError:
            raise Exception('target stream is not a video stream. It does not have 4 dims (height, width, color, time)'
                            'and/or the number of its color channel does not equal 3.')
        frame_size = (data[video_stream_name][0].shape[1], data[video_stream_name][0].shape[0])
        output_path = os.path.join(data_root, '{0}_{1}.avi'.format(data_fn.split('.')[0],
                                                                   video_stream_name)) if output_path == '' else output_path

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), frate, frame_size)

        for i in range(frame_count):
            print('Creating video progress {}%'.format(str(round(100 * i / frame_count, 2))), sep=' ', end='\r',
                  flush=True)
            img = video_frame_stream[:, :, :, i]
            # img = np.reshape(img, newshape=list(frame_size) + [-1,])
            out.write(img)

        out.release()


def plot_stream(stream, timestamps):
    timestamps = timestamps - timestamps[0]  # baseline the timestamps
    plt.plot(timestamps, stream)
    plt.xlabel('Time (sec)')
    plt.show()


def modify_indice_to_cover(i1, i2, coverage, tolerance=3):
    assert i1 < i2
    assert abs(coverage - (i2 - i1)) <= tolerance
    is_modifying_i1 = True
    if i2 - i1 > coverage:
        while i2 - i1 != coverage:
            if is_modifying_i1:
                i1 += 1
            else:
                i2 -= 1
        print('Modified')

    elif i2 - i1 < coverage:
        while i2 - i1 != coverage:
            if is_modifying_i1:
                i1 -= 1
            else:
                i2 += 1
        print('Modified')

    return i1, i2


def process_data(file_path, EM_stream_name, EEG_stream_name, target_labels, pre_stimulus_time, post_stimulus_time,
                 EEG_stream_preset, lowcut=0.5, highcut=50., notch_band_demoninator=200, EEG_fresample=50,
                 baselining=True):
    EEG_num_sample_per_trail = int(EEG_stream_preset['NominalSamplingRate'] * (post_stimulus_time - pre_stimulus_time))
    EEG_num_sample_per_trail_RESAMPLED = int(EEG_fresample * (post_stimulus_time - pre_stimulus_time))
    EEG_num_chan = EEG_stream_preset['GroupChannelsInPlot'][1] - EEG_stream_preset['GroupChannelsInPlot'][0]
    epoched_EEG = np.empty(shape=(0, EEG_num_chan, EEG_num_sample_per_trail))
    biosemi_64_montage = mne.channels.make_standard_montage('biosemi64')
    EEG_chan_names = biosemi_64_montage.ch_names
    events = np.empty(shape=(0, 3), dtype=int)
    reject = dict(eeg=500.)
    _evoked_target_all_session = []
    _evoked_nontarget_all_session = []

    target_count = 0
    nontarget_count = 0

    target_label = 3
    nontarget_label = [2,4,5,6]
    for fp in file_path:

        rns = RNStream(fp)
        data = rns.stream_in(ignore_stream=('monitor1', '0'))
        # plot_stream(data['Unity.VisualSearch.EventMarkers'][0][-1, :], data['Unity.VisualSearch.EventMarkers'][1])
        # plot_stream(data['Unity.RotationWheel.EventMarkers'][0][-1, :], data['Unity.RotationWheel.EventMarkers'][1])

        # get all needed streams ##########################################################################
        '''
        EM = Event Marker
        '''
        stream_EM = data[EM_stream_name][0]
        timestamps_EM = data[EM_stream_name][1]
        stream_EEG = data[EEG_stream_name][0]
        timestamps_EEG = data[EEG_stream_name][1]

        array_event_label = stream_EM[-1, :]

        # event label sanity check #############################################################################################
        # target_label = 1
        # target_onset_em = np.logical_and(event_label_stream == target_label, np.concatenate([np.array([0]), np.diff(event_label_stream)]) != 0)
        # plt.scatter(timestamps_stream, target_onset_em, c='r')
        #
        # target_label = 3
        # target_onset_em = np.logical_and(event_label_stream == target_label, np.concatenate([np.array([0]), np.diff(event_label_stream)]) != 0)
        # plt.scatter(timestamps_stream, target_onset_em, c='b')
        # plt.show()

        # take out the electrode channels
        array_EEG = stream_EEG[
                    EEG_stream_preset['GroupChannelsInPlot'][0]:EEG_stream_preset['GroupChannelsInPlot'][1],
                    :]
        # not worrying about the timeoffset in mne data structure as we use our own timestamps
        array_EEG = mne.io.RawArray(array_EEG, mne.create_info(EEG_chan_names, EEG_stream_preset['NominalSamplingRate'],
                                                               ch_types='eeg'))
        array_EEG.set_montage(biosemi_64_montage)
        # notch filter
        # stream_EEG_preprocessed = notch_filter(stream_EEG_preprocessed, notch_f0, notch_f0 / notch_band_demoninator,
        #                                        EEG_stream_preset['NominalSamplingRate'], channel_format='first')
        array_EEG, _ = mne.set_eeg_reference(array_EEG, 'average',
                                             projection=False)  # re-reference using average as the reference
        # bandpass filter
        array_EEG = array_EEG.filter(l_freq=lowcut, h_freq=highcut)  # re-reference using average as the reference

        # stream_EEG_preprocessed = butter_bandpass_filter(stream_EEG_preprocessed, lowcut=lowcut, highcut=highcut, fs=EEG_stream_preset['NominalSamplingRate'])

        # [plt.plot(timestamps_EEG[:15000], array_EEG.get_data()[i, :15000]) for i in range(EEG_num_chan)]
        # plt.show()
        # if baselining:
        #     print('Performing baseline correction, this may take a while')
        #     stream_EEG_preprocessed = baseline_correction(stream_EEG_preprocessed, lam=10, p=0.05)

        for tl in target_labels:
            array_target_onset_EM = np.logical_and(array_event_label == tl,
                                                   np.concatenate([np.array([0]), np.diff(array_event_label)]) != 0)
            print('Number of trials is {0} for label {1}'.format(np.count_nonzero(array_target_onset_EM), tl))
            array_target_PRE_onset_EM_timestamps = timestamps_EM[array_target_onset_EM] + pre_stimulus_time
            array_target_onset_EM_timestamps = timestamps_EM[array_target_onset_EM]
            array_target_POST_onset_EM_timestamps = timestamps_EM[array_target_onset_EM] + post_stimulus_time
            array_target_onset_EM_indices = np.argwhere(array_target_onset_EM)[:, 0]

            # find the nearest timestamp index in eeg #####################################################
            array_target_PRE_onset_EEG_indices = np.array(
                [(np.abs(timestamps_EEG - x)).argmin() for x in array_target_PRE_onset_EM_timestamps])
            array_target_onset_EEG_indices = np.array(
                [(np.abs(timestamps_EEG - x)).argmin() for x in array_target_onset_EM_timestamps])
            array_target_POST_onset_EEG_indices = np.array(
                [(np.abs(timestamps_EEG - x)).argmin() for x in array_target_POST_onset_EM_timestamps])
            array_target_onset_EEG_timestamps = timestamps_EEG[array_target_onset_EEG_indices]
            print('EM-EEG target onset timestamp discrepency: mean {0}, std {1}'.format(
                np.mean(array_target_onset_EM_timestamps - array_target_onset_EEG_timestamps),
                np.std(array_target_onset_EM_timestamps - array_target_onset_EEG_timestamps)))

            # epoch eeg data #############################################
            # modify pre and post indices for possible remaining jitter

            array_prepost_target_onset_i = [modify_indice_to_cover(pre_onset_i, post_onset_i, EEG_num_sample_per_trail)
                                            for
                                            pre_onset_i, post_onset_i in
                                            zip(array_target_PRE_onset_EEG_indices,
                                                array_target_POST_onset_EEG_indices)]

            # epoching using mne
            # manually construct event array based on https://mne.tools/stable/generated/mne.find_events.html#mne.find_events
            ems = np.ones(shape=array_target_onset_EEG_indices.shape) * tl
            new_events = np.array(
                [array_target_onset_EEG_indices, array_event_label[array_target_onset_EM_indices - 1], ems],
                dtype=int).T
            events = np.concatenate([events, new_events], axis=0)
        events = np.sort(events, axis=0)
        target_count += np.count_nonzero(events[:, 2] == target_label)
        nontarget_count += np.count_nonzero(np.isin(events[:, 2], nontarget_label))

        ##################################################################
        epochs_params = dict(events=events, event_id=target_label, tmin=pre_stimulus_time, tmax=post_stimulus_time,
                             baseline=(-0.2, 0.), reject=reject)
        evoked_target = mne.Epochs(raw=array_EEG, **epochs_params)
        _evoked_target_all_session.append(evoked_target)
        epochs_params = dict(events=events, event_id=nontarget_label, tmin=pre_stimulus_time, tmax=post_stimulus_time,
                             baseline=(-0.2, 0.), reject=reject)
        evoked_nontarget = mne.Epochs(raw=array_EEG, **epochs_params)
        _evoked_nontarget_all_session.append(evoked_nontarget)

        # epoched_EEG_new = np.array([stream_EEG_preprocessed[:, pre_onset_i:post_onset_i] for pre_onset_i, post_onset_i in
        #                         array_prepost_target_onset_i])
        # epoched_EEG = np.concatenate([epoched_EEG, epoched_EEG_new], axis=0)
        print('Total number of trials for label {0} is {1}'.format(str(target_labels), len(epoched_EEG)))

    evoked_target_all_session = mne.concatenate_epochs(_evoked_target_all_session).average().resample(sfreq=EEG_fresample)
    print('{0}/{1} was dropped for Target.'.format(target_count - mne.concatenate_epochs(_evoked_target_all_session).get_data().shape[0], target_count))
    print('---------------------loading nonTargets')
    evoked_nontarget_all_session = mne.concatenate_epochs(_evoked_nontarget_all_session).average().resample(sfreq=EEG_fresample)
    print('{0}/{1} was dropped for NonTarget.'.format(nontarget_count - mne.concatenate_epochs(_evoked_nontarget_all_session).get_data().shape[0], nontarget_count))

    title = 'EEG Targets'
    # evoked_target_all_session.plot(titles=dict(eeg=title), time_unit='s')
    evoked_target_all_session.plot_topomap(times=[0.3], size=3., title=title, time_unit='s', scalings=dict(eeg=1.))
    evoked_target_all_session.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-20, 20]), titles=dict(eeg=title), scalings=dict(eeg=1.))

    title = 'EEG Nontargets'
    # evoked_nontarget_all_session.plot(titles=dict(eeg=title), time_unit='s')
    evoked_nontarget_all_session.plot_topomap(times=[0.3], size=3., title=title, time_unit='s', scalings=dict(eeg=1.))
    evoked_nontarget_all_session.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-20, 20]), titles=dict(eeg=title), scalings=dict(eeg=1.))

    pass
    # # down sample the epoch eeg
    # epoched_EEG_RESAMPLED = resample(epoched_EEG, EEG_num_sample_per_trail_RESAMPLED, axis=-1)
    #
    # # averaging
    # epoched_EEG_average_trial = np.mean(epoched_EEG_RESAMPLED, axis=0)
    # epoched_EEG_average_trial_chan = np.mean(epoched_EEG_average_trial, axis=0)
    #
    # epoched_EEG_max_trial = np.max(epoched_EEG_RESAMPLED, axis=0)
    # epoched_EEG_max_trial_chan = np.max(epoched_EEG_max_trial, axis=0)
    #
    # epoched_EEG_min_trial = np.min(epoched_EEG_RESAMPLED, axis=0)
    # epoched_EEG_min_trial_chan = np.min(epoched_EEG_min_trial, axis=0)
    #
    # epoched_EEG_timevector = np.linspace(pre_stimulus_time, post_stimulus_time, EEG_num_sample_per_trail_RESAMPLED)
    # return epoched_EEG_timevector, epoched_EEG_average_trial_chan, epoched_EEG_max_trial_chan, epoched_EEG_min_trial_chan
