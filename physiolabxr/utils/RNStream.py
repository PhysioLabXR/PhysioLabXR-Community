import os
import warnings
from pathlib import Path

import cv2
import numpy as np

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
        """
        serialize the content of the buffer to the file path pointed by self.fn
        :param buffer: a dictionary, key is a string for stream name, value is a iterable of two ndarray
                        the first of the two ndarray is the data samples, the second of the two ndarray are the timestamps
                        of the data samples. The time axis for the data array must be the last. The timestamp array must
                         have exactly one dimension (the time dimension). The data and timestamps
                        array must have the same length in their time dimensions.
                        The timestamps array must also in a increasing order, otherwise a warning will be raised
        :return: the total number of byptes that has been streamed out
        """
        out_file = open(self.fn, "ab")
        stream_label_bytes, dtype_bytes, dim_bytes, shape_bytes, data_bytes, ts_bytes = \
            b'', b'', b'', b'', b'', b''
        total_bytes = 0
        for stream_label, data_ts_array in buffer.items():
            data_array, ts_array = data_ts_array[0], data_ts_array[1]

            # cast the arrays in
            if type(data_array) != np.ndarray:
                data_array = np.array(data_array)
            if type(ts_array) != np.ndarray:
                ts_array = np.array(ts_array)

            try:
                assert len(ts_array.shape) == 1
            except AssertionError:
                raise Exception('timestamps must have exactly one dimension.')

            try:
                assert all(i < j for i, j in zip(ts_array, ts_array[1:]))
            except AssertionError:
                warnings.warn(f'RNStream: [{stream_label}] timestamps must be in increasing order.', UserWarning)
            stream_label_bytes = \
                bytes(stream_label[:max_label_len] + "".join(
                    " " for x in range(max_label_len - len(stream_label))), encoding)
            try:
                dtype_str = str(data_array.dtype)
                assert len(dtype_str) < max_dtype_len
            except AssertionError:
                raise Exception('dtype encoding exceeds max dtype length: {0}, please contact support'.format(max_dtype_len))
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
            total_bytes += len(magic) + len(stream_label_bytes) + len(dtype_bytes) + len(dim_bytes) + len(shape_bytes) + len(data_bytes) + len(ts_bytes)
        out_file.close()
        return total_bytes

    def stream_in(self, ignore_stream=None, only_stream=None, jitter_removal=True, reshape_stream_dict=None):
        """
        different from LSL XDF importer, this jitter removal assumes no interruption in the data
        :param ignore_stream:
        :param only_stream:
        :param jitter_removal:
        :param reshape_stream_dict:
        :return:
        """
        total_bytes = float(os.path.getsize(self.fn))  # use floats to avoid scalar type overflow
        buffer = {}
        read_bytes_count = 0.
        with open(self.fn, "rb") as file:
            while True:
                if total_bytes:
                    print('Streaming in progress {0}%'.format(str(round(100 * read_bytes_count/total_bytes, 2))), sep=' ', end='\r', flush=True)
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
                if len(ts_array) < 2:
                    print("Ignore jitter remove for stream {0}, because it has fewer than two samples".format(stream_name))
                    continue
                if np.std(ts_array) > 0.1:
                    warnings.warn(f"Stream {stream_name} may have a irregular sampling rate with its timestamp's std {np.std(ts_array)}. Jitter removal should not be applied to irregularly sampled streams.", RuntimeWarning)
                print('Removing jitter for streams {0}/{1}'.format(i, len(buffer)), sep=' ',
                      end='\r', flush=True)
                coefs = np.polyfit(list(range(len(ts_array))), ts_array, 1)
                smoothed_ts_array = np.array([i * coefs[0] + coefs[1] for i in range(len(ts_array))])
                buffer[stream_name][1] = smoothed_ts_array

        # reshape img, time series, time frames data
        if reshape_stream_dict is not None:
            for reshape_stream_name in reshape_stream_dict:
                if reshape_stream_name in buffer:  # reshape the stream[0] to [(a,b,c), (d, e), x, y] etc

                    shapes = reshape_stream_dict[reshape_stream_name]
                    # check if the number of channel matches the number of reshape channels
                    total_reshape_channel_num = 0
                    for shape_item in shapes: total_reshape_channel_num += np.prod(shape_item)
                    if total_reshape_channel_num == buffer[reshape_stream_name][0].shape[0]:
                        # number of channels matches, start reshaping
                        reshape_data_buffer = {}
                        offset = 0
                        for index, shape_item in enumerate(shapes):
                            reshape_channel_num = np.prod(shape_item)
                            data_slice = buffer[reshape_stream_name][0][offset:offset + reshape_channel_num,
                                         :]  # get the slice
                            # reshape all column to shape_item
                            print((shape_item + (-1,)))
                            data_slice = data_slice.reshape((shape_item + (-1,)))
                            reshape_data_buffer[index] = data_slice
                            offset += reshape_channel_num

                        #     replace buffer[stream_name][0] with new reshaped buffer
                        buffer[reshape_stream_name][0] = reshape_data_buffer

                    else:
                        raise Exception(
                            'Error: The given total number of reshape channel does not match the total number of saved '
                            'channel for stream: ({0})'.format(reshape_stream_name))

                else:
                    raise Exception(
                        'Error: The give target reshape stream ({0}) does not exist in the data buffer, please use ('
                        'get_stream_names) function to check the stream names'.format(reshape_stream_name))

        print("Stream-in completed: {0}".format(self.fn))
        return buffer

    def stream_in_stepwise(self, file, buffer, read_bytes_count, ignore_stream=None, only_stream=None, jitter_removal=True, reshape_stream_dict=None):
        total_bytes = float(os.path.getsize(self.fn))  # use floats to avoid scalar type overflow
        buffer = {} if buffer is None else buffer
        read_bytes_count = 0. if read_bytes_count is None else read_bytes_count
        file = open(self.fn, "rb") if file is None else file
        finished = False

        if total_bytes:
            print('Streaming in progress {0}%'.format(str(round(100 * read_bytes_count/total_bytes, 2))), sep=' ', end='\r', flush=True)
        # read magic
        read_bytes = file.read(len(magic))
        read_bytes_count += len(read_bytes)
        if len(read_bytes) == 0:
            finished = True
        if not finished:
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
        if finished:
            if jitter_removal:
                i = 1
                for stream_name, (d_array, ts_array) in buffer.items():
                    if len(ts_array) < 2:
                        print("Ignore jitter remove for stream {0}, because it has fewer than two samples".format(stream_name))
                        continue
                    if np.std(ts_array) > 0.1:
                        warnings.warn("Stream {0} may have a irregular sampling rate with std {0}. Jitter removal should not be applied to irregularly sampled streams.".format(stream_name, np.std(ts_array)), RuntimeWarning)
                    print('Removing jitter for streams {0}/{1}'.format(i, len(buffer)), sep=' ',
                          end='\r', flush=True)
                    coefs = np.polyfit(list(range(len(ts_array))), ts_array, 1)
                    smoothed_ts_array = np.array([i * coefs[0] + coefs[1] for i in range(len(ts_array))])
                    buffer[stream_name][1] = smoothed_ts_array

            # reshape img, time series, time frames data
            if reshape_stream_dict is not None:
                for reshape_stream_name in reshape_stream_dict:
                    if reshape_stream_name in buffer:  # reshape the stream[0] to [(a,b,c), (d, e), x, y] etc

                        shapes = reshape_stream_dict[reshape_stream_name]
                        # check if the number of channel matches the number of reshape channels
                        total_reshape_channel_num = 0
                        for shape_item in shapes: total_reshape_channel_num += np.prod(shape_item)
                        if total_reshape_channel_num == buffer[reshape_stream_name][0].shape[0]:
                            # number of channels matches, start reshaping
                            reshape_data_buffer = {}
                            offset = 0
                            for index, shape_item in enumerate(shapes):
                                reshape_channel_num = np.prod(shape_item)
                                data_slice = buffer[reshape_stream_name][0][offset:offset + reshape_channel_num,
                                             :]  # get the slice
                                # reshape all column to shape_item
                                print((shape_item + (-1,)))
                                data_slice = data_slice.reshape((shape_item + (-1,)))
                                reshape_data_buffer[index] = data_slice
                                offset += reshape_channel_num

                            #     replace buffer[stream_name][0] with new reshaped buffer
                            buffer[reshape_stream_name][0] = reshape_data_buffer

                        else:
                            raise Exception(
                                'Error: The given total number of reshape channel does not match the total number of saved '
                                'channel for stream: ({0})'.format(reshape_stream_name))

                    else:
                        raise Exception(
                            'Error: The give target reshape stream ({0}) does not exist in the data buffer, please use ('
                            'get_stream_names) function to check the stream names'.format(reshape_stream_name))

        return file, buffer, read_bytes_count, total_bytes, finished

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
        output_path = os.path.join(data_root, '{0}_{1}.avi'.format(data_fn.split('.')[0], video_stream_name)) if output_path == '' else output_path

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'),frate, frame_size)

        for i in range(frame_count):
            print('Creating video progress {}%'.format(str(round(100 * i / frame_count, 2))), sep=' ', end='\r',
                  flush=True)
            img = video_frame_stream[:, :, :, i]
            # img = np.reshape(img, newshape=list(frame_size) + [-1,])
            out.write(img)

        out.release()
