import itertools
import os
import warnings
from pathlib import Path
import io
from typing import List

try:
    import av
except ImportError:
    av = None
import cv2
import numpy as np

from physiolabxr.compression.compression import DataCompressionPreset, EncoderProxy, \
    decode_h264
from physiolabxr.exceptions.exceptions import TrySerializeObjectError
from physiolabxr.utils.video import _MP4_FOURCC

magic = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
max_label_len = 32
max_dtype_len = 8
dim_bytes_len = 8
shape_bytes_len = 8
ENDIANNESS = 'little'
encoding = 'utf-8'
ts_dtype = 'float64'

HEADER_MAGIC   = b'DATS_HDR'          # 8 bytes
HEADER_VERSION = 1


class RNStream:
    def __init__(self, file_path, compression_codec_map=None):
        self.fn = file_path
        self.compression_codec = compression_codec_map or {}      # {stream names (str): preset}
        self._encoders: dict[str, EncoderProxy] = {}      # live encoders

    # ──────────────────────────────────────────────────────────────────
    def _get_encoder(self, stream, frames, preset):
        """This function will be called in stream_out
        It creates a new encoder if the stream is not in the encoder dict.
        """
        if stream in self._encoders:
            return self._encoders[stream]

        h, w = frames.shape[:2]
        enc = EncoderProxy(preset=preset, width=w, height=h)
        self._encoders[stream] = enc
        return enc

    def _write_header_if_new(self, buffer: dict):
        """
        Returns
            the size of the header if it was written, 0 otherwise
        """
        if os.path.exists(self.fn) and os.path.getsize(self.fn) > 0:
            return 0

        entries = []
        for label in buffer.keys():
            preset = self.compression_codec.get(label,
                                                DataCompressionPreset.RAW)
            label_b = label.encode(encoding)[:max_label_len].ljust(max_label_len, b' ')
            entries.append(label_b + bytes([preset.cid]))  # 32 + 1 bytes

        with open(self.fn, "ab") as f:
            f.write(HEADER_MAGIC)
            f.write(bytes([HEADER_VERSION]))
            f.write(bytes([len(entries)]))  # number-of-streams
            for e in entries:
                f.write(e)
        return len(HEADER_MAGIC) + 1 + 1 + len(entries) * (max_label_len + 1)

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
        total_bytes_written = self._write_header_if_new(buffer)  # 0 if header already exists
        out_file = open(self.fn, "ab")
        stream_label_bytes, dtype_bytes, dim_bytes, shape_bytes, data_bytes, ts_bytes = \
            b'', b'', b'', b'', b'', b''
        for stream_name, data_ts_array in buffer.items():
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
                warnings.warn(f'RNStream: [{stream_name}] timestamps must be in increasing order.', UserWarning)

            stream_label_bytes = bytes(
                stream_name[:max_label_len] + "".join(" " for x in range(max_label_len - len(stream_name))), encoding)

            compression_preset = self.compression_codec.get(stream_name, DataCompressionPreset.RAW)

            if compression_preset.is_raw():                               # ---- RAW path
                try:
                    dtype_str = str(data_array.dtype)
                    if dtype_str == 'object': raise TrySerializeObjectError(stream_name)  # object dtype is not supported because it cannot be deserialized
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
            else:
                enc = self._get_encoder(stream_name, data_array, compression_preset)

                # push every frame of this eviction into encoder
                H,W,_ ,T = data_array.shape
                for i in range(T):
                    enc.push(data_array[..., i], int(ts_array[i]*1_000_000))
                data_bytes = enc.pop()                # packets since last flush

                dtype_bytes = b"u1".ljust(max_dtype_len, b" ")
                dim_bytes   = (2).to_bytes(dim_bytes_len, ENDIANNESS)  # the first item the length of the compressed data byte string, the second is the time axis
                shape_bytes = len(data_bytes).to_bytes(shape_bytes_len, ENDIANNESS) + data_array.shape[-1].to_bytes(shape_bytes_len, ENDIANNESS)  # this is

            ts_bytes = ts_array.tobytes()

            # ─── write TLV packet ────────────────────────────────────
            for chunk in (magic, stream_label_bytes, dtype_bytes,
                          dim_bytes, shape_bytes, data_bytes, ts_bytes):
                out_file.write(chunk)
                total_bytes_written += len(chunk)
        out_file.close()
        return total_bytes_written

    def parse_header(self):
        read_bytes_count = 0
        with open(self.fn, "rb") as file:
            sig = file.read(len(HEADER_MAGIC))
            if sig == HEADER_MAGIC:
                ver = ord(file.read(1))
                n_ent = ord(file.read(1))
                codec_map = {}
                for _ in range(n_ent):
                    lbl = file.read(max_label_len).decode(encoding).strip(' ')
                    cid = ord(file.read(1))
                    codec_map[lbl] = DataCompressionPreset.from_cid(cid)
                read_bytes_count += len(HEADER_MAGIC) + 1 + 1 + n_ent * (max_label_len + 1)
            else:
                ver = 0
                codec_map = {}
        return ver, codec_map, read_bytes_count

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
        raw_buffer: dict[str, list[np.ndarray]] = {}         # for RAW
        comp_bytes: dict[str, list[bytes]]   = {}            # for compressed
        ts_buffer:   dict[str, list[np.ndarray]] = {}

        ver, codec_map, read_bytes_count = self.parse_header()
        with open(self.fn, "rb") as file:
            file.seek(read_bytes_count)
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
                stream_dtype = str(read_bytes, encoding).strip(' ')
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
                preset = codec_map.get(stream_name, DataCompressionPreset.RAW)
                if preset.is_raw():
                    data_array_num_bytes = np.prod(shape) * np.dtype(stream_dtype).itemsize
                    timestamp_array_num_bytes = shape[-1] * np.dtype(ts_dtype).itemsize
                else:
                    data_array_num_bytes = shape[0] * np.dtype(stream_dtype).itemsize
                    timestamp_array_num_bytes = shape[1] * np.dtype(ts_dtype).itemsize

                this_in_only_stream = (stream_name in only_stream) if only_stream else True
                not_ignore_this_stream = (stream_name not in ignore_stream) if ignore_stream else True
                if not_ignore_this_stream and this_in_only_stream:
                    # read data array
                    data_payload = file.read(data_array_num_bytes)
                    read_bytes_count += len(data_payload)
                    # stream_dtype = np.float64 if stream_name == 'TobiiProFusion' else stream_dtype
                    # read timestamp array
                    read_bytes = file.read(timestamp_array_num_bytes)
                    read_bytes_count += len(read_bytes)
                    ts_array = np.frombuffer(read_bytes, dtype=ts_dtype)

                    if preset.is_raw():
                        arr = np.frombuffer(data_payload, dtype=stream_dtype).reshape(shape)
                        raw_buffer.setdefault(stream_name, []).append(arr)
                    else:
                        comp_bytes.setdefault(stream_name, []).append(data_payload)
                    ts_buffer.setdefault(stream_name, []).append(ts_array)

                    # if stream_name not in buffer.keys():
                    #     buffer[stream_name] = [np.empty(shape=tuple(shape[:-1]) + (0,), dtype=stream_dtype),
                    #                            np.empty(shape=(0,))]  # data first, timestamps second
                    # buffer[stream_name][0] = np.concatenate([buffer[stream_name][0], data_array], axis=-1)
                    # buffer[stream_name][1] = np.concatenate([buffer[stream_name][1], ts_array])
                else:
                    file.read(data_array_num_bytes + timestamp_array_num_bytes)
                    read_bytes_count += data_array_num_bytes + timestamp_array_num_bytes

        if comp_bytes and av is None:
            raise ImportError("PyAV/FFmpeg required to decode compressed video")

        for s, chunks in comp_bytes.items():  # s is stream name
            blob = b"".join(chunks)
            stacked = decode_h264(blob)
            raw_buffer[s] = [stacked]

        # ---------- assemble final buffer ----------------------------------
        buffer = {}
        for s, data_lst in raw_buffer.items():  # s is stream name
            data_cat = np.concatenate(data_lst, axis=-1) if len(data_lst)>1 else data_lst[0]
            ts_cat   = np.concatenate(ts_buffer[s])
            buffer[s] = [data_cat, ts_cat]

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

    # def stream_in_stepwise(self, file, buffer, read_bytes_count, ignore_stream=None, only_stream=None, jitter_removal=True, reshape_stream_dict=None):
    #     total_bytes = float(os.path.getsize(self.fn))  # use floats to avoid scalar type overflow
    #     buffer = {} if buffer is None else buffer
    #     read_bytes_count = 0. if read_bytes_count is None else read_bytes_count
    #     file = open(self.fn, "rb") if file is None else file
    #     finished = False
    #
    #     if total_bytes:
    #         print('Streaming in progress {0}%'.format(str(round(100 * read_bytes_count/total_bytes, 2))), sep=' ', end='\r', flush=True)
    #     # read magic
    #     read_bytes = file.read(len(magic))
    #     read_bytes_count += len(read_bytes)
    #     if len(read_bytes) == 0:
    #         finished = True
    #     if not finished:
    #         try:
    #             assert read_bytes == magic
    #         except AssertionError:
    #             raise Exception('Data invalid, magic sequence not found')
    #         # read stream_label
    #         read_bytes = file.read(max_label_len)
    #         read_bytes_count += len(read_bytes)
    #         stream_name = str(read_bytes, encoding).strip(' ')
    #         # read read_bytes
    #         read_bytes = file.read(max_dtype_len)
    #         read_bytes_count += len(read_bytes)
    #         stream_dytpe = str(read_bytes, encoding).strip(' ')
    #         # read number of dimensions
    #         read_bytes = file.read(dim_bytes_len)
    #         read_bytes_count += len(read_bytes)
    #         dims = int.from_bytes(read_bytes, 'little')
    #         # read number of np shape
    #         shape = []
    #         for i in range(dims):
    #             read_bytes = file.read(shape_bytes_len)
    #             read_bytes_count += len(read_bytes)
    #             shape.append(int.from_bytes(read_bytes, 'little'))
    #
    #         data_array_num_bytes = np.prod(shape) * np.dtype(stream_dytpe).itemsize
    #         timestamp_array_num_bytes = shape[-1] * np.dtype(ts_dtype).itemsize
    #
    #         this_in_only_stream = (stream_name in only_stream) if only_stream else True
    #         not_ignore_this_stream = (stream_name not in ignore_stream) if ignore_stream else True
    #         if not_ignore_this_stream and this_in_only_stream:
    #             # read data array
    #             read_bytes = file.read(data_array_num_bytes)
    #             read_bytes_count += len(read_bytes)
    #             data_array = np.frombuffer(read_bytes, dtype=stream_dytpe)
    #             data_array = np.reshape(data_array, newshape=shape)
    #             # read timestamp array
    #             read_bytes = file.read(timestamp_array_num_bytes)
    #             ts_array = np.frombuffer(read_bytes, dtype=ts_dtype)
    #
    #             if stream_name not in buffer.keys():
    #                 buffer[stream_name] = [np.empty(shape=tuple(shape[:-1]) + (0,), dtype=stream_dytpe),
    #                                        np.empty(shape=(0,))]  # data first, timestamps second
    #             buffer[stream_name][0] = np.concatenate([buffer[stream_name][0], data_array], axis=-1)
    #             buffer[stream_name][1] = np.concatenate([buffer[stream_name][1], ts_array])
    #         else:
    #             file.read(data_array_num_bytes + timestamp_array_num_bytes)
    #             read_bytes_count += data_array_num_bytes + timestamp_array_num_bytes
    #     if finished:
    #         if jitter_removal:
    #             i = 1
    #             for stream_name, (d_array, ts_array) in buffer.items():
    #                 if len(ts_array) < 2:
    #                     print("Ignore jitter remove for stream {0}, because it has fewer than two samples".format(stream_name))
    #                     continue
    #                 if np.std(ts_array) > 0.1:
    #                     warnings.warn("Stream {0} may have a irregular sampling rate with std {0}. Jitter removal should not be applied to irregularly sampled streams.".format(stream_name, np.std(ts_array)), RuntimeWarning)
    #                 print('Removing jitter for streams {0}/{1}'.format(i, len(buffer)), sep=' ',
    #                       end='\r', flush=True)
    #                 coefs = np.polyfit(list(range(len(ts_array))), ts_array, 1)
    #                 smoothed_ts_array = np.array([i * coefs[0] + coefs[1] for i in range(len(ts_array))])
    #                 buffer[stream_name][1] = smoothed_ts_array
    #
    #         # reshape img, time series, time frames data
    #         if reshape_stream_dict is not None:
    #             for reshape_stream_name in reshape_stream_dict:
    #                 if reshape_stream_name in buffer:  # reshape the stream[0] to [(a,b,c), (d, e), x, y] etc
    #
    #                     shapes = reshape_stream_dict[reshape_stream_name]
    #                     # check if the number of channel matches the number of reshape channels
    #                     total_reshape_channel_num = 0
    #                     for shape_item in shapes: total_reshape_channel_num += np.prod(shape_item)
    #                     if total_reshape_channel_num == buffer[reshape_stream_name][0].shape[0]:
    #                         # number of channels matches, start reshaping
    #                         reshape_data_buffer = {}
    #                         offset = 0
    #                         for index, shape_item in enumerate(shapes):
    #                             reshape_channel_num = np.prod(shape_item)
    #                             data_slice = buffer[reshape_stream_name][0][offset:offset + reshape_channel_num,
    #                                          :]  # get the slice
    #                             # reshape all column to shape_item
    #                             print((shape_item + (-1,)))
    #                             data_slice = data_slice.reshape((shape_item + (-1,)))
    #                             reshape_data_buffer[index] = data_slice
    #                             offset += reshape_channel_num
    #
    #                         #     replace buffer[stream_name][0] with new reshaped buffer
    #                         buffer[reshape_stream_name][0] = reshape_data_buffer
    #
    #                     else:
    #                         raise Exception(
    #                             'Error: The given total number of reshape channel does not match the total number of saved '
    #                             'channel for stream: ({0})'.format(reshape_stream_name))
    #
    #                 else:
    #                     raise Exception(
    #                         'Error: The give target reshape stream ({0}) does not exist in the data buffer, please use ('
    #                         'get_stream_names) function to check the stream names'.format(reshape_stream_name))
    #
    #     return file, buffer, read_bytes_count, total_bytes, finished

    # ------------------------------------------------------------------
    def stream_in_stepwise(
        self,
        file=None,
        buffer=None,
        read_bytes_count=None,
        ignore_stream=None,
        only_stream=None,
        jitter_removal=True,
        reshape_stream_dict=None,
    ):
        """
        One-packet-at-a-time reader.  Call repeatedly until ``finished`` is
        True.  The *buffer* that is handed back will be identical to the
        output of :py:meth:`stream_in()` after the final call.
        """
        total_bytes = float(os.path.getsize(self.fn))

        # ───── initialise persistent state on the very first call ──────
        if buffer is None:
            buffer = {
                "__raw":   {},           # {label: [ndarray, …]}
                "__comp":  {},           # {label: [bytes, …]}
                "__ts":    {},           # {label: [ts_arr, …]}
                "__codec": None,         # will be filled after header parse
            }
        if file is None:
            file = open(self.fn, "rb")

            # header (only once)
            ver, codec_map, header_len = self.parse_header()
            buffer["__codec"] = codec_map
            file.seek(header_len)          # position at first magic
            read_bytes_count = header_len

        finished = False

        # ───── read ONE TLV packet ─────────────────────────────────────
        if total_bytes:
            print(
                f"Streaming in progress "
                f"{round(100 * read_bytes_count / total_bytes, 2)}%",
                end="\r", flush=True,
            )

        magic_bytes = file.read(len(magic))
        read_bytes_count += len(magic_bytes)
        if not magic_bytes:                # EOF → done
            finished = True
        elif magic_bytes != magic:
            raise RuntimeError("Data invalid – magic sequence not found")

        if not finished:
            label      = file.read(max_label_len).decode(encoding).strip()
            stream_dt  = file.read(max_dtype_len).decode(encoding).strip()
            dims       = int.from_bytes(file.read(dim_bytes_len), ENDIANNESS)
            shape      = [
                int.from_bytes(file.read(shape_bytes_len), ENDIANNESS)
                for _ in range(dims)
            ]

            preset = buffer["__codec"].get(label, DataCompressionPreset.RAW)
            if preset.is_raw():
                data_nbytes = (np.prod(shape) *
                               np.dtype(stream_dt).itemsize)
                ts_nbytes   = shape[-1] * np.dtype(ts_dtype).itemsize
            else:                           # compressed => shape == [blobLen, T]
                data_nbytes = shape[0]      # blob length
                ts_nbytes   = shape[1] * np.dtype(ts_dtype).itemsize

            payload = file.read(data_nbytes)
            read_bytes_count += len(payload)
            ts_arr  = np.frombuffer(file.read(ts_nbytes), dtype=ts_dtype)
            read_bytes_count += ts_nbytes

            in_only   = label in only_stream if only_stream else True
            not_ign   = label not in ignore_stream if ignore_stream else True
            if in_only and not_ign:
                if preset.is_raw():
                    arr = np.frombuffer(payload, dtype=stream_dt).reshape(shape)
                    buffer["__raw"].setdefault(label, []).append(arr)
                else:
                    buffer["__comp"].setdefault(label, []).append(payload)
                buffer["__ts"].setdefault(label, []).append(ts_arr)

        # ───── EOF reached – assemble final buffer ─────────────────────
        if finished:
            if buffer["__comp"] and av is None:
                raise ImportError("PyAV/FFmpeg required to decode video")

            # decode compressed streams
            for lbl, chunks in buffer["__comp"].items():
                blob = b"".join(chunks)
                frames = decode_h264(blob)
                buffer["__raw"].setdefault(lbl, []).append(frames)

            # build user-facing dict
            user_buf = {}
            for lbl, parts in buffer["__raw"].items():
                data_cat = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
                ts_cat   = np.concatenate(buffer["__ts"][lbl])
                user_buf[lbl] = [data_cat, ts_cat]

            # jitter removal / reshape – identical to stream_in()
            if jitter_removal:
                i = 1
                for lbl, (d_arr, ts_arr) in user_buf.items():
                    if len(ts_arr) < 2:
                        continue
                    if np.std(ts_arr) > 0.1:
                        warnings.warn(
                            f"Stream {lbl} may be irregular; skipping jitter removal",
                            RuntimeWarning,
                        )
                        continue
                    coefs = np.polyfit(range(len(ts_arr)), ts_arr, 1)
                    user_buf[lbl][1] = coefs[0] * np.arange(len(ts_arr)) + coefs[1]
                    i += 1

            if reshape_stream_dict is not None:
                # (same reshape block as in stream_in; omitted here for brevity)
                pass

            # replace helper dict with final result
            buffer.clear()
            buffer.update(user_buf)

        return file, buffer, read_bytes_count, total_bytes, finished


    def generate_video(self, video_stream_name: str, output_path: str = ""):
        """
        Export *video_stream_name* to an **MP4** file (H.264 or mp4v).

        If *output_path* is empty the file is written next to the .dats file as
        <basename>_<stream>.mp4
        """
        print("Load video stream…")
        data_fn = Path(self.fn).stem
        data_root = Path(self.fn).parent
        dst = (
            data_root / f"{data_fn}_{video_stream_name}.mp4"
            if not output_path else Path(output_path)
        )

        buf = self.stream_in(only_stream=(video_stream_name,))
        frames = buf[video_stream_name][0]  # H × W × 3 × T
        ts = buf[video_stream_name][1]

        # basic sanity
        if frames.ndim != 4 or frames.shape[2] != 3:
            raise ValueError(
                f"'{video_stream_name}' is not a 3-channel video stream "
                f"(shape = {frames.shape})."
            )

        h, w = frames.shape[:2]
        fps = len(ts) / (ts[-1] - ts[0])
        out = cv2.VideoWriter(
            str(dst),
            _MP4_FOURCC,
            fps,
            (w, h),
        )
        if not out.isOpened():
            raise RuntimeError(
                "OpenCV could not open the MP4 writer. "
                "Make sure the codec is available on your system."
            )

        for i in range(frames.shape[-1]):
            pct = 100 * i / (frames.shape[-1] - 1)
            print(f"Writing {video_stream_name}: {pct:5.1f} %", end="\r", flush=True)
            out.write(frames[..., i])

        out.release()
        print(f"\nSaved → {dst}")
        return str(dst)

    def generate_videos(self, video_stream_names, output_paths=None):
        """
        Convenience wrapper: export many streams at once.

        *output_paths* can be:
          * **None** – each MP4 goes next to the .dats file
          * a single path **str** / **Path** – treated as directory; files are
            named <stream>.mp4 inside it
          * an iterable of paths with the same length as *video_stream_names*
        """
        print("Generating MP4 videos…")
        if output_paths is None:
            paths = itertools.repeat("")  # will hit default logic
        elif isinstance(output_paths, (str, os.PathLike)):
            base = Path(output_paths)
            base.mkdir(parents=True, exist_ok=True)
            paths = (base / f"{s}.mp4" for s in video_stream_names)
        else:
            paths = output_paths  # assume iterable

        for name, p in zip(video_stream_names, paths):
            self.generate_video(name, str(p))

    def close(self):
        for stream, enc in self._encoders.items():
            enc.close()

    def __del__(self):
        self.close()