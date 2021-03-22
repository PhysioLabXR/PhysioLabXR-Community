import numpy as np


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
endianness = 'little'
encoding = 'utf-8'

class RNStream:
    def __init__(self, file_path):
        self.fn = file_path
        self.out_file = open(file_path, "wb")

    def stream_out(self, buffer):
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
                dim_bytes = len(data_array.shape).to_bytes(8, 'little')
                shape_bytes = b''.join([s.to_bytes(8, 'little') for s in data_array.shape])  # the last axis is time
            except OverflowError:
                raise Exception('RN requires its stream to have number of dimensions less than 2^40, '
                      'and the size of any dimension to be less than the same number ')
            data_bytes = data_array.tobytes()
            ts_bytes = ts_array.tobytes()
            self.out_file.write(magic)
            self.out_file.write(stream_label_bytes)
            self.out_file.write(dtype_bytes)
            self.out_file.write(dim_bytes)
            self.out_file.write(shape_bytes)
            self.out_file.write(data_bytes)
            self.out_file.write(ts_bytes)

        return len(magic + stream_label_bytes + dtype_bytes + dim_bytes + shape_bytes + data_bytes + ts_bytes)

    def stream_in(self):
        with open(self.fn, "rb") as file:
            in_bytes = file.read()

        buffer = {}
        while in_bytes:
            # raed magic
            in_bytes, read_bytes = in_bytes[len(magic):], in_bytes[:len(magic)]
            try:
                assert read_bytes == magic
            except AssertionError:
                raise Exception('Data invalid, magic sequence not found')
            in_bytes, read_bytes = in_bytes[max_label_len:], in_bytes[:max_label_len]
            stream_label = str(read_bytes, encoding).strip(' ')
            pass