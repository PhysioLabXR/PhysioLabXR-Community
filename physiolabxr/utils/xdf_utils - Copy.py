import struct
import warnings
import xml.etree.ElementTree as ET
from enum import Enum

import numpy as np
import pyxdf

from physiolabxr.presets.presets_utils import get_stream_data_type, get_stream_nominal_sampling_rate


def create_xml_string(child_dict: dict):
    root = ET.Element("info")
    childs = []
    for key, value in child_dict.items():
        child = ET.SubElement(root, key)
        child.text = value if isinstance(value, str) else str(value)
        childs.append(child)

    # Create the XML string from the root element
    xml_string = ET.tostring(root, encoding="utf-8", method="xml")

    # Print the XML string
    return xml_string.decode("utf-8")


class XdfTag(Enum):
    FileHeader = 1
    StreamHeader = 2
    Samples = 3
    ClockOffset = 4
    Boundary = 5
    StreamFooter = 6



def save_xdf(file_path, buffer, sample_chunk_max_size=50):
    file_header_info = {'name': 'Test', 'user': 'ixi'}
    file_header = create_xml_string(file_header_info)
    stream_headers = {}
    stream_footers = {}
    idx = 0

    # create stream headers and footers
    for stream_label, (data_array, ts_array) in buffer.items():
        stream_data_type = str(data_array.dtype)

        # make the data type consistent with the xdf format
        # any uint type will be converted to int type
        stream_data_type_xdf = stream_data_type
        if stream_data_type_xdf.startswith('uint'):
            stream_data_type_xdf = stream_data_type_xdf.replace('uint', 'int')
        elif stream_data_type_xdf == 'float64':
            stream_data_type_xdf = 'double64'

        nchannels = np.prod(data_array.shape[:-1])
        stream_header_info = {'name': stream_label,
                              'nominal_srate': str(get_stream_nominal_sampling_rate(stream_label)),
                              'channel_count': str(nchannels),
                              'channel_format': stream_data_type_xdf}
        stream_header_xml = create_xml_string(stream_header_info)
        stream_headers[stream_label] = stream_header_xml
        stream_footer_info = {'first_timestamp': str(ts_array[0]),
                              'last_timestamp': str(ts_array[-1]),
                              'sample_count': str(len(ts_array)), 'stream_name': stream_label,
                              'stream_id': idx,
                              'frame_dimension': data_array.shape,
                              'real_data_type': stream_data_type}
        stream_footers[stream_label] = stream_footer_info
        idx += 1

    magic = b'XDF:'
    out_file = open(file_path, "ab")
    # write magic
    out_file.write(magic)
    NumLenByte_decoder = lambda bytes: 1 if (bytes.bit_length() + 7) // 8 <= 1 else 4 if (bytes.bit_length() + 7) // 8 <= 4 else 8
    file_header_len = len(file_header) + 2
    file_header_len_bytes = NumLenByte_decoder(file_header_len)
    file_header = file_header_len_bytes.to_bytes(1, byteorder='little') + file_header_len.to_bytes(
        file_header_len_bytes, byteorder='little') + XdfTag.FileHeader.value.to_bytes(2, byteorder='little') + file_header.encode('utf-8')
    # write file header
    out_file.write(file_header)
    stream_label_list = list(buffer)
    for stream_label, _ in buffer.items():
        stream_header_len = len(stream_headers[stream_label]) + 2 + 4

        stream_header_len_bytes = NumLenByte_decoder(stream_header_len)

        stream_header = stream_header_len_bytes.to_bytes(1, byteorder='little') + \
                        stream_header_len.to_bytes(stream_header_len_bytes, byteorder='little') + \
                        XdfTag.StreamHeader.value.to_bytes(2, byteorder='little') + \
                        stream_footers[stream_label]['stream_id'].to_bytes(4, byteorder='little') + stream_headers[stream_label].encode('utf-8')

        # write stream header
        out_file.write(stream_header)

    # write stream data
    for stream_label, (data_array, ts_array) in buffer.items():

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
            warnings.warn(f'Stream: [{stream_label}] timestamps must be in increasing order.', UserWarning)

        total_samples = len(ts_array)
        stream_id = stream_footers[stream_label]['stream_id']
        nchannels = np.prod(data_array.shape[:-1])  # the number of channels in the stream, the last dimension is the sample/time dimension

        stream_data_type = data_array.dtype
        n_sample_chunks = total_samples // 50 + 1
        samples_stored = 0
        for i in range(n_sample_chunks):
            num_samples = total_samples - samples_stored if i == n_sample_chunks - 1 else sample_chunk_max_size
            num_sample_bytes = NumLenByte_decoder(num_samples)  # compute the total sample chunk length
            samples_byte_len = 9 * num_samples + nchannels * num_samples * stream_data_type.itemsize
            content_tag_byte_len = int(2 + 4 + 1 + num_sample_bytes + samples_byte_len)  # the length of the content plus tag in bytes, 2 for tag, 4 for stream id, 1 for NumSampleBytes
            num_chunk_byte_len = NumLenByte_decoder(content_tag_byte_len)  # the byte length of the total chunk
            stream_content_head = num_chunk_byte_len.to_bytes(1, byteorder='little') + \
                                  content_tag_byte_len.to_bytes(num_chunk_byte_len, byteorder='little') + \
                                  XdfTag.Samples.value.to_bytes(2, byteorder='little') + \
                                  stream_id.to_bytes(4, byteorder='little') + \
                                  num_sample_bytes.to_bytes(1, byteorder='little') + \
                                  num_samples.to_bytes(num_sample_bytes, byteorder='little')
            out_file.write(stream_content_head)
            for j in range(num_samples):
                timestampbytes = int(8).to_bytes(1, byteorder='little')
                timestamp = struct.pack('<d', ts_array[i * sample_chunk_max_size + j])  # timestamp is a double
                values = data_array[..., i * sample_chunk_max_size + j].tobytes()

                out_file.write(timestampbytes + timestamp + values)
            samples_stored += num_samples

    # write stream footers
    for stream_label, _ in buffer.items():
        footer = create_xml_string(stream_footers[stream_label])
        stream_footer_len = len(footer) + 2 + 4
        stream_footer_len_bytes = NumLenByte_decoder(stream_footer_len)
        stream_footer = stream_footer_len_bytes.to_bytes(1, byteorder='little') + \
                        stream_footer_len.to_bytes(stream_footer_len_bytes, byteorder='little') + \
                        XdfTag.StreamFooter.value.to_bytes(2, byteorder='little') + \
                        stream_footers[stream_label]['stream_id'].to_bytes(4, byteorder='little') + \
                        footer.encode('utf-8')
        out_file.write(stream_footer)

    out_file.close()

def load_xdf(filename):
    xdf_data = pyxdf.load_xdf(filename)
    dats_data = {}
    for stream_data in xdf_data[0]:
        stream_footer = stream_data['footer']
        stream_name = stream_footer['info']['stream_name'][0]
        sample_array = stream_data['time_series']
        stream_shape = eval(stream_footer['info']['frame_dimension'][0])
        sample_array = sample_array.T.reshape(stream_shape)
        sample_array = sample_array.astype(stream_footer['info']['real_data_type'][0])
        dats_data[stream_name] = [sample_array, stream_data['time_stamps']]
    return dats_data




