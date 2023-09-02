import struct
import warnings
import xml.etree.ElementTree as ET
from enum import Enum

import numpy as np
import pyxdf

from physiolabxr.presets.presets_utils import get_stream_data_type


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


class XDF:
    def __init__(self, file_header_xml, stream_headers_dict, stream_footers_dict):
        self.file_header = file_header_xml
        self.stream_headers = stream_headers_dict
        self.stream_footers = stream_footers_dict

    def store_xdf(self, file_path, buffer, sample_chunk_max_size=50):
        magic = b'XDF:'
        out_file = open(file_path, "ab")
        # write magic
        out_file.write(magic)
        NumLenByte_decoder = lambda bytes: 1 if (bytes.bit_length() + 7) // 8 <= 1 else 4 if (bytes.bit_length() + 7) // 8 <= 4 else 8
        file_header_len = len(self.file_header) + 2
        file_header_len_bytes = NumLenByte_decoder(file_header_len)
        file_header = file_header_len_bytes.to_bytes(1, byteorder='little') + file_header_len.to_bytes(
            file_header_len_bytes, byteorder='little') + XdfTag.FileHeader.value.to_bytes(2, byteorder='little') + self.file_header.encode('utf-8')
        # write file header
        out_file.write(file_header)
        stream_label_list = list(buffer)
        for stream_label, _ in buffer.items():
            stream_header_len = len(self.stream_headers[stream_label]) + 2 + 4

            stream_header_len_bytes = NumLenByte_decoder(stream_header_len)

            stream_header = stream_header_len_bytes.to_bytes(1, byteorder='little') + \
                            stream_header_len.to_bytes(stream_header_len_bytes, byteorder='little') + \
                            XdfTag.StreamHeader.value.to_bytes(2, byteorder='little') + \
                            self.stream_footers[stream_label]['stream_id'].to_bytes(4, byteorder='little') + self.stream_headers[stream_label].encode('utf-8')

            # write stream header
            out_file.write(stream_header)

        # write stream data
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
                warnings.warn(f'Stream: [{stream_label}] timestamps must be in increasing order.', UserWarning)

            if stream_label == 'monitor 0':
                total_samples = len(ts_array)
                # write one sample per chunk
                stream_id = self.stream_footers[stream_label]['stream_id']
                nchannels = data_array.shape[0] * data_array.shape[1] * data_array.shape[2]
                samples_per_chunk = 1
                for i in range(total_samples):
                    num_sample_bytes = NumLenByte_decoder(samples_per_chunk)
                    samples_byte_len = 9 * samples_per_chunk + nchannels * samples_per_chunk * 1
                    content_tag_byte_len = 2 + 4 + 1 + num_sample_bytes + samples_byte_len  # the length of the content plus tag in bytes, 2 for tag, 4 for stream id, 1 for NumSampleBytes
                    num_chunk_byte_len = NumLenByte_decoder(content_tag_byte_len)  # the byte length of the total chunk
                    stream_content_head = num_chunk_byte_len.to_bytes(1, byteorder='little') + \
                                          content_tag_byte_len.to_bytes(num_chunk_byte_len, byteorder='little') + \
                                          XdfTag.Samples.value.to_bytes(2, byteorder='little') + \
                                          stream_id.to_bytes(4, byteorder='little') + \
                                          num_sample_bytes.to_bytes(1, byteorder='little') + \
                                          samples_per_chunk.to_bytes(num_sample_bytes, byteorder='little')
                    out_file.write(stream_content_head)
                    timestampbytes = int(8).to_bytes(1, byteorder='little')
                    timestamp = struct.pack('<d', ts_array[i])
                    values = data_array[:, :, :, i].tobytes()
                    out_file.write(timestampbytes + timestamp + values)
            else:
                stream_data_type = np.dtype(get_stream_data_type(stream_label).value)
                total_samples = len(ts_array)
                n_sample_chunks = total_samples // 50 + 1
                stream_id = self.stream_footers[stream_label]['stream_id']
                nchannels = data_array.shape[0]
                samples_stored = 0
                for i in range(n_sample_chunks):
                    if i != n_sample_chunks - 1:
                        num_sample_bytes = NumLenByte_decoder(sample_chunk_max_size)
                        # compute the total sample chunk length
                        samples_byte_len = 9 * sample_chunk_max_size + nchannels * sample_chunk_max_size * stream_data_type.itemsize
                        content_tag_byte_len = 2 + 4 + 1 + num_sample_bytes + samples_byte_len # the length of the content plus tag in bytes, 2 for tag, 4 for stream id, 1 for NumSampleBytes
                        num_chunk_byte_len = NumLenByte_decoder(content_tag_byte_len) # the byte length of the total chunk
                        stream_content_head = num_chunk_byte_len.to_bytes(1, byteorder='little') + \
                                              content_tag_byte_len.to_bytes(num_chunk_byte_len, byteorder='little') + \
                                              XdfTag.Samples.value.to_bytes(2, byteorder='little') + \
                                              stream_id.to_bytes(4, byteorder='little') + \
                                              num_sample_bytes.to_bytes(1, byteorder='little') + \
                                              sample_chunk_max_size.to_bytes(num_sample_bytes, byteorder='little')
                        out_file.write(stream_content_head)
                        for j in range(sample_chunk_max_size):
                            timestampbytes = int(8).to_bytes(1, byteorder='little')
                            timestamp = struct.pack('<d', ts_array[i * sample_chunk_max_size + j])
                            values = b''
                            if stream_data_type == 'float64':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<d', k)
                            elif stream_data_type == 'float32':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<f', k)
                            elif stream_data_type == 'float16':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<e', k)
                            elif stream_data_type == 'int64':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<q', k)
                            elif stream_data_type == 'int32':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<i', k)
                            elif stream_data_type == 'int16':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<h', k)
                            elif stream_data_type == 'int8':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<b', k)
                            elif stream_data_type == 'uint64':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<Q', k)
                            elif stream_data_type == 'uint32':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<I', k)
                            elif stream_data_type == 'uint16':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<H', k)
                            elif stream_data_type == 'uint8':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<B', k)
                            else:
                                raise Exception('unknown stream type')
                            out_file.write(timestampbytes + timestamp + values)
                        samples_stored += sample_chunk_max_size
                    else:
                        num_samples = total_samples - samples_stored
                        num_sample_bytes = NumLenByte_decoder(num_samples)
                        # compute the total sample chunk length
                        samples_byte_len = 9 * num_samples + nchannels * num_samples * stream_data_type.itemsize
                        content_tag_byte_len = 2 + 4 + 1 + num_sample_bytes + samples_byte_len  # the length of the content plus tag in bytes, 2 for tag, 4 for stream id, 1 for NumSampleBytes
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
                            timestamp = struct.pack('<d', ts_array[i * sample_chunk_max_size + j])
                            values = b''
                            if stream_data_type == 'float64':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<d', k)
                            elif stream_data_type == 'float32':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<f', k)
                            elif stream_data_type == 'float16':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<e', k)
                            elif stream_data_type == 'int64':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<q', k)
                            elif stream_data_type == 'int32':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<i', k)
                            elif stream_data_type == 'int16':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<h', k)
                            elif stream_data_type == 'int8':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<b', k)
                            elif stream_data_type == 'uint64':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<Q', k)
                            elif stream_data_type == 'uint32':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<I', k)
                            elif stream_data_type == 'uint16':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<H', k)
                            elif stream_data_type == 'uint8':
                                for k in data_array[:, i * sample_chunk_max_size + j]:
                                    values += struct.pack('<B', k)
                            else:
                                raise Exception('unknown stream type')
                            out_file.write(timestampbytes + timestamp + values)

        # write stream footers
        for stream_label, _ in buffer.items():
            footer = create_xml_string(self.stream_footers[stream_label])
            stream_footer_len = len(footer) + 2 + 4
            stream_footer_len_bytes = NumLenByte_decoder(stream_footer_len)
            stream_footer = stream_footer_len_bytes.to_bytes(1, byteorder='little') + \
                            stream_footer_len.to_bytes(stream_footer_len_bytes, byteorder='little') + \
                            XdfTag.StreamFooter.value.to_bytes(2, byteorder='little') + \
                            self.stream_footers[stream_label]['stream_id'].to_bytes(4, byteorder='little') + \
                            footer.encode('utf-8')
            out_file.write(stream_footer)

        out_file.close()

def load_xdf(filename):
    xdf_data = pyxdf.load_xdf(filename)
    dats_data = {}
    for stream_data in xdf_data[0]:
        stream_footer = stream_data['footer']
        stream_name = stream_footer['info']['stream_name'][0]
        if stream_name == 'monitor 0':
            sample_array = stream_data['time_series']
            stream_shape = eval(stream_footer['info']['frame_dimension'][0]) + (sample_array.shape[0],)
            sample_array = sample_array.T.reshape(stream_shape)
            sample_array = sample_array.astype(np.uint8)
            dats_data[stream_name] = [sample_array, stream_data['time_stamps']]
        else:
            data = [stream_data['time_series'].T, stream_data['time_stamps']]
            dats_data[stream_name] = data
    return dats_data




