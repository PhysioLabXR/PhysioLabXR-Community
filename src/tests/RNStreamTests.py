import os
import tempfile
import unittest
import warnings
import math
from random import gauss
import io
from contextlib import redirect_stdout

from src.utils.data_utils import RNStream

def create_test_stream(test_stream_length, test_stream_mean, test_stream_variance):
    temp_dir_path = tempfile.mkdtemp()
    stream = RNStream(os.path.join(temp_dir_path, 'test.dats'))
    data_ts = [gauss(test_stream_mean, math.sqrt(test_stream_variance)) for i in range(test_stream_length)]
    return stream, temp_dir_path, data_ts

class MyTestCase(unittest.TestCase):
    def test_irregular_sampling_interval(self):
        '''
        streams with timestamp's std greater than 0.1 may be a stream with irregular sampling intervals.
        If jitter removal is enabled, a RuntimeWarning should be raise to the user's notice because
        jitter removal on irregularly sampled data would trash the timestamps
        '''
        test_stream_length = 1000
        test_stream_mean = 0
        test_stream_variance = 0.15 ** 2 # a warning should be raise for any stream whose timestamp's std is greater than 0.1 second
        stream, temp_dir_path, data_ts = create_test_stream(test_stream_length, test_stream_mean, test_stream_variance)
        # order the timestamps in increasing order
        data_ts.sort()

        buffer = {}
        buffer['test_stream_name'] = [data_ts, data_ts]
        stream.stream_out(buffer)

        with warnings.catch_warnings(record=True) as w:
            buffer_in = stream.stream_in()
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "irregularly sampled streams" in str(w[-1].message)

    def test_stream_out_timestamp_mono_increasing(self):
        test_stream_length = 1000
        test_stream_mean = 0
        test_stream_variance = 0.01 ** 2 # a warning should be raise for any stream whose timestamp's std is greater than 0.1 second
        stream, temp_dir_path, data_ts = create_test_stream(test_stream_length, test_stream_mean, test_stream_variance)

        buffer = {}
        data_ts.sort(reverse=True)  # sort the timestamps in decreasing order
        buffer['test_stream_name'] = [data_ts, data_ts]

        with warnings.catch_warnings(record=True) as w:
            stream.stream_out(buffer)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "timestamps must be in increasing order." in str(w[-1].message)

    def test_stream_out_returns_total_byte_count(self):
        # TODO
        pass


    def test_stream_in_single_sample_jitter_removal(self):
        '''
        the jitter removal function in RN streams should not operate on streams
        that has fewer than two samples exclusive.
        '''
        test_stream_length = 1
        test_stream_mean = 0
        test_stream_variance = 0.01 ** 2 # a warning should be raise for any stream whose timestamp's std is greater than 0.1 second
        stream, temp_dir_path, data_ts = create_test_stream(test_stream_length, test_stream_mean, test_stream_variance)

        buffer = {}
        data_ts.sort()  # sort the timestamps in decreasing order
        buffer['test_stream_name'] = [data_ts, data_ts]

        stream.stream_out(buffer)

        # redirect the print info message
        f = io.StringIO()
        with redirect_stdout(f):
            buffer_in = stream.stream_in()
        out = f.getvalue()
        assert 'because it has fewer than two samples' in out

if __name__ == '__main__':
    unittest.main()
