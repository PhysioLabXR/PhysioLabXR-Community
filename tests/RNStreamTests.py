import os
import tempfile
import unittest
import warnings
import math
from random import gauss

import numpy as np

from utils.data_utils import RNStream


class MyTestCase(unittest.TestCase):
    def test_irregular_sampling_interval(self):
        '''
        streams with timestamp's std greater than 0.1 may be a stream with irregular sampling intervals.
        If jitter removal is enabled, a RuntimeWarning should be raise to the user's notice because
        jitter removal on irregularly sampled data would trash the timestamps
        '''
        temp_dir_path = tempfile.mkdtemp()
        stream = RNStream(os.path.join(temp_dir_path, 'test.dats'))
        buffer = {}

        test_stream_length = 1000
        test_stream_mean = 0
        test_stream_variance = 0.15 ** 2  # a warning should be raise for any stream whose timestamp's std is greater than 0.1 second
        data_ts = [gauss(test_stream_mean, math.sqrt(test_stream_variance)) for i in range(test_stream_length)]

        buffer['test_stream_name'] = [data_ts, data_ts]
        stream.stream_out(buffer)

        with warnings.catch_warnings(record=True) as w:
            buffer_in = stream.stream_in()
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "irregularly sampled streams" in str(w[-1].message)

    def test_stream_in_single_sample_jitter_removal(self):
        '''
        the jitter removal function in RN streams should not operate on streams
        that has fewer than two samples exclusive.
        '''
        # create the app
        # main window init



if __name__ == '__main__':
    unittest.main()
