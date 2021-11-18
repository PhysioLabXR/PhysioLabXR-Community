import sys
from scipy.io import savemat
import os
from pathlib import Path

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from rena.utils.data_utils import RNStream



#
# test_rns = RNStream('C:/Recordings/05_24_2021_13_31_12-Exp_myexperiment-Sbj_someone-Ssn_0.dats')
# reloaded_data = test_rns.stream_in(ignore_stream=('monitor1', '0'))


def get_RNStream(file_path, ignore_stream=None, only_stream=None):
    rns_object = RNStream(file_path)
    rns_data = rns_object.stream_in(ignore_stream=ignore_stream, only_stream=only_stream)
    key_list = list(rns_data.keys())
    for key in key_list:
        new_key = key
        if new_key.isdigit():
            new_key = 'video_' + new_key
        new_key = new_key.replace('.','_')
        rns_data[new_key] = rns_data.pop(key)

    return rns_data


# n = len(sys.argv)
input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
rns_data = get_RNStream(input_path)
_, filename = os.path.split(input_path)
filename = str(filename)
filename = filename.replace('dats','mat')
output_path = os.path.join(output_path, filename)
savemat(output_path, rns_data, oned_as='row')


