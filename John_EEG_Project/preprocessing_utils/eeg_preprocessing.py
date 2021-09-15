import os
import numpy as np
import json
from utils.data_utils import RNStream, integer_one_hot, corrupt_frame_padding, time_series_static_clutter_removal
from sklearn.preprocessing import OneHotEncoder


def load_idp_file(file_path, DataStreamName, reshape_dict, exp_info_dict_json_path, sample_num, rd_cr_ratio=None, ra_cr_ratio=None, all_categories=None):
    exp_info_dict = json.load(open(exp_info_dict_json_path))
    ExpID = exp_info_dict['ExpID']
    ExpLSLStreamName = exp_info_dict['ExpLSLStreamName']
    ExpStartMarker = exp_info_dict['ExpStartMarker']
    ExpEndMarker = exp_info_dict['ExpEndMarker']
    ExpLabelMarker = exp_info_dict['ExpLabelMarker']
    ExpInterruptMarker = exp_info_dict['ExpInterruptMarker']
    ExpErrorMarker = exp_info_dict['ExpErrorMarker']
