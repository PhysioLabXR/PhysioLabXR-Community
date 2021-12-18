import numpy as np

from utils.data_utils import RNStream


test_rns = RNStream('C:/Recordings/John_error_test_A-E/06_02_2021_00_44_53-Exp_John_error_test-Sbj_HW-Ssn_0.dats')
reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
                }

mmWave_data = test_rns.stream_in(reshape_stream_dict=reshape_dict, jitter_removal=False)

rd = mmWave_data['TImmWave_6843AOP'][0][0]
ra = mmWave_data['TImmWave_6843AOP'][0][1]



