import numpy as np

from utils.data_utils import RNStream


test_rns = RNStream('C:/Recordings/05_29_2021_03_03_16-Exp_myexperiment-Sbj_someone-Ssn_0.dats')
reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
                }

test_reloaded_data = test_rns.stream_in(reshape_stream_dict=reshape_dict)







