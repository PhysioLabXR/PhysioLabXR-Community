import numpy as np

from utils.data_utils import RNStream


test_rns = RNStream('C:/Recordings/05_29_2021_03_03_16-Exp_myexperiment-Sbj_someone-Ssn_0.dats')
reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
                }

mmWave_data = test_rns.stream_in(reshape_stream_dict=reshape_dict)

rd = mmWave_data['TImmWave_6843AOP'][0][0]
ra = mmWave_data['TImmWave_6843AOP'][0][1]

rd = np.moveaxis(rd, -1, 0)
test = np.squeeze(rd[0], axis=-1)




